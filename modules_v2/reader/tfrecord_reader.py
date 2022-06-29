"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""
import os
import logging
import random
import json
from copy import deepcopy
from functools import partial
from abc import abstractmethod
from typing import Optional, List, Iterator, Dict

import tensorflow as tf
import numpy as np

from ..config import DataItemConfigurator as DataItemCfg
from .base_reader import BaseReaderV2, DataArgumentation

# pylint: disable=unexpected-keyword-arg,no-value-for-parameter


class DefaultTFReader(BaseReaderV2):
    """
    The standard TF records reader
    """
    def __init__(self, data_dir, batch_size, num_devices, shuffle, split,
                 infinite, in_params, out_params, w_params, prefix, rel_path,
                 interleave=False, compress='', name=None, **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite, in_params,
                         out_params, w_params, name, **kwargs)
        self.prefix = prefix
        self.compress = compress
        self.interleave = interleave

        self.record_dir = os.path.join(data_dir, rel_path) if rel_path else data_dir

        self.all_params: Optional[List[DataItemCfg]] = [
            *self.in_params, *self.out_params, *self.w_params]
        self.dataset = self.create_dataset()
        self.iterator: Optional[Iterator] = iter(self.dataset)

    @abstractmethod
    def create_dataset(self) -> tf.data.Dataset:
        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir) if f.find(self.prefix) != -1 and f.endswith('records')]
        if self.shuffle:
            random.shuffle(record_files)

        tf_datasets = partial(tf.data.TFRecordDataset, compression_type=self.compress)
        if self.interleave:
            dataset = tf.data.Dataset.from_tensor_slices(record_files)
            dataset = dataset.interleave(tf_datasets,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                         deterministic=False)
        else:
            dataset = tf.data.TFRecordDataset(
                record_files, compression_type=self.compress)
        logging.info(record_files)
        proc_op = partial(self.preprocess, params=self.all_params)
        dataset = dataset.map(proc_op, tf.data.experimental.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 50)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    @staticmethod
    def map_tf_keys_to_features(params):
        def _map_feat(_type):
            _raw, _ = _type.split('-')
            if hasattr(tf.dtypes, _raw):
                return tf.io.FixedLenFeature([], _raw)
            else:
                raise NotImplementedError

        request_dict = dict()
        for a_p in params:
            if a_p.name in request_dict.keys():
                continue
            request_dict[a_p.name] = _map_feat(a_p.type)
        return request_dict

    @staticmethod
    def map_request_feat(raw_feat, tf_kay_name, params):
        tf_key = [a_p for a_p in params if tf_kay_name == a_p.name]
        if not tf_key:
            return raw_feat
        tf_key = tf_key[0]
        raw, decoded = tf_key.type.split('-')
        if raw == 'string':
            raw_decoded = tf.io.decode_raw(raw_feat, getattr(tf.dtypes, decoded)) \
                if decoded != 'img' else tf.image.decode_image(raw_feat, channels=tf_key.raw_shape[-1])
        elif hasattr(tf.dtypes, decoded):
            raw_decoded = raw_feat
        else:
            raise NotImplementedError
        if tf_key.raw_shape[0] >= 1:
            raw_decoded = tf.reshape(raw_decoded, tf_key.raw_shape)
        return raw_decoded

    @staticmethod
    def decode_raw(example_proto, params: List[DataItemCfg]):
        request_dict = __class__.map_tf_keys_to_features(params)
        all_raw_feats = tf.io.parse_single_example(example_proto, request_dict)
        all_decoded_feats = {
            r: __class__.map_request_feat(
                all_raw_feats[r],
                r, params) for r in request_dict.keys()}
        return all_decoded_feats

    @staticmethod
    def preprocess_methods(inputs, pre_str, group, group_augmentation):
        p_method, p_args = [a_[..., 0] for a_ in np.split(np.reshape(np.asarray(pre_str.split('-')), [-1, 2]), 2,
                                                          axis=-1)]
        for p_m, p_a in zip(p_method, p_args):
            if p_m == 'crop':
                r_ratio, r_round = [float(p_) for p_ in p_a.split('_')]
                inputs = group_augmentation.random_crop(
                    inputs, r_ratio, int(r_round), f'{p_m}-{r_ratio}', group)
            else:
                raise NotImplementedError
        return inputs

    @staticmethod
    def preprocess(example_proto, params: List[DataItemCfg]):
        decoded_feats = __class__.decode_raw(example_proto, params)
        processed_feats = list()
        group_augmentation = DataArgumentation(with_batch=False)
        for a_p in params:
            inputs = decoded_feats[a_p.name]
            if a_p.preprocess:
                inputs = __class__.preprocess_methods(
                    inputs, a_p.preprocess, a_p.process_group, group_augmentation)
            processed_feats.append(inputs)
        return processed_feats

    def postprocess_methods(self, inputs: tf.Tensor, post_str: List[str], group: str,
        group_augmentation: str) -> tf.Tensor:
        """
        Post processing methods applied to inputs

        Args:
            inputs (tf.Tensor): the input data
            post_str (List[str]): the post-processing string
            group (str): the group to be appilied random augmentation
            group_augmentation (str): the group name

        Returns:
            tf.Tensor: the data after post-processing
        """
        x = inputs
        for p_str in post_str:
            x = eval(p_str)
        return x

    @staticmethod
    def dequantization(data: tf.Tensor, max_v: tf.Tensor, min_v: tf.Tensor):
        """
            Restore the quantized value
        """
        assert data.dtype == tf.uint8
        with tf.name_scope("Dequentization"):
            data = tf.cast(data, max_v.dtype) / data.dtype.max
            data = data * (max_v - min_v) + min_v
        return data

    # @tf.function
    def postprocess(self, inputs):
        """
            Post-process will cook data in GPU pipeline
        """
        list_inputs, dict_inputs = inputs
        devices_inputs = dict()
        group_augs = list()
        for g_a in range(self.num_devices):
            group_augs.append(DataArgumentation(with_batch=True))
        for param, in_data in zip(self.all_params, list_inputs):
            device_inputs = list()
            gpus_data = tf.split(in_data, self.num_devices, 0)
            if param.quantized:
                gpus_max_qt = tf.split(dict_inputs[f'{param.name}_max'], self.num_devices, 0)
                gpus_min_qt = tf.split(dict_inputs[f'{param.name}_min'], self.num_devices, 0)
            else:
                gpus_max_qt = [[]] * self.num_devices
                gpus_min_qt = [[]] * self.num_devices
            zip_iter = (range(self.num_devices), gpus_data, group_augs, gpus_max_qt, gpus_min_qt)
            for g_id, g_data, g_a, g_max, g_min in zip(*zip_iter):
                with tf.device(f'/gpu:{g_id}'):
                    if param.quantized:
                        g_data = __class__.dequantization(g_data, g_max, g_min)
                    if param.postprocess:
                        g_data = self.postprocess_methods(
                            g_data, param.postprocess, param.process_group, g_a)
                    device_inputs.append(g_data)
            devices_inputs[param.name] = device_inputs
        return devices_inputs

    def _reorganize_io_data(self, elements):
        del self
        return elements, dict()

    def _reorganize_next_data(self, elements):
        def _assamble_term(_params: List[DataItemCfg]):
            _term = dict() if self.dict_mode else list()
            for _p in _params:
                if self.dict_mode:
                    _term[_p.name] = elements[_p.name]
                else:
                    _term.append(elements[_p.name])
            return _term
        return dict(
            inputs=_assamble_term(self.in_params),
            outputs=_assamble_term(self.out_params),
            weights=_assamble_term(self.w_params)
        )

    def next(self):
        while True:
            try:
                next_elem = self._reorganize_io_data(next(self.iterator))
                next_elem = self.postprocess(next_elem)
                next_elem_dict = self._reorganize_next_data(next_elem)
                return next_elem_dict
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                raise StopIteration  # pylint: disable=raise-missing-from


class DictTFReader(DefaultTFReader):
    """
    The dictory layout TF records
    """
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_devices,
                 shuffle,
                 split,
                 infinite,
                 in_params,
                 out_params,
                 w_params,
                 prefix,
                 rel_path,
                 interleave=False,
                 compress='',
                 name=None,
                 **kwargs):
        self.used_rk_pirs: Dict[str, List] = dict()
        super().__init__(data_dir, batch_size, num_devices, shuffle, split,
                         infinite, in_params, out_params, w_params, prefix, rel_path,
                         interleave=interleave, compress=compress, name=name, **kwargs)

    def create_dataset(self) -> tf.data.Dataset:  # pylint: disable=too-many-locals
        """
        Create the hybird dataset from different keys existed in multiple tfrecords
        """
        json_path = os.path.join(self.record_dir, self.prefix)
        with open(json_path) as fp:  # pylint: disable=invalid-name
            records_dict: Dict = json.load(fp)

        def _attach_extra_item(_name, _alias, _group):
            _item = DataItemCfg()
            _item.name = f'{_name}_{_alias}'
            _item_desc = _group[_item.name]
            _item.type = _item_desc['type']
            _item.raw_shape = _item_desc['shape']
            return _item

        used_rk_pairs: Dict[str, List] = dict()
        for r_key, r_value in records_dict.items():
            for p_name, p_attrs in r_value.items():
                for p_used in self.all_params:
                    if p_used.name != p_name:
                        continue
                    item = deepcopy(p_used)
                    item.raw_shape = p_attrs['shape']
                    item.type = p_attrs['type']
                    p_attrs.setdefault('quantized', 0)
                    item.quantized = p_attrs['quantized']
                    p_used.quantized = item.quantized
                    if r_key not in used_rk_pairs.keys():
                        used_rk_pairs[r_key] = list()
                    used_rk_pairs[r_key].append(item)
                    if item.quantized:
                        used_rk_pairs[r_key].append(_attach_extra_item(item.name, 'max', r_value))
                        used_rk_pairs[r_key].append(_attach_extra_item(item.name, 'min', r_value))
                    break
        self.used_rk_pirs = used_rk_pairs

        records_files = sorted([f_ for f_ in os.listdir(self.record_dir) if
                            os.path.splitext(f_)[-1] == '.records'])
        datasets = list()
        for u_key, u_value in used_rk_pairs.items():
            tf_datasets = partial(tf.data.TFRecordDataset, compression_type=self.compress)

            selected_files = [os.path.join(self.record_dir, f_) for f_ in records_files
                                if f_.startswith(u_key)]
            logging.info(selected_files)
            if self.interleave:
                dataset = tf.data.Dataset.from_tensor_slices(selected_files)
                dataset = dataset.interleave(tf_datasets, deterministic=True)
            else:
                dataset: tf.data.Dataset = tf_datasets(selected_files)
            proc_op = partial(self.preprocess, params=u_value)
            dataset = dataset.map(proc_op, tf.data.experimental.AUTOTUNE)
            datasets.append(dataset)
        packed_dataset = tf.data.Dataset.zip(tuple(datasets))
        if self.shuffle:
            packed_dataset = packed_dataset.shuffle(self.batch_size * 50)
        packed_dataset = packed_dataset.batch(self.batch_size, drop_remainder=True)
        return packed_dataset

    def enable_dict_mode(self):
        self.dict_mode = True

    def _reorganize_io_data(self, elements):
        io_list = list()
        elem_dict = dict()
        for u_id, u_pairs in enumerate(self.used_rk_pirs.values()):
            for u_ele, u_pair in zip(elements[u_id], u_pairs):
                elem_dict[u_pair.name] = u_ele
        for param in self.all_params:
            io_list.append(elem_dict[param.name])
        return io_list, elem_dict

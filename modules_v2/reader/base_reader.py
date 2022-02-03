"""
Copyright (c) Microsoft Corporation.
"""
import json

from abc import abstractmethod
import logging
from typing import Optional, List, Dict, Tuple

import tensorflow as tf

from ..config import DataItemConfigurator as DataItemCfg


class DataArgumentation(object):
    def __init__(self, with_batch=True):
        self.data_argument_dict: Dict[str, tf.Tensor] = {}
        self.with_batch = with_batch

    def get_rand_int(self, inputs, random=None):
        if random is None:
            random = tf.random.uniform(
                [inputs.shape.dims[0].value] if self.with_batch else [], dtype=tf.float32)
        rand = tf.cast(tf.round(random), dtype=inputs.dtype)
        rand = tf.reshape(rand, [-1, *[1] * (inputs.shape.ndims - 1)])
        return rand

    @staticmethod
    def get_rand_float(rand_shape, random=None):
        if random is None:
            random = tf.random.uniform(rand_shape, dtype=tf.float32)
        return random

    def update_or_get_from_dict(self, name, group, func, func_args):
        if group == 0:
            return func(*func_args)
        req_name = f'{name}-{group}'
        if req_name not in self.data_argument_dict.keys():
            self.data_argument_dict[req_name] = func(*func_args)
        return self.data_argument_dict[req_name]

    def axis_unary_aug(self, tf_op, inputs: tf.Tensor, axis, name, group):
        rand = self.update_or_get_from_dict(
            name, group, self.get_rand_int, (inputs,))
        rand = tf.cast(rand, dtype=inputs.dtype)
        r_inputs_ = tf_op(inputs, axis)
        return inputs * rand + r_inputs_ * (1 - rand)

    def random_flip(self, inputs: tf.Tensor, axis, name, group):
        return self.axis_unary_aug(tf.reverse, inputs, [axis], name, group)

    def random_transpose(self, inputs: tf.Tensor, axis, name, group):
        return self.axis_unary_aug(tf.transpose, inputs, axis, name, group)

    def random_crop(self, inputs: tf.Tensor, crop_ratio,
                    crop_round, name, group):
        assert self.with_batch is False
        rand = self.update_or_get_from_dict(
            name, group, self.get_rand_float, ([2], ))
        raw_size = tf.shape(inputs)[:-1]
        crop_size = tf.cast(
            tf.cast(
                raw_size,
                tf.float32) *
            crop_ratio,
            raw_size.dtype)
        crop_range = raw_size - crop_size
        crop_pos = tf.cast(
            tf.cast(
                crop_range //
                crop_round,
                tf.float32) *
            rand,
            raw_size.dtype)
        crop_pos = crop_pos * crop_round
        return tf.slice(inputs, [crop_pos[0], crop_pos[1], 0], [
                        crop_size[0], crop_size[1], -1])


class BaseReaderV2(object):
    def __init__(self,
                 batch_size,
                 num_devices,
                 shuffle,
                 split,
                 infinite,
                 in_params,
                 out_params,
                 w_params,
                 name=None,
                 **kwargs):
        del kwargs
        self.batch_size = batch_size * num_devices if not split else num_devices
        self.infinite = infinite
        self.num_devices = num_devices
        self.shuffle = shuffle
        self.name = name

        self.dict_mode = False

        self.in_params: Optional[List[DataItemCfg]] = in_params
        self.out_params: Optional[List[DataItemCfg]] = out_params
        self.w_params: Optional[List[DataItemCfg]] = w_params

    @abstractmethod
    def next(self):
        """
        Fill the data for the next training
        """

    def post(self, post_inputs, **kwargs):
        """
        The post method after an iteration finished. Currently, only recursive
            reader is dependended on this feature.
        """

    def enable_dict_mode(self):
        """
        Dict mode will return the next data organized as the dictory
        """
        logging.error('Dict mode is default disable unless the reader explicit supports')
        raise NotImplementedError

    @staticmethod
    def parse_keys_desc_from_json(json_path) -> Tuple[Dict, Dict]:
        """
        Parse the key-descs pairs
        """
        with open(json_path) as fp:  # pylint: disable=invalid-name
            files_dict: Dict = json.load(fp)

        key_attrs = dict()
        key_map = dict()
        for f_key, f_value in files_dict.items():
            for p_name, p_attrs in f_value.items():
                key_attrs[p_name] = p_attrs
                key_map[p_name] = f_key
        return key_attrs, key_map

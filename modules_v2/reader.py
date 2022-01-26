import os
import tensorflow as tf
import numpy as np
import logging

from abc import abstractmethod
from typing import Iterator, Optional, List, Dict

from tensorflow.python.keras.backend import dtype
from .config import DataItemConfigurator as DataItemCfg


class DataArgumentation(object):
    def __init__(self, with_batch=True):
        self.data_argument_dict: Dict[str, tf.Tensor] = {}
        self.with_batch = with_batch

    def get_rand_int(self, inputs, random=None):
        if random is None:
            random = tf.random.uniform([inputs.shape.dims[0].value] if self.with_batch else [], dtype=tf.float32)
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
        rand = self.update_or_get_from_dict(name, group, self.get_rand_int, (inputs,))
        rand = tf.cast(rand, dtype=inputs.dtype)
        r_inputs_ = tf_op(inputs, axis)
        return inputs * rand + r_inputs_ * (1 - rand)

    def random_flip(self, inputs: tf.Tensor, axis, name, group):
        return self.axis_unary_aug(tf.reverse, inputs, [axis], name, group)

    def random_transpose(self, inputs: tf.Tensor, axis, name, group):
        return self.axis_unary_aug(tf.transpose, inputs, axis, name, group)

    def random_crop(self, inputs: tf.Tensor, crop_ratio, crop_round, name, group):
        assert self.with_batch is False
        rand = self.update_or_get_from_dict(name, group, self.get_rand_float, ([2], ))
        raw_size = tf.shape(inputs)[:-1]
        crop_size = tf.cast(tf.cast(raw_size, tf.float32) * crop_ratio, raw_size.dtype)
        crop_range = raw_size - crop_size
        crop_pos = tf.cast(tf.cast(crop_range // crop_round, tf.float32) * rand, raw_size.dtype)
        crop_pos = crop_pos * crop_round
        return tf.slice(inputs, [crop_pos[0], crop_pos[1], 0], [crop_size[0], crop_size[1], -1])


class BaseReaderV2(object):
    def __init__(self, batch_size, num_devices, shuffle, split, infinite, in_params, out_params, w_params, name=None,
                 **kwargs):
        self.batch_size = batch_size * num_devices if not split else num_devices
        self.infinite = infinite
        self.num_devices = num_devices
        self.shuffle = shuffle
        self.name = name

        self.in_params: Optional[List[DataItemCfg]] = in_params
        self.out_params: Optional[List[DataItemCfg]] = out_params
        self.w_params: Optional[List[DataItemCfg]] = w_params

    @abstractmethod
    def next(self): pass

    def post(self, post_inputs, **kwargs):
        """
        The post method after an iteration finished. Currently, only recursive
            reader is dependended on this feature.
        """
        pass


class DefaultTFReader(BaseReaderV2):
    def __init__(self, data_dir, batch_size, num_devices, shuffle, split, infinite, in_params, out_params, w_params,
                 prefix, rel_path, compress='', name=None, **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite, in_params, out_params, w_params, name,
                         **kwargs)
        self.record_dir = os.path.join(data_dir, rel_path) if rel_path else data_dir

        self.all_params: Optional[List[DataItemCfg]] = [*self.in_params, *self.out_params, *self.w_params]

        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir) if f.find(prefix) != -1
                        and f.endswith('records')]
        logging.info(record_files)
        dataset = tf.data.TFRecordDataset(record_files, compression_type=compress)
        dataset = dataset.map(self.preprocess, tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(self.batch_size * 50)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = dataset
        self.iterator: Optional[Iterator] = iter(self.dataset)

    def map_tf_keys_to_features(self):
        def _map_feat(_type):
            _raw, _ = _type.split('-')
            if hasattr(tf.dtypes, _raw):
                return tf.io.FixedLenFeature([], _raw)
            else:
                raise NotImplementedError

        request_dict = dict()
        for a_p in self.all_params:
            if a_p.name in request_dict.keys():
                continue
            request_dict[a_p.name] = _map_feat(a_p.type)
        return request_dict

    def map_request_feat(self, raw_feat, tf_kay_name):
        tf_key = [a_p for a_p in self.all_params if tf_kay_name == a_p.name]
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
        elif tf_key.raw_shape[0] == 0:
            raw_decoded = raw_decoded
        return raw_decoded

    def decode_raw(self, example_proto):
        request_dict = self.map_tf_keys_to_features()
        all_raw_feats = tf.io.parse_single_example(example_proto, request_dict)
        all_decoded_feats = {r: self.map_request_feat(all_raw_feats[r], r) for r in request_dict.keys()}
        return all_decoded_feats

    def preprocess_methods(self, inputs, pre_str, group, group_augmentation):
        p_method, p_args = [a_[..., 0] for a_ in np.split(np.reshape(np.asarray(pre_str.split('-')), [-1, 2]), 2,
                                                          axis=-1)]
        for p_m, p_a in zip(p_method, p_args):
            if p_m == 'crop':
                r_ratio, r_round = [float(p_) for p_ in p_a.split('_')]
                inputs = group_augmentation.random_crop(inputs, r_ratio, int(r_round), f'{p_m}-{r_ratio}', group)
            else:
                raise NotImplementedError
        return inputs

    def preprocess(self, example_proto):
        decoded_feats = self.decode_raw(example_proto)
        processed_feats = list()
        group_augmentation = DataArgumentation(with_batch=False)
        for a_p in self.all_params:
            inputs = decoded_feats[a_p.name]
            if a_p.preprocess:
                inputs = self.preprocess_methods(inputs, a_p.preprocess, a_p.process_group, group_augmentation)
            processed_feats.append(inputs)
        return processed_feats

    def postprocess_methods(self, inputs, post_str, group, group_augmentation):
        p_method, p_args = [a_[..., 0] for a_ in np.split(np.reshape(np.asarray(post_str.split('-')), [-1, 2]), 2,
                                                          axis=-1)]
        for p_m, p_a in zip(p_method, p_args):
            if p_m == 'flip':
                f_axis = [a_ + 1 for a_ in range(5) if 2 ** a_ & int(p_a)]
                for a_ in f_axis:
                    inputs = group_augmentation.random_flip(inputs, a_, f'{p_m}-{a_}', group)
            elif p_m == 'transpose':
                f_axis = [int(a_) for a_ in p_a]
                inputs = group_augmentation.random_transpose(inputs, f_axis, f'{p_m}-{f_axis}', group)
            elif p_m == 'reshape':
                inputs = tf.reshape(inputs, (-1, int(p_a)) + tuple(inputs.shape.dims[1:]))
            elif p_m == 'oh':
                inputs = tf.one_hot(inputs, int(p_a), dtype=tf.float32)
            elif p_m == 'dim':
                inputs = tf.reshape(inputs, tuple(inputs.shape.dims) + (1,) * (int(p_a) - inputs.shape.ndims + 1))
            elif p_m == 'cast':
                inputs = tf.cast(inputs, getattr(tf.dtypes, p_a))
            elif p_m == 'divide':
                inputs = inputs / float(p_a)
            elif p_m == 'scale':
                inputs = inputs * float(p_a)
            elif p_m == 'min':
                inputs = tf.maximum(inputs, tf.convert_to_tensor(float(p_a), dtype=inputs.dtype))
            elif p_m == 'max':
                inputs = tf.minimum(inputs, tf.convert_to_tensor(float(p_a), dtype=inputs.dtype))
            elif p_m == 'pad':
                axis, pad_size = [int(a_) for a_ in p_a.split('_')]
                pad_array = [[0, 0]] * inputs.shape.ndims
                pad_array[axis + 1] = [pad_size, pad_size]
                inputs = tf.pad(inputs, pad_array)
            elif p_m == 'blur':
                x_points = np.arange(-(int(p_a) - 1) // 2, (int(p_a) - 1) // 2 + 1, 1)
                xs, ys = np.meshgrid(x_points, x_points, indexing='ij')
                kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * 1 ** 2)) / (2 * np.pi * 1 ** 2)
                kernel = (kernel / kernel.sum())[..., np.newaxis, np.newaxis]
                kernel = tf.tile(tf.convert_to_tensor(kernel, tf.float32), [1, 1, tf.shape(inputs)[-1], 1])
                inputs = tf.nn.depthwise_conv2d(inputs, kernel, strides=(1, 1, 1, 1), padding='SAME')
            else:
                raise NotImplementedError
        return inputs

    @tf.function
    def postprocess(self, inputs):
        devices_inputs = list()
        group_augmentation = list()
        for g_a in range(self.num_devices):
            group_augmentation.append(DataArgumentation(with_batch=True))
        for p, i in zip(self.all_params, inputs):
            device_inputs = list()
            gpu_i = tf.split(i, self.num_devices, axis=0)
            for g, g_i, g_a in zip(range(self.num_devices), gpu_i, group_augmentation):
                with tf.device(f'/gpu:{g}'):
                    if p.postprocess:
                        g_i = self.postprocess_methods(g_i, p.postprocess, p.process_group, g_a)
                    device_inputs.append(g_i)
            devices_inputs.append(device_inputs)
        return devices_inputs

    def next(self):
        while True:
            try:
                next_elem = next(self.iterator)
                next_elem = self.postprocess(next_elem)
                in_off = len(self.in_params)
                out_off = in_off + len(self.out_params)
                w_off = out_off + len(self.w_params)
                next_elem_dict = dict(inputs=next_elem[:in_off], outputs=next_elem[in_off: out_off],
                                      weights=next_elem[out_off:w_off], alias=next_elem[w_off:])
                return next_elem_dict
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                else:
                    raise StopIteration


class RandomReader(BaseReaderV2):
    def __init__(self, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params, w_params,
                 name=None,
                 **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite, in_params, out_params, w_params, name,
                         **kwargs)
        self.deterministic = None
        self.cur_samples = 0
        self.num_samples = num_samples

    def next_stochastic(self):
        in_elem = list()
        for i_p in self.in_params:
            all_data = tf.split(tf.random.normal([self.batch_size, *i_p.raw_shape], dtype=tf.float32),
                                self.num_devices)
            in_elem.append(all_data)
        return dict(inputs=in_elem, outputs=list(), weights=list(), alias=list())

    def next_deterministic(self):
        if self.deterministic is None:
            self.deterministic = list()
            for i_p in self.in_params:
                rand_nd = np.random.normal(size=[self.num_samples, *i_p.raw_shape]).astype(np.float32)
                self.deterministic.append(rand_nd)
        try:
            in_elem = list()
            if self.cur_samples > self.num_samples - self.batch_size:
                raise StopIteration
            for d in self.deterministic:
                all_data = tf.split(d[self.cur_samples: self.cur_samples + self.batch_size], self.num_devices, axis=0)
                in_elem.append(all_data)
            self.cur_samples += self.batch_size
            return dict(inputs=in_elem, outputs=list(), weights=list(), alias=list())
        except StopIteration:
            self.cur_samples = 0
            raise StopIteration

    def next(self):
        try:
            if self.shuffle:
                return self.next_stochastic()
            else:
                return self.next_deterministic()
        except StopIteration:
            if self.infinite:
                return self.next()
            else:
                raise StopIteration


class RecursiveReader(BaseReaderV2):
    def __init__(self, batch_size, num_devices, shuffle, split,
                 infinite, in_params, out_params, w_params, name, **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite,
                         in_params, out_params, w_params, name=name, **kwargs)

        self.inputs_holder: List[tf.Variable] = list()
        assert not self.out_params
        assert not self.w_params
        for in_param in self.in_params:
            in_var = tf.Variable(np.zeros([self.batch_size, *in_param.raw_shape]),
                                 name=f'{self.name}-{in_param.name}',
                                 dtype=getattr(tf.dtypes, in_param.type))
            self.inputs_holder.append(in_var)

    def next(self):
        return dict(inputs=[self.inputs_holder],
                    outputs=list(),
                    weights=list(),
                    alias=list())

    def post(self, post_inputs, **kwargs):
        if not isinstance(post_inputs, (list, tuple)):
            post_inputs = [post_inputs]
        assert len(post_inputs) >= len(self.inputs_holder)
        for p_var, p_input in zip(self.inputs_holder,
                                  post_inputs[:len(self.inputs_holder)]):
            p_var.assign(p_input, use_locking=True)

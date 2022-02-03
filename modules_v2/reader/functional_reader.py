"""
Copyright (c) Microsoft Corporation.
"""
import os

from typing import List

import tensorflow as tf
import numpy as np

from .base_reader import BaseReaderV2

# pylint: disable=unexpected-keyword-arg,no-value-for-parameter


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
        return dict(inputs=in_elem, outputs=list(),
                    weights=list(), alias=list())

    def next_deterministic(self):
        if self.deterministic is None:
            self.deterministic = list()
            for i_p in self.in_params:
                rand_nd = np.random.normal(
                    size=[
                        self.num_samples,
                        *
                        i_p.raw_shape]).astype(
                    np.float32)
                self.deterministic.append(rand_nd)
        try:
            in_elem = list()
            if self.cur_samples > self.num_samples - self.batch_size:
                raise StopIteration
            for d in self.deterministic:
                all_data = tf.split(
                    d[self.cur_samples: self.cur_samples + self.batch_size], self.num_devices, axis=0)
                in_elem.append(all_data)
            self.cur_samples += self.batch_size
            return dict(inputs=in_elem, outputs=list(),
                        weights=list(), alias=list())
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
    """
    Recursive reader will dump the output from the previous forward and feed into 
        next forward iteration as the inputs.
    """
    def __init__(self, data_dir, batch_size, num_devices, shuffle, split,
                 infinite, in_params, out_params, w_params, prefix, rel_path,
                 name, **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite,
                         in_params, out_params, w_params, name=name, **kwargs)

        data_dir = os.path.join(data_dir, rel_path) if rel_path else data_dir
        json_path = os.path.join(data_dir, prefix)
        key_descs, _ = __class__.parse_keys_desc_from_json(json_path)

        self.inputs_holder: List[tf.Variable] = list()
        assert not self.out_params
        assert not self.w_params
        for in_param in self.in_params:
            assert in_param.name in key_descs.keys()
            in_desc = key_descs[in_param.name]
            in_init = np.zeros([self.batch_size, *in_desc['shape']])
            in_name = f'{self.name}-{in_param.name}'
            in_var = tf.Variable(in_init, name=in_name, dtype=getattr(tf.dtypes, in_desc['type']))
            self.inputs_holder.append(in_var)

    def enable_dict_mode(self):
        self.dict_mode = True

    def next(self):
        if self.dict_mode:
            inputs_data = {_k.name: [_i] for _k, _i in zip(self.in_params, self.inputs_holder)}
        else:
            inputs_data = [[_i] for _i in self.inputs_holder]
        return dict(inputs=inputs_data,
                    outputs=dict() if self.dict_mode else list(),
                    weights=dict() if self.dict_mode else list(),
                    alias=dict() if self.dict_mode else list())

    def post(self, post_inputs, **kwargs):
        for p_idx, p_holder in enumerate(self.inputs_holder):
            p_data = list()
            for p_input in post_inputs:
                alias_name = self.in_params[p_idx].name
                p_data.append(p_input[alias_name] if self.dict_mode else p_input[p_idx])
            p_data = tf.concat(p_data, axis=0)
            p_holder.assign(p_data, use_locking=True)

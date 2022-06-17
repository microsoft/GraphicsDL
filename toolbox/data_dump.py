# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-module-docstring,too-few-public-methods,no-member

import os
from typing import Callable, Dict, List, Type, Optional
from enum import Enum

import tensorflow as tf
import numpy as np

from ..utils.design_pattern import Singleton
from ..utils.log import LoggingOnce


class DumpType(Enum):
    """
    The dump type of target tensor
    """
    IMAGE_RGB = 0
    IMAGE_GREY = 1
    ARRAY_NPY = 2


class ExtraAttr(Enum):
    """
    The extra information attached to the dump data
    """
    MEAN = 0
    STD = 1


class DumpItem:
    """
    The attributes of the dump item
    """
    def __init__(self, dump_type, extra_attrs) -> None:
        self.dump_type: DumpType = dump_type
        self.extra_attrs: List[ExtraAttr] = extra_attrs


class DataDumpLocal:
    """
    DataDump will control all the internal dump behavior during network
        training/inference.
    """
    def __init__(self) -> None:
        LoggingOnce()('Detect DataDump module enable')
        self._enable: bool = False
        self._iter: int = 0
        self._root_dir: str = ''
        self._register_items: Dict[str, DumpItem] = dict()

        self._dump_stride = 1
        self._default_rgb_mapping = lambda x: x

    def register(self, key: str, dump_desc: DumpItem):
        """
        Register the dump data
        """
        self._register_items[key] = dump_desc

    def update(self, key: str, data_tensor: tf.Tensor, map_func: Optional[Callable] = None) \
            -> None:
        """
        Update the dump data
        """
        if not tf.executing_eagerly():
            LoggingOnce()(f'Please enable eagerly executing to dump {key}')
            return
        if (self._iter % self._dump_stride != 0 and self._iter > 20) or not self._enable:
            return
        used_method = self._register_items[key].dump_type.name.lower()
        key = key.replace('/', '-').replace('\\', '-')
        getattr(self, f'_dump_{used_method}')(key, data_tensor.numpy(), map_func)

    def increment(self, inc: int = 1):
        """
        Increment the global iteration index
        """
        self._iter += inc

    def reset(self):
        """
        Reset the global dictory index
        """
        self._iter = 0

    def assign_rgb_mapping_func(self, map_func: Callable):
        """
        Assign the color mapping function
        """
        self._default_rgb_mapping = map_func

    def assign_dump_stride(self, stride: int):
        """
        Assign the dump stride
        """
        self._dump_stride = stride

    def assign_enable(self):
        """
        Enable the dump module
        """
        self._enable = True

    def assign_root_dir(self, root_dir):
        """
        The root folder to save all results
        """
        self._root_dir = root_dir

    def _dump_image_rgb(self, key: str, data_np: np.ndarray, map_func: Optional[Callable]):
        saving_dir = os.path.join(self._root_dir, key)
        os.makedirs(saving_dir, exist_ok=True)
        if data_np.ndim == 3:
            data_np = data_np[np.newaxis, ...]
        for b_idx, b_data_np in enumerate(data_np):
            map_func = map_func if map_func is not None else self._default_rgb_mapping
            data_np = map_func(b_data_np)

            attached_fonts = str()
            for extra_attr in self._register_items[key].extra_attrs:
                stat_value = getattr(np, extra_attr.name.lower())(data_np)
                attached_fonts += f'{extra_attr.name}: {stat_value}\n'

            data_np = np.clip(data_np * 255, 0, 255).astype(np.uint8)
            data_alias = f'{key}_B{b_idx}_I{self._iter}.png'
            data_path = os.path.join(saving_dir, data_alias)

            # if attached_fonts:
            #     f_pos = (50, 50)
            #     f_color = (0, 0, 255)
            #     f_style = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(data_np, attached_fonts, f_pos, f_style, 0.5, f_color)
            # cv2.imwrite(data_path, data_np[..., ::-1])  # Change BGR to RGB

    def _dump_image_grey(self, key: str, data_np: np.ndarray, map_func: Callable):
        del map_func
        if data_np.shape[-1] != 1:
            data_np = data_np[..., np.newaxis]
        data_np = np.repeat(data_np, 3, axis=-1)
        return self._dump_image_rgb(key, data_np, lambda x: x)

    def _dump_array_npy(self, key: str, data_np: np.ndarray, map_func: Callable):
        del map_func
        saving_dir = os.path.join(self._root_dir, key)
        os.makedirs(saving_dir, exist_ok=True)
        data_alias = f'{key}_I{self._iter}.npy'
        np.save(os.path.join(saving_dir, data_alias), data_np)


DataDump: Type[DataDumpLocal] = Singleton(DataDumpLocal)

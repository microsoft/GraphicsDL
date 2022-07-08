"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""
import os
import pickle
from abc import abstractmethod

from graphics_dl.utils import log as gtf_log


class BasicCallback:
    """
    Callback function will be called during training/test phase with the forwared
        results as the inputs.
    """
    def __init__(self, result_root, dataset_name, compress, eval_keys) -> None:
        self.callback_root = result_root
        self.dataset_name = dataset_name
        self.compress = compress
        self.eval_keys = eval_keys

        self.eval_root = str()
        self.cache_stat = str()

        self.iter_idx = 0
        self.epoch = 0

    def initialize(self, epoch, *args, **kwargs) -> bool:
        """
        Returns:
            bool: whether the callback initializes success or not
        """
        del args, kwargs
        self.eval_root = os.path.join(self.callback_root, str(epoch), 'eval')
        gtf_log.LogOnce(f'Evaluated data saved in {self.eval_root}')
        self.cache_stat = os.path.join(self.eval_root, 'cached.pkl')
        os.makedirs(self.eval_root, exist_ok=True)

        if os.path.exists(self.cache_stat):
            with open(self.cache_stat, 'rb') as c_fp:
                cache_stat: set = pickle.load(c_fp)
            all_queries = [f'{self.epoch}-{self.dataset_name}-{_k}' for _k in\
                self.eval_keys]
            for query in all_queries:
                if query not in cache_stat:
                    self.iter_idx = 0
                    self.epoch = epoch
                    return True
            return False
        return True

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        The update function will be called in every iteration
        """

    def close(self, *args, **kwargs) -> None:
        """
        The close function will be called while forwards all data
        """
        del args, kwargs
        if os.path.exists(self.cache_stat):
            with open(self.cache_stat, 'rb') as c_fp:
                cache_stat: set = pickle.load(c_fp)
        else:
            cache_stat = set()
        for e_key in self.eval_keys:
            cache_stat.add(f'{self.epoch}-{self.dataset_name}-{e_key}')
        with open(self.cache_stat, 'wb') as c_fp:
            pickle.dump(cache_stat, c_fp)
        return None

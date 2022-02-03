"""
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
"""
import os
from abc import abstractmethod

class BasicCallback:
    """
    Callback function will be called during training/test phase with the forwared
        results as the inputs.
    """
    def __init__(self, result_dir, compress, eval_keys) -> None:
        self.callback_root = result_dir
        self.compress = compress
        self.eval_keys = eval_keys
        self.train_root = os.path.join(self.callback_root, 'train')
        os.makedirs(self.train_root, exist_ok=True)
        self.eval_root = os.path.join(self.callback_root, 'eval')
        os.makedirs(self.eval_root, exist_ok=True)
        self.metric_root = os.path.join(self.callback_root, 'metric')
        os.makedirs(self.metric_root, exist_ok=True)
        self.mediate_root = os.path.join(self.callback_root, 'mediate')
        os.makedirs(self.mediate_root, exist_ok=True)

    @abstractmethod
    def initialize(self, *args, **kwargs) -> bool:
        """
        Returns:
            bool: whether the callback initializes success or not
        """
        return False

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        The update function will be called in every iteration
        """

    @abstractmethod
    def close(self, *args, **kwargs) -> None:
        """
        The close function will be called while forwards all data
        """

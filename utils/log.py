# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-module-docstring,too-few-public-methods

import logging
from typing import Set

from .design_pattern import Singleton


@Singleton
class LoggingOnce:
    """
    Once print the log_str once
    """
    def __init__(self) -> None:
        self.log_dict: Set = set()

    def __call__(self, log_str):
        if log_str not in self.log_dict:
            self.log_dict.add(log_str)
            logging.info(log_str)


LogOnce = LoggingOnce()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-module-docstring,too-few-public-methods


class Singleton:
    """
    The singleton pattern
    """
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]

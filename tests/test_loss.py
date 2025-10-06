"""test_loss.py

Smoke import for loss module placeholder.
"""

import importlib


def test_module_docstring_present():
    mod = importlib.import_module("modules.loss")
    assert mod.__doc__ is not None

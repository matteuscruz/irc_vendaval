"""
Pytest configuration and shared fixtures.

Mocks modules that are imported by src/models/nn_model.py but do not yet
exist in the repository (src.cnn_emos, src.loading_data, etc.).
These stubs must be inserted into sys.modules before any test file imports
src.models.nn_model.
"""
import sys
from unittest.mock import MagicMock

_MISSING_MODULES = [
    "src.cnn_emos",
    "src.cnn_emos.nn_distributions",
    "src.cnn_emos.nn_forecast",
    "src.loading_data",
    "src.loading_data.get_data",
]

for _mod in _MISSING_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

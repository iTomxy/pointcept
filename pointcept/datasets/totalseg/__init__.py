"""TotalSegmentator support code: the vendored class map plus its derived constants.

Deliberately **not** re-exported from ``datasets/__init__.py``: that module is
the training import path, and no training run should pull a 1000-line class
dictionary in order to read a cached npz.
"""

from .classes import TOTALSEG_CLASSES, TOTALSEG_RIB_CLASSES

__all__ = [
    "TOTALSEG_CLASSES",
    "TOTALSEG_RIB_CLASSES",
]

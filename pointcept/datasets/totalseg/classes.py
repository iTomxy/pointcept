"""TotalSegmentator class ids this repo uses, derived from ``map_to_binary.py``.

Only the v2 ``total`` class set is used here, so these are plain constants
rather than anything parameterised over subtask.

``map_to_binary.py`` is vendored verbatim and re-downloadable (see
``download_class_map.sh``), which is why these derived constants live
beside it rather than being edited into it: any edit there would be lost on the
next re-download.
"""

from __future__ import annotations

from .map_to_binary import class_map

# The v2 ``total`` class set, 117 classes. This repo only uses v2 ``total``:
# the predictions under ``~/data/ribseg/totalseg_pred/raw/`` were produced with
# ``task="total"``.
TOTALSEG_CLASSES: dict[int, str] = class_map["total"]

# ``rib_left_1..12`` then ``rib_right_1..12``, ids 92..115 -- ordered by id, as
# the map numbers them, not grouped by side. Derived by name prefix rather than
# written as ``tuple(range(92, 116))`` so it stays honest against the vendored
# file, which is re-downloadable.
TOTALSEG_RIB_CLASSES: tuple[int, ...] = tuple(
    sorted(id_ for id_, name in TOTALSEG_CLASSES.items() if name.startswith("rib_"))
)

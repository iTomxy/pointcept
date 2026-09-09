"""The "+totalseg_pred" task, end to end: cache, summary, and label spaces.

This module owns the "+totalseg_pred" line of work: building the all-bones
cache from an existing point-cloud cache (``--patch-totalseg``), summarising
which TotalSegmentator classes actually occur in it (``--summarise-totalseg``),
and two label spaces plus ``CombineBoneLabel`` that a config applies online.

No single source labels every bone: RibSegv2 only annotates ribs, manually,
while TotalSegmentator predicts every other bone kind but is not ground truth.
For every bone except ribs, TotalSegmentator is the sole source. For ribs, the
two are not merged by priority -- RibSegv2 is the *only* source: TotalSegmentator's
rib predictions are discarded outright rather than filling in where RibSegv2 is
silent (see ``combine_bone_label``'s docstring for why). This comes in two
phases, over two label spaces:

- Phase 1, coarse (see ``COARSE_CLASS_NAMES``): one class per bone kind --
  background plus nine named bones -- with every rib and every vertebra
  sharing a single ``rib``/``vertebrae`` class regardless of which one it is.
- Phase 2, fine (see ``FINE_CLASS_NAMES``): the same nine kinds, but with
  individual instances distinguished -- ribs become ``rib_left_1..12`` /
  ``rib_right_1..12``, vertebrae become the individual vertebrae, and
  left/right bones are separate classes.

A config's ``num_classes`` must equal ``NUM_COARSE_CLASSES`` (10) or
``NUM_FINE_CLASSES`` (62), according to the selected granularity. Selecting
fine explicitly looks like::

    dict(type="CombineBoneLabel", granularity="fine")

Usage:
    python -m pointcept.datasets.ribsegv2.plus_totalseg_pred --patch-totalseg \
        --cache-root data/ribsegv2/pt_preproc -o ~/data/ribseg/pt_preproc_totalseg
    python -m pointcept.datasets.ribsegv2.plus_totalseg_pred --summarise-totalseg \
        --cache-root ~/data/ribseg/pt_preproc_totalseg
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pointcept.datasets.totalseg import TOTALSEG_CLASSES, TOTALSEG_RIB_CLASSES
from pointcept.datasets.transform import TRANSFORMS

# Index is the class id. Background first, then every bone kind this task
# segments, built from RibSegv2's manual rib annotation plus TotalSegmentator's
# prediction for everything else.
COARSE_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "rib",
    "vertebrae",
    "scapula",
    "humerus",
    "clavicula",
    "sternum",
    "costal_cartilages",
    "skull",
    "hip",
)
NUM_COARSE_CLASSES = len(COARSE_CLASS_NAMES)

# Name prefixes matched against TOTALSEG_CLASSES, in priority order. Order
# only matters if a name could match more than one prefix, which none here do,
# but keeping this ordered documents the intended left-to-right build. This
# also defines the fine class space below: for each coarse bone kind, in this
# order, the TotalSegmentator ids matching its prefixes become consecutive
# fine ids.
_PREFIX_TO_COARSE_NAME: tuple[tuple[str, str], ...] = (
    ("rib_", "rib"),
    ("vertebrae_", "vertebrae"),
    # `sacrum` (id 25) and `vertebrae_S1` (id 26) are the same vertebral
    # column, just named differently in the v2 class map, so they share a
    # class rather than getting their own.
    ("sacrum", "vertebrae"),
    ("scapula_", "scapula"),
    ("humerus_", "humerus"),
    ("clavicula_", "clavicula"),
    ("sternum", "sternum"),
    ("costal_cartilages", "costal_cartilages"),
    ("skull", "skull"),
    ("hip_", "hip"),
)

# Femur is deliberately absent and folds into background: it appears in only
# 19 of 657 volumes at 0.011% of points, too rare to learn -- its per-class
# IoU would be noise. Every other remaining TotalSegmentator class -- organs,
# vessels, muscle -- also maps to background. Those do occur, at about 0.8% of
# points in total, because contrast-enhanced vessels, calcifications and
# implants clear the 200 HU threshold used to build the point cloud.
# TotalSegmentator's rib ids (``TOTALSEG_RIB_CLASSES``) now join that list too,
# via ``_suppress_totalseg_ribs`` below -- not because they are rare, but
# because the rib class must come from RibSegv2 alone (see that helper and
# ``combine_bone_label`` for why).


def _suppress_totalseg_ribs(table: np.ndarray) -> np.ndarray:
    """Zero ``TOTALSEG_RIB_CLASSES`` entries in a TotalSegmentator lookup table.

    RibSegv2's manual rib annotation is the ground truth for ribs, and
    TotalSegmentator's own rib prediction adds nothing where RibSegv2 already
    labels a rib -- the RibSegv2 label wins there anyway, via the override in
    ``combine_bone_label``. Where RibSegv2 says background, though, a
    TotalSegmentator rib id is a false positive that would otherwise be
    written into the target as ``rib``. Measured over a 12-volume sample,
    points where TotalSegmentator says rib but RibSegv2 does not are 0.455% of
    all cached points -- about 2% of everything that would otherwise be
    labelled rib, and up to 7% of the rib class on the worst volume in the
    sample (600). They concentrate exactly where TotalSegmentator confuses
    ribs with vertebrae and transverse processes, which is the very confusion
    this experiment exists to measure. Leaving them in the target would bake
    that confusion into the ground truth of an experiment designed to detect
    it.

    So the rib class must come from RibSegv2 alone. Zeroing these entries here,
    in the lookup table itself, makes "no TotalSegmentator id can ever produce
    a rib class" a property of the table rather than of the order statements
    run in ``combine_bone_label``.

    Args:
        table: a TotalSegmentator-id to class-id lookup table, modified and
            also returned.

    Returns:
        ``table``, with every index in ``TOTALSEG_RIB_CLASSES`` set to 0.
    """
    table[list(TOTALSEG_RIB_CLASSES)] = 0
    return table


def _build_totalseg_to_coarse() -> np.ndarray:
    """Build the TotalSegmentator-id to coarse-bone-id lookup, once at import."""
    name_to_coarse_id = {name: index for index, name in enumerate(COARSE_CLASS_NAMES)}
    table = np.zeros(max(TOTALSEG_CLASSES) + 1, dtype=np.uint8)
    matched_prefixes: set[str] = set()
    for totalseg_id, name in TOTALSEG_CLASSES.items():
        for prefix, coarse_name in _PREFIX_TO_COARSE_NAME:
            if name.startswith(prefix):
                table[totalseg_id] = name_to_coarse_id[coarse_name]
                matched_prefixes.add(prefix)
                break
    # A re-downloaded map_to_binary.py that renamed a class would otherwise
    # silently zero out (background) that entire prefix's bone class.
    unmatched = [prefix for prefix, _ in _PREFIX_TO_COARSE_NAME if prefix not in matched_prefixes]
    if unmatched:
        raise RuntimeError(
            f"TOTALSEG_CLASSES has no name matching prefix(es) {unmatched!r}; "
            "map_to_binary.py may have been re-downloaded with renamed classes"
        )
    return table


TOTALSEG_TO_COARSE: np.ndarray = _suppress_totalseg_ribs(_build_totalseg_to_coarse())


def _build_fine_class_names() -> tuple[str, ...]:
    """Build the fine class name list, once at import.

    Walks ``COARSE_CLASS_NAMES`` in order (skipping ``"background"``), and for
    each coarse bone kind takes the TotalSegmentator ids whose names match that
    kind's prefixes, sorted ascending, appending their names in that order.
    This derives the space from the vendored class map rather than hardcoding
    it, so a re-download that only adds or renames classes elsewhere in the map
    cannot silently desync this list from ``TOTALSEG_CLASSES``.
    """
    names: list[str] = ["background"]
    for coarse_name in COARSE_CLASS_NAMES[1:]:
        prefixes = [prefix for prefix, name in _PREFIX_TO_COARSE_NAME if name == coarse_name]
        matching_ids = sorted(
            totalseg_id
            for totalseg_id, name in TOTALSEG_CLASSES.items()
            if any(name.startswith(prefix) for prefix in prefixes)
        )
        names.extend(TOTALSEG_CLASSES[totalseg_id] for totalseg_id in matching_ids)
    return tuple(names)


FINE_CLASS_NAMES: tuple[str, ...] = _build_fine_class_names()
NUM_FINE_CLASSES = len(FINE_CLASS_NAMES)

# This invariant is load-bearing, not incidental: TotalSegmentator numbers
# rib_left_1..12 as 92..103 and rib_right_1..12 as 104..115, both contiguous
# ascending blocks, so sorting ascending (as _build_fine_class_names does)
# lands them on fine ids 1..24 in exactly RibSegv2's own numbering
# (rib_left_N -> N, rib_right_N -> N + 12). Two things below depend on this
# holding exactly, not approximately:
#   - the RibSegv2 override inside combine_bone_label's fine branch is a
#     straight copy of the RibSegv2 label, with no remapping table of its own;
#   - the RibSegV2-specific metrics name rib ids 1..24 and so would need this
#     same convention if extended to the fine space.
# A re-downloaded map_to_binary.py that reordered or renumbered the rib
# classes would silently break both without this check, so it is asserted
# here at import rather than trusted.
for _k in range(1, 25):
    _expected = f"rib_left_{_k}" if _k <= 12 else f"rib_right_{_k - 12}"
    if FINE_CLASS_NAMES[_k] != _expected:
        raise RuntimeError(
            f"FINE_CLASS_NAMES[{_k}] is {FINE_CLASS_NAMES[_k]!r}, expected {_expected!r}; "
            "TotalSegmentator's rib numbering no longer lines up with RibSegv2's, so the "
            "fine-space RibSegv2 override in combine_bone_label and "
            "RibSegV2's 1..24 metric convention is no longer valid as written"
        )
del _k, _expected


def _build_totalseg_to_fine() -> np.ndarray:
    """Build the TotalSegmentator-id to fine-class-id lookup, once at import."""
    name_to_fine_id = {name: index for index, name in enumerate(FINE_CLASS_NAMES)}
    table = np.zeros(max(TOTALSEG_CLASSES) + 1, dtype=np.uint8)
    for totalseg_id, name in TOTALSEG_CLASSES.items():
        # Names absent from FINE_CLASS_NAMES (organs, vessels, muscle, femur)
        # simply stay at the table's zero (background) default; no separate
        # unmatched-prefix check is needed here since _build_fine_class_names
        # already runs the same prefix matching that _build_totalseg_to_coarse
        # validates.
        if name in name_to_fine_id:
            table[totalseg_id] = name_to_fine_id[name]
    return table


TOTALSEG_TO_FINE: np.ndarray = _suppress_totalseg_ribs(_build_totalseg_to_fine())


def combine_bone_label(
    data: dict[str, Any],
    label_key: str = "segment",
    totalseg_key: str = "totalseg_pred",
    granularity: str = "coarse",
    drop_source: bool = True,
) -> dict[str, Any]:
    """Combine a RibSegv2 rib label with a TotalSegmentator prediction.

    In both granularities, RibSegv2's rib annotation is manual ground truth,
    and TotalSegmentator only scores about 0.83 dice against it, so wherever
    RibSegv2 has labelled a point (label > 0) that label wins. TotalSegmentator's
    prediction only fills in the bones RibSegv2 never labelled at all.

    For the rib class specifically, this is not a priority order but exclusivity:
    the rib class comes from RibSegv2 alone. ``TOTALSEG_TO_COARSE`` and
    ``TOTALSEG_TO_FINE`` have every TotalSegmentator rib id (``TOTALSEG_RIB_CLASSES``)
    zeroed at build time (see ``_suppress_totalseg_ribs``), so TotalSegmentator's
    rib predictions are discarded rather than merged in: they concentrate exactly
    where TotalSegmentator confuses ribs with vertebrae and transverse processes,
    the very confusion this task exists to measure, and merging them in would bake
    that confusion into the ground truth. The consequence is that a point
    TotalSegmentator calls a rib and RibSegv2 does not label becomes background,
    not vertebrae or anything else -- TotalSegmentator gives one class per point
    and, once its rib guess is discarded, offers no better guess for that point.

    In ``"coarse"``, the override always writes the single ``rib`` class,
    collapsing ribs 1..24 to one id. That is convenient for RibSegv2's known
    wrong-label volumes (``IGNORE_VOLUMES`` in ``base.py``):
    those defects are rib *mis-numbering*, not mis-segmentation, so a coarse
    target that never distinguishes individual ribs cannot be corrupted by
    them, and excluding those volumes could in principle be relaxed here.

    In ``"fine"``, the same statement is false and actively misleading: the
    override writes the RibSegv2 label itself as the fine class id (see the
    invariant documented above ``TOTALSEG_TO_FINE``), so a mis-numbered rib
    volume would train the fine target on the wrong individual rib id --
    exactly the defect that matters at this granularity. ``IGNORE_VOLUMES``
    is therefore load-bearing for the fine phase, not just a convenience.

    Args:
        data: Data dict already holding both source arrays, point-aligned.
        label_key: Key of the RibSegv2 label array; also where the combined
            result is written.
        totalseg_key: Key of the TotalSegmentator prediction array.
        granularity: ``"coarse"`` for ``COARSE_CLASS_NAMES``/``NUM_COARSE_CLASSES``,
            or ``"fine"`` for ``FINE_CLASS_NAMES``/``NUM_FINE_CLASSES``.
        drop_source: When true, delete ``data[totalseg_key]`` once consumed.

    Returns:
        ``data``, with ``data[label_key]`` replaced by the combined label at
        the requested granularity and, when ``drop_source``, ``totalseg_key``
        removed.
    """
    if granularity not in ("coarse", "fine"):
        raise ValueError(f"granularity must be one of ('coarse', 'fine'), got {granularity!r}")
    if label_key not in data:
        raise KeyError(label_key)
    if totalseg_key not in data:
        raise KeyError(totalseg_key)

    label = np.asarray(data[label_key])
    totalseg_pred = np.asarray(data[totalseg_key])
    if label.ndim != 1 or totalseg_pred.ndim != 1 or label.shape != totalseg_pred.shape:
        raise ValueError(
            f"{label_key} and {totalseg_key} must be 1-D and the same length "
            f"(they are point-aligned; a sampling transform likely filtered one "
            f"and not the other), got shapes {label.shape} and {totalseg_pred.shape}"
        )

    if not np.issubdtype(label.dtype, np.integer):
        raise TypeError(f"{label_key} must have an integer dtype, got {label.dtype}")
    if not np.issubdtype(totalseg_pred.dtype, np.integer):
        raise TypeError(
            f"{totalseg_key} must have an integer dtype, got {totalseg_pred.dtype}"
        )
    if np.any(label < 0) or np.any(label > 24):
        raise ValueError(
            f"{label_key} contains a value outside RibSegV2's manual-label range 0..24"
        )

    # NUM_FINE_CLASSES is 62, still well within uint8's range, same as the 10
    # coarse classes, so this needs no dtype widening for either granularity.
    totalseg_to_class = TOTALSEG_TO_COARSE if granularity == "coarse" else TOTALSEG_TO_FINE

    # Guard the index rather than raising: an id at or beyond the table length
    # is defensive against a prediction made with a different subtask, whose
    # ids would otherwise index out of bounds.
    in_range = (totalseg_pred >= 0) & (totalseg_pred < totalseg_to_class.shape[0])
    combined = np.zeros(label.shape, dtype=label.dtype)
    combined[in_range] = totalseg_to_class[totalseg_pred[in_range]]

    if granularity == "coarse":
        rib_id = COARSE_CLASS_NAMES.index("rib")
        # RibSegv2's manual rib label overrides the TotalSegmentator prediction
        # wherever it exists.
        combined[label > 0] = rib_id
    else:
        # In the fine space, RibSegv2's label *is* the fine rib id already --
        # rib_left_1..12 / rib_right_1..12 land on fine ids 1..24 in exactly
        # RibSegv2's own numbering (see the invariant checked at import above
        # TOTALSEG_TO_FINE) -- so the override is a straight copy, not a lookup.
        combined[label > 0] = label[label > 0]

    # Keep the incoming integer dtype so this composes either side of TypeCast.
    data[label_key] = combined.astype(label.dtype, copy=False)

    if drop_source:
        # Consumed: leaving it would carry a per-point array through the rest
        # of the pipeline that no later transform subsamples, so it would
        # silently desynchronise from the points if anything downstream read it.
        del data[totalseg_key]

    return data


@TRANSFORMS.register_module()
class CombineBoneLabel:
    """Pointcept transform wrapper around :func:`combine_bone_label`.

    With ``granularity=None``, consume the ``bone_label_granularity`` sample
    metadata set by the RibSegV2 dataset. The shared PTv3 pipeline uses this
    mode so coarse and fine configs do not duplicate transform lists.
    """

    def __init__(self, label_key="segment", totalseg_key="totalseg_pred",
                 granularity="coarse", drop_source=True):
        self.label_key = label_key
        self.totalseg_key = totalseg_key
        self.granularity = granularity
        self.drop_source = drop_source

    def __call__(self, data):
        if self.granularity is None:
            granularity = data.pop("bone_label_granularity", "coarse")
        else:
            data.pop("bone_label_granularity", None)
            granularity = self.granularity
        return combine_bone_label(data, self.label_key, self.totalseg_key,
                                  granularity, self.drop_source)


# --------------------------------------------------------------------------- #
# Offline entry points
# --------------------------------------------------------------------------- #
#
# This module sits on the training import path for the complemented configs.
# ``preproc.py`` imports nibabel, so the NIfTI alignment helper is imported
# inside ``patch_totalseg_pred`` rather than making every dataloader worker pay
# that cost merely to obtain the array-only transform above.


def _open_jsonl_log(path: Path, header: dict[str, Any]) -> Path:
    """Start a JSON-lines log and preserve the previous run as ``.bak``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        path.replace(path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(header) + "\n")
    return path


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _finalize_log(path: Path) -> None:
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def patch_totalseg_pred(
    cache_root: str = "data/ribsegv2/pt_preproc",
    save_path: str = "~/data/ribseg/pt_preproc_totalseg",
    tspred_path: str = "~/data/ribseg/totalseg_pred/raw",
    subtask: str = "total",
    volume_ids: Sequence[int] | None = None,
    overwrite: bool = False,
) -> None:
    """Derive an all-bones cache from ``cache_root`` by adding TotalSegmentator ids.

    Reads each entry of ``cache_root``, adds ``totalseg_pred``, and writes the
    complete entry to ``save_path``, leaving ``cache_root`` untouched.

    ``totalseg_pred`` is one raw TotalSegmentator class id per cached point,
    uint8, aligned index-for-index with ``voxel_index`` in the same entry. These
    are the model's own label ids, not RibSegv2's 1..24 -- the v2 ``total`` set,
    the only one this repo uses, numbers ``rib_left_1..12`` and
    ``rib_right_1..12`` as 92..103 and 104..115 -- and the mapping onto RibSegv2
    (``rib_left_N -> N``, ``rib_right_N -> N + 12``) is deliberately left to
    an online transform rather than applied here, so the non-rib anatomy
    TotalSegmentator also predicted stays available downstream.

    No NIfTI CT is re-read to build this. ``ct_to_point_cloud`` thresholds HU
    above 200 over the whole volume with no class restriction, so the cached
    cloud already contains spine, sternum, scapula and clavicle points -- only
    the *label* array was rib-specific. The new cache is therefore the same
    points as ``cache_root`` with one array added, and stays point-for-point
    comparable with the rib experiments.

    Each prediction is reoriented to the cache's LPS by its own axcodes and
    checked against the entry's grid -- the same mechanism ``_load_sieve`` uses,
    via ``_load_aligned_volume`` -- before being indexed by ``voxel_index``.

    Args:
        cache_root: point-cloud cache to read from; never written to.
        save_path: directory for the all-bones cache; must not be ``cache_root``.
        tspred_path: directory of ``{vid}-ts_pred-{subtask}.nii.gz`` predictions.
        subtask: TotalSegmentator subtask name, part of the prediction filename.
            Only ``total`` is valid because the vendored class map and both
            training taxonomies use its IDs.
        volume_ids: patch only these volumes; when omitted, every ``.npz`` found
            in ``cache_root`` is visited instead of ``ALL_VOLUME_IDS``, so this
            also works unchanged against a stage-2 cache holding a subset of
            ids, and a "missing prediction" report below cannot be a false
            alarm for ids the cache never had. An id named explicitly here that
            has no cache entry is an error rather than a skip, since the caller
            chose it deliberately.
        overwrite: rewrite an output entry that already exists at ``save_path``.

    Raises:
        ValueError: if ``save_path`` and ``cache_root`` are the same directory.
        RuntimeError: if any visited volume had no TotalSegmentator prediction
            file. Every other volume has already been patched by then, so a
            re-run has no work to redo.
    """
    if subtask != "total":
        raise ValueError(
            f"subtask must be 'total' because the label maps use TotalSegmentator "
            f"v2 total IDs, got {subtask!r}"
        )
    from pointcept.datasets.ribsegv2.preproc import PREPROC_IGNORE_VOLUMES, _load_aligned_volume
    from pointcept.utils.misc import np_smallest_dtype

    cache_dir = Path(cache_root).expanduser()
    save_dir = Path(save_path).expanduser()
    tspred_dir = Path(tspred_path).expanduser()
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"cache_root does not exist: {cache_dir}")
    if not tspred_dir.is_dir():
        raise NotADirectoryError(f"tspred_path does not exist: {tspred_dir}")
    # An npz cannot be appended to, so patching in place would mean rewriting
    # every entry of a cache that live 2-stage experiments are currently
    # reading, and would make every rib config pay the extra per-sample
    # LoadNPZ read of a field only the all-bones runs use.
    if save_dir.resolve() == cache_dir.resolve():
        raise ValueError(
            f"save_path and cache_root are the same directory ({cache_dir}). "
            "cache_root is read-only here -- -o/save_path must name the all-bones "
            "cache to write, conventionally ~/data/ribseg/pt_preproc_totalseg."
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    if volume_ids is None:
        # Enumerating the cache itself, rather than ALL_VOLUME_IDS, is what makes
        # this work unchanged on a stage-2 cache holding a subset of ids.
        ids_to_patch = sorted(int(entry.stem) for entry in cache_dir.glob("*.npz"))
    else:
        ids_to_patch = []
        for volume_id in volume_ids:
            if volume_id in PREPROC_IGNORE_VOLUMES:
                continue
            cache_file = cache_dir / f"{volume_id}.npz"
            if not cache_file.is_file():
                raise FileNotFoundError(f"Volume {volume_id}: no cache entry at {cache_file}")
            ids_to_patch.append(volume_id)
    if not ids_to_patch:
        raise RuntimeError("No cache entries selected; check cache_root and volume_ids")

    missing_pred: list[int] = []
    rib_fractions: dict[int, float] = {}

    for volume_id in ids_to_patch:
        print(volume_id, end="\r")
        out_path = save_dir / f"{volume_id}.npz"
        if out_path.is_file() and not overwrite:
            continue

        cache_file = cache_dir / f"{volume_id}.npz"
        with np.load(cache_file, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}

        ts_path = tspred_dir / f"{volume_id}-ts_pred-{subtask}.nii.gz"
        if not ts_path.is_file():
            missing_pred.append(volume_id)
            continue

        grid = _load_aligned_volume(
            ts_path,
            arrays["affine"],
            tuple(arrays["nifti_shape"].tolist()),
            volume_id,
            dtype=np.int16,
            what="TotalSegmentator prediction",
        )
        if grid.size and (grid.min() < 0 or grid.max() > max(TOTALSEG_CLASSES)):
            raise ValueError(
                f"Volume {volume_id}: TotalSegmentator prediction contains ids outside "
                f"the v2 total range 0..{max(TOTALSEG_CLASSES)}"
            )
        voxel_index = arrays["voxel_index"]
        prediction = grid[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]]
        arrays["totalseg_pred"] = np_smallest_dtype(prediction)

        # A cheap sanity figure computed for every volume: the cached points are
        # bone above 200 HU, so a correctly aligned prediction puts a large share
        # of them on ribs, while a mirrored one -- the same failure mode the
        # grid check inside _load_aligned_volume guards against -- collapses
        # towards zero.
        rib_fractions[volume_id] = (
            float(np.isin(prediction, TOTALSEG_RIB_CLASSES).mean()) if prediction.size else 0.0
        )

        # The resume rule above trusts the presence of an output file, so an
        # interrupted run that left a truncated entry behind would be skipped as
        # done on the next pass and silently poison the cache -- stricter than
        # preprocess_ptcloud, which writes straight to out_path. np.savez_compressed
        # appends ".npz" to a path argument, which would defeat a ".tmp" name, so
        # the temp file is opened here and its handle is passed in instead.
        tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(tmp_path, out_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    if rib_fractions:
        mean_fraction = sum(rib_fractions.values()) / len(rib_fractions)
        lowest = sorted(rib_fractions.items(), key=lambda item: item[1])[:10]
        listed = ", ".join(f"{vid} ({fraction:.3f})" for vid, fraction in lowest)
        print(
            f"Patched {len(rib_fractions)} volume(s); mean rib fraction {mean_fraction:.3f}. "
            f"Ten lowest -- inspect any far below the rest: {listed}"
        )

    if missing_pred:
        listed = ", ".join(str(vid) for vid in sorted(missing_pred))
        raise RuntimeError(
            f"{len(missing_pred)} volume(s) had no TotalSegmentator prediction under "
            f"{tspred_dir}: {listed}. Every other volume has already been patched, so "
            "a re-run has no work to redo."
        )


def summarise_totalseg_classes(
    cache_root: str = "~/data/ribseg/pt_preproc_totalseg",
    log_path: str = "data/ribsegv2/totalseg-class-set.jsonl",
    volume_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarise which TotalSegmentator classes occur at the cached points.

    This reads the **cache**, not the raw ``.nii.gz`` predictions under
    ``tspred_path``. The class set that matters for choosing a coarse bone
    taxonomy is the one present at the cached points -- thresholded above 200 HU
    in ``ct_to_point_cloud`` -- since that is what a training transform will
    actually see. The raw ``.nii.gz`` prediction covers the whole volume,
    including organ interiors that never become points, so reading it would
    overstate the label space this task actually has to deal with. Reading the
    cache is also far cheaper: no 512x512xZ decompression per volume, just the
    one array already sitting in the entry.

    Args:
        cache_root: the all-bones cache to read, carrying ``totalseg_pred``
            (see ``patch_totalseg_pred``).
        log_path: JSON-lines output; one line per volume plus a final summary.
        volume_ids: summarise only these volumes; when omitted, every ``.npz``
            found in ``cache_root`` is visited, as in ``patch_totalseg_pred``.

    Returns:
        The summary record also appended as the log's last line: total volume
        and point counts, and per-class point counts, fractions and volume
        counts, sorted by ``n_points`` descending.

    Raises:
        NotADirectoryError: if ``cache_root`` does not exist.
        RuntimeError: if no volume contributed any points.
    """
    from pointcept.datasets.ribsegv2.preproc import PREPROC_IGNORE_VOLUMES

    cache_dir = Path(cache_root).expanduser()
    log_file = Path(log_path).expanduser()
    if not cache_dir.is_dir():
        raise NotADirectoryError(f"cache_root does not exist: {cache_dir}")

    log_file = _open_jsonl_log(
        log_file,
        {
            "time": time.asctime(time.gmtime()),
            "content": "Which TotalSegmentator classes occur at the cached points, and how "
            "much of the cloud each accounts for.",
            "cache_root": str(cache_dir),
        },
    )

    if volume_ids is None:
        # Enumerating the cache itself, rather than ALL_VOLUME_IDS, is what makes
        # this work unchanged on a stage-2 cache holding a subset of ids.
        ids_to_visit = sorted(int(entry.stem) for entry in cache_dir.glob("*.npz"))
    else:
        ids_to_visit = []
        for volume_id in volume_ids:
            if volume_id in PREPROC_IGNORE_VOLUMES:
                continue
            cache_file = cache_dir / f"{volume_id}.npz"
            if not cache_file.is_file():
                raise FileNotFoundError(f"Volume {volume_id}: no cache entry at {cache_file}")
            ids_to_visit.append(volume_id)

    total_points_per_class: dict[int, int] = {}
    volumes_per_class: dict[int, int] = {}
    n_volumes = 0
    n_points = 0

    for volume_id in ids_to_visit:
        print(volume_id, end="\r")
        cache_file = cache_dir / f"{volume_id}.npz"
        with np.load(cache_file) as archive:
            if "totalseg_pred" not in archive.files:
                raise KeyError(
                    f"Volume {volume_id}: {cache_file} has no totalseg_pred array; "
                    "the cache was built without --patch-totalseg"
                )
            # Index by the one key needed rather than materialising every array
            # in the entry -- these carry several megabytes of other data per
            # volume, none of it needed here.
            prediction = archive["totalseg_pred"]

        ids, counts = np.unique(prediction, return_counts=True)
        classes = {str(int(class_id)): int(count) for class_id, count in zip(ids, counts)}
        _append_jsonl(log_file, {"vid": volume_id, "n_points": int(prediction.size), "classes": classes})

        n_volumes += 1
        n_points += int(prediction.size)
        for class_id, count in zip(ids, counts):
            class_id = int(class_id)
            count = int(count)
            total_points_per_class[class_id] = total_points_per_class.get(class_id, 0) + count
            volumes_per_class[class_id] = volumes_per_class.get(class_id, 0) + 1

    if n_points == 0:
        raise RuntimeError("No volumes contributed any points; check cache_root and volume_ids")

    classes_summary = sorted(
        (
            {
                "id": class_id,
                "name": "background" if class_id == 0 else TOTALSEG_CLASSES.get(class_id, f"unknown_{class_id}"),
                "n_points": count,
                "fraction": round(count / n_points, 6),
                "n_volumes": volumes_per_class[class_id],
            }
            for class_id, count in total_points_per_class.items()
        ),
        key=lambda record: record["n_points"],
        reverse=True,
    )
    summary = {"n_volumes": n_volumes, "n_points": n_points, "classes": classes_summary}
    _append_jsonl(log_file, summary)
    _finalize_log(log_file)
    print(f"TotalSegmentator class summary over {n_volumes} volumes -> {log_file} ({len(classes_summary)} classes)")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--patch-totalseg",
        action="store_true",
        help="derive an all-bones cache into -o by adding raw TotalSegmentator class ids "
        "to each entry of --cache-root",
    )
    parser.add_argument(
        "--summarise-totalseg",
        action="store_true",
        help="summarise which TotalSegmentator classes occur at the cached points, one JSON line per volume",
    )
    parser.add_argument(
        "-o",
        "--save-path",
        type=str,
        default="~/data/ribseg/pt_preproc_totalseg",
        help="with --patch-totalseg, the all-bones cache to write, which must not be --cache-root",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default="",
        help="with --patch-totalseg it is the cache read from -- defaulting to "
        "data/ribsegv2/pt_preproc when left empty -- and is never written to; "
        "with --summarise-totalseg it is the cache read, defaulting to "
        "~/data/ribseg/pt_preproc_totalseg when left empty",
    )
    parser.add_argument(
        "--tspred-path",
        type=str,
        default="~/data/ribseg/totalseg_pred/raw",
        help="with --patch-totalseg, directory of {vid}-ts_pred-{subtask}.nii.gz predictions",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="total",
        help="with --patch-totalseg, TotalSegmentator subtask name in the prediction filename; only total is supported",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="with --patch-totalseg, rewrite an output entry that already exists",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="",
        help="with --summarise-totalseg, where to write the JSON-lines summary, defaulting to "
        "data/ribsegv2/totalseg-class-set.jsonl",
    )
    parser.add_argument(
        "--volume-ids",
        type=int,
        nargs="+",
        default=None,
        help="with --patch-totalseg or --summarise-totalseg, process only these volume ids "
        "instead of every entry found in --cache-root",
    )
    args = parser.parse_args()
    if not (args.patch_totalseg or args.summarise_totalseg):
        parser.error("choose at least one of --patch-totalseg, --summarise-totalseg")
    return args


def main() -> None:
    args = _parse_args()
    if args.patch_totalseg:
        patch_totalseg_pred(
            args.cache_root or "data/ribsegv2/pt_preproc",
            args.save_path,
            args.tspred_path,
            args.subtask,
            args.volume_ids,
            args.overwrite,
        )
    if args.summarise_totalseg:
        summarise_totalseg_classes(
            args.cache_root or "~/data/ribseg/pt_preproc_totalseg",
            args.log_path or "data/ribsegv2/totalseg-class-set.jsonl",
            args.volume_ids,
        )


if __name__ == "__main__":
    main()

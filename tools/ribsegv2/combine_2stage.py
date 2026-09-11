"""Combine stage-1 and stage-2 predictions and score the result.

Step 6 of the 2-stage rib pipeline (see plans/20260824-2stage.md). Stage 1
predicts foreground/background over every point of the one-stage cache; stage 2
numbers the ribs of the subset stage 1 kept. The combined prediction starts at
background and takes stage 2's rib number wherever stage 1 said foreground.

Points stage 1 called background stay 0. So do points it called foreground that
never reached stage 2: `recon_3d_bin_pred`'s `remove_statistical_outlier` drops
spatial outliers before the sieve is written, so stage 2's point set is a strict
subset of stage 1's foreground and a dropped point has no rib number to take.
That count is reported per volume as `num_unlabelled_foreground` rather than
hidden -- it is a real loss of the combined prediction, not an accounting quirk.

Scoring runs over the STAGE-1 POINT SET, i.e. every point in the one-stage
cache, which is the same denominator the 1-stage runs report on, so these
numbers drop straight into the plan's Performance Table. Scoring only stage 2's
smaller set would quietly excuse every voxel stage 1 lost. The 25-class ground
truth therefore comes from `--cache-root`, not from either prediction dump:
stage 1's `label` was binarised by `BinarizeLabel` and stage 2's covers only its
own subset.

Metrics come from `pointcept.utils.eval_cm.eval_volume` / `reduce_records`, the
very functions `Ribsegv2VolumeTester` scores with, so the combined numbers
cannot drift from the per-stage ones.

Usage (from the repo root -- `pointcept/datasets/ribsegv2/base.py` reads
`data/ribsegv2/ribsegv2-statistics.json` through a bare relative path at import
time, so any other working directory fails there):
    python tools/ribsegv2/combine_2stage.py --stage1-pred <s1_log>/result \
        --stage2-pred <s2_log>/result --cache-root data/ribsegv2/pt_preproc \
        --split test --save-path <s2_log>
"""

import os
import sys

# runnable as `python tools/ribsegv2/<name>.py` from anywhere; scripts/*.sh
# instead export PYTHONPATH, so guard against inserting the repo root twice
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse, json, time
import numpy as np

from pointcept.utils.eval_cm import RIBSEG_DEFAULT_METRICS, eval_volume, reduce_records
from pointcept.utils.misc import natural_sort_key, to_jsonable


def _load_split_definitions():
    """`SPLITS` and `IGNORE_VOLUMES` from ribsegv2/base.py, without its package

    A plain `from pointcept.datasets.ribsegv2.base import ...` runs
    `pointcept/datasets/__init__.py`, which eagerly imports every dataset --
    including `modelnet`, which imports the `pointops` CUDA extension. This
    script is pure post-processing over two .npz dumps and should run wherever
    the numbers are being read, not only on a GPU-built environment, so the
    module is loaded from its file instead. Single-sourced either way: there is
    no second copy of the split lists here.
    """
    import importlib.util

    path = os.path.join(_ROOT, "pointcept", "datasets", "ribsegv2", "base.py")
    spec = importlib.util.spec_from_file_location("_ribsegv2_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPLITS, module.IGNORE_VOLUMES


NUM_CLASSES = 1 + 24  # background + rib1..rib24, what stage 2 predicts
BG_CLASS = 0


def voxel_keys(voxel_index, nifti_shape, source):
    """one collision-free integer per voxel, so two point sets can be matched

    Both prediction dumps and the cache index the same LPS grid, so a voxel's
    [i, j, k] triple is a stable identity across them -- but each array holds a
    different subset in its own order, so they must be matched BY VALUE. A
    positional assumption would fail silently the day an export order changes.
    Args:
        voxel_index: int numpy.ndarray[n_points, 3], voxel indices into the grid
        nifti_shape: int[3], shape of that grid
        source: str, file the indices came from, for error messages only
    Returns:
        int numpy.ndarray[n_points], the raveled index of each voxel
    """
    voxel_index = np.asarray(voxel_index, dtype=np.int64)
    assert voxel_index.ndim == 2 and voxel_index.shape[1] == 3, (
        "{}: voxel_index must be [n_points, 3], got {}".format(source, voxel_index.shape)
    )
    shape = tuple(int(x) for x in np.asarray(nifti_shape).reshape(-1)[:3])
    assert voxel_index.size == 0 or (
        voxel_index.min() >= 0 and bool((voxel_index.max(axis=0) < np.array(shape)).all())
    ), "{}: voxel_index falls outside the volume shape {}".format(source, shape)
    return np.ravel_multi_index(voxel_index.T, shape)


def locate(needles, haystack, source, target):
    """position of each `needles` key within `haystack`, erroring on absentees
    Args:
        needles, haystack: int numpy.ndarray[n], voxel keys from `voxel_keys`
        source, target: str, named in the error message
    Returns:
        int numpy.ndarray[len(needles)], index into `haystack`
    """
    if needles.size == 0:
        return np.empty(0, dtype=np.intp)
    assert haystack.size > 0, "{}: {} voxels against an empty {}".format(
        source, needles.size, target
    )
    order = np.argsort(haystack)
    sorted_haystack = haystack[order]
    position = np.searchsorted(sorted_haystack, needles)
    # searchsorted returns the insertion point, which is one past the end for a
    # key larger than everything in the haystack -- not a match either.
    in_range = position < sorted_haystack.size
    found = in_range & (sorted_haystack[np.where(in_range, position, 0)] == needles)
    assert found.all(), "{}: {} of {} voxels are absent from {}".format(
        source, int((~found).sum()), found.size, target
    )
    return order[position]


def combine_volume(cache_path, stage1_path, stage2_path, vid):
    """combined prediction and 25-class ground truth of one volume, in cache order
    Args:
        cache_path: str, {cache_root}/{vid}.npz, the one-stage cache entry
        stage1_path, stage2_path: str, {vid}.npz written by Ribsegv2VolumeTester(save_pred=True)
        vid: int, volume id, used in error messages only
    Returns:
        pred: int numpy.ndarray[n_points], combined prediction over the stage-1 point set
        label: same shape, 25-class ground truth
        counts: dict, point accounting of this volume
    """
    with np.load(cache_path) as cache:
        label = np.asarray(cache["label"], dtype=np.int64)
        nifti_shape = np.asarray(cache["nifti_shape"])
        cache_keys = voxel_keys(cache["voxel_index"], nifti_shape, "volume {} cache".format(vid))
    with np.load(stage1_path) as archive:
        stage1_pred = np.asarray(archive["pred"], dtype=np.int64)
        stage1_keys = voxel_keys(archive["voxel_index"], nifti_shape, "volume {} stage 1".format(vid))
    with np.load(stage2_path) as archive:
        stage2_pred = np.asarray(archive["pred"], dtype=np.int64)
        stage2_keys = voxel_keys(archive["voxel_index"], nifti_shape, "volume {} stage 2".format(vid))

    assert label.max() < NUM_CLASSES, "volume {}: cache label holds class {} >= {}".format(
        vid, label.max(), NUM_CLASSES
    )
    assert stage1_pred.min() >= 0 and stage1_pred.max() <= 1, (
        "volume {}: stage-1 prediction is not binary ([{}, {}]). --stage1-pred must point at "
        "a 2-class run's result/.".format(vid, stage1_pred.min(), stage1_pred.max())
    )

    # The stage-1 run must have been tested on this very cache: it defines both
    # the scored point set and the ground truth, so a different one would move
    # every number without anything raising.
    assert stage1_keys.size == cache_keys.size, (
        "volume {}: stage 1 covers {} points but the cache holds {}. The stage-1 run used a "
        "different cache (or did not cover every point).".format(vid, stage1_keys.size, cache_keys.size)
    )
    stage1_slot = locate(stage1_keys, cache_keys, "volume {} stage 1".format(vid), "the one-stage cache")
    assert np.array_equal(np.sort(stage1_slot), np.arange(cache_keys.size)), (
        "volume {}: stage-1 voxels are not the cache's voxels".format(vid)
    )
    # A stage-2 voxel outside stage 1's set means the two runs came from
    # different caches -- a setup error, not something to paper over.
    stage2_slot = locate(stage2_keys, cache_keys, "volume {} stage 2".format(vid), "the one-stage cache")

    # Stage 2 was preprocessed from stage 1's OWN reconstructed foreground, so
    # every stage-2 point must be stage-1 foreground. This is also what makes
    # `num_unlabelled_foreground` below a plain subtraction.
    stage1_at_stage2 = stage1_pred[locate(stage2_keys, stage1_keys, "volume {} stage 2".format(vid), "stage 1")]
    assert np.all(stage1_at_stage2 == 1), (
        "volume {}: {} stage-2 points were called background by stage 1. The two prediction "
        "directories are not from the same pipeline run.".format(vid, int((stage1_at_stage2 != 1).sum()))
    )

    # Background everywhere, then stage 2's rib number wherever it made one.
    # Stage-1 foreground that outlier removal dropped never reached stage 2 and
    # stays 0, which is the documented behaviour of the combination.
    pred = np.zeros(cache_keys.size, dtype=np.int64)
    pred[stage2_slot] = stage2_pred

    n_foreground = int((stage1_pred == 1).sum())
    counts = {
        "num_stage1_foreground": n_foreground,
        "num_stage2_points": int(stage2_keys.size),
        "num_unlabelled_foreground": n_foreground - int(stage2_keys.size),
    }
    return pred, label, counts


def combine_split(stage1_pred, stage2_pred, cache_root, split="test", save_path=".", volume_ids=None):
    """score the combined prediction over one split and write both reports
    Args:
        stage1_pred, stage2_pred: str, directories of {vid}.npz predictions
        cache_root: str, the one-stage cache, which defines the scored point set
        split: str, one of train/val/test/all
        save_path: str, where the two reports go
        volume_ids: List[int] = None, score these volumes instead of the whole split
    Returns:
        summary: dict, the reduced metrics, as written to {split}-combined.json
    """
    stage1_pred = os.path.expanduser(stage1_pred)
    stage2_pred = os.path.expanduser(stage2_pred)
    cache_root = os.path.expanduser(cache_root)
    save_path = os.path.expanduser(save_path)
    for directory in (stage1_pred, stage2_pred, cache_root):
        assert os.path.isdir(directory), "No such directory: {}".format(directory)

    # loaded here, not at import time: base.py reads a data file through a bare
    # relative path, so it only resolves when run from the repo root
    SPLITS, IGNORE_VOLUMES = _load_split_definitions()
    if volume_ids is None:
        assert split in SPLITS, "Unknown split {}, expected one of {}".format(split, sorted(SPLITS))
        volume_ids = SPLITS[split]
    # SPLITS already drops these, but Ribsegv2VolumeDataset re-filters too, so
    # do the same here: an explicit --volume-ids must not sneak one back in.
    volume_ids = [vid for vid in volume_ids if vid not in IGNORE_VOLUMES]

    records = {}  # volume id -> per-volume metrics, keyed as the tester keys them
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n_unlabelled = 0
    missing = []
    for vid in volume_ids:
        cache_path = os.path.join(cache_root, "{}.npz".format(vid))
        stage1_path = os.path.join(stage1_pred, "{}.npz".format(vid))
        stage2_path = os.path.join(stage2_pred, "{}.npz".format(vid))
        if not all(os.path.isfile(p) for p in (cache_path, stage1_path, stage2_path)):
            # `--min-points` legitimately drops a volume stage 1 found no
            # foreground in, so a gap is reported rather than fatal.
            missing.append(vid)
            continue

        print("volume {}".format(vid), end='\r')
        pred, label, counts = combine_volume(cache_path, stage1_path, stage2_path, vid)
        # `n_grid` counts what the model saw and has no meaning for a merge of
        # two dumps; the point count keeps the field self-consistent.
        row = eval_volume(
            pred, label, pred.size, NUM_CLASSES, BG_CLASS,
            RIBSEG_DEFAULT_METRICS, rib_metrics=True,
        )
        # the counts ride along in the per-volume record, so the jsonl carries
        # them and `reduce_records` reports their per-volume mean too
        row.update(counts)
        records[str(vid)] = row
        n_unlabelled += counts["num_unlabelled_foreground"]
        # exactly as Ribsegv2VolumeTester builds it, so the combined confusion
        # matrix is comparable with a per-stage one
        conf_mat += np.bincount(
            label * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    print("")  # the progress line above is rewritten in place, so close it
    assert records, "None of the {} volumes of split '{}' had all three files".format(
        len(volume_ids), split
    )
    summary = reduce_records(
        records, conf_mat, NUM_CLASSES, BG_CLASS,
        RIBSEG_DEFAULT_METRICS, rib_metrics=True,
    )

    os.makedirs(save_path, exist_ok=True)
    prefix = os.path.join(save_path, "{}-combined".format(split))
    with open(prefix + ".json", 'w') as f:
        json.dump(to_jsonable({
            "time": time.asctime(time.gmtime()),
            "n_volumes": len(records),
            "num_unlabelled_foreground": n_unlabelled,
            "metrics": summary,
            "args": {
                "stage1_pred": stage1_pred, "stage2_pred": stage2_pred,
                "cache_root": cache_root, "split": split, "save_path": save_path,
                "volume_ids": volume_ids,
            },
        }), f, indent=1)
    # one self-contained object per line, as Ribsegv2VolumeTester writes it:
    # streamable, and appendable without re-reading what is already there
    with open(prefix + "-per_volume.jsonl", 'w') as f:
        f.write(json.dumps({"time": time.asctime(time.gmtime())}) + "\n")
        for name in sorted(records, key=natural_sort_key):
            f.write(json.dumps(to_jsonable({"vid": name, "metrics": records[name]})) + "\n")

    if missing:
        print("WARNING: {} of {} volumes lacked a stage-1 prediction, a stage-2 prediction or a "
              "cache entry and were skipped: {}".format(len(missing), len(volume_ids), missing))
    print("Wrote {}.json and {}-per_volume.jsonl".format(prefix, prefix))
    print("{}: {} volumes, {} foreground points stage 2 never saw".format(split, len(records), n_unlabelled))
    for m in RIBSEG_DEFAULT_METRICS:
        print("  {}_fg: {:.4f}".format(m, summary["{}_fg".format(m)]))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage1-pred", type=str, required=True,
                        help="directory of stage-1 {vid}.npz, i.e. <stage1_log>/result")
    parser.add_argument("--stage2-pred", type=str, required=True,
                        help="directory of stage-2 {vid}.npz, i.e. <stage2_log>/result")
    parser.add_argument("--cache-root", type=str, default="data/ribsegv2/pt_preproc",
                        help="the one-stage cache: 25-class ground truth, and the scored point set")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--save-path", type=str, default=".", help="where to write the two reports")
    parser.add_argument("--volume-ids", type=int, nargs="+", default=None,
                        help="score only these volumes instead of the whole split (smoke runs)")
    return parser.parse_args(argv)


if "__main__" == __name__:
    args = parse_args()
    combine_split(
        stage1_pred=args.stage1_pred,
        stage2_pred=args.stage2_pred,
        cache_root=args.cache_root,
        split=args.split,
        save_path=args.save_path,
        volume_ids=args.volume_ids,
    )

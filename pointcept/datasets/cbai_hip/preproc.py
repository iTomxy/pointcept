import os, json, time, glob, argparse
import numpy as np
import nibabel as nib
from ..transform import Compose
from ...utils.misc import np_smallest_dtype

def default_split_path(data_root):
    """The split cache lives next to the data (./data/cbai_hip/), not inside
    the source package -- the package dir is baked into docker images."""
    return os.path.join(data_root, "split.json")


def get_splits(split_path=None):
    """Load the cached split json; raises if it does not exist yet."""
    if split_path is None:
        split_path = default_split_path("data/cbai_hip")
    with open(split_path) as f:
        splits = json.load(f)
    splits["all"] = splits["train"] + splits["val"] + splits["test"]
    return splits


def make_split(
        data_root="data/cbai_hip", split_path=None, val_ratio=0.2, seed=0, force=False):
    """Separate a validation set from the training set and cache it as a json.

    cbai_hip ships with only training and test splits (imagesTr/labelsTr and
    imagesTs/labelsTs, see README.md). This carves `val_ratio` of the training
    volumes out as a validation set with a seeded random shuffle, so downstream
    preprocess/grid-stats all agree on the same train/val/test split.

    The result is cached in `split_path` (default: {data_root}/split.json).
    Idempotent: an existing cache file is reused as-is, unless `force=True`.

    Args:
        data_root: str, path to the dataset root (contains labelsTr/labelsTs)
        split_path: str, where to cache the split json
        val_ratio: float, fraction of the training volumes held out as validation
        seed: int, RNG seed for the shuffle (for reproducibility)
        force: bool, rebuild the split even if the cache file already exists
    """
    if split_path is None:
        split_path = default_split_path(data_root)

    def list_vids(folder):
        pattern = os.path.join(data_root, folder, "*.nii.gz")
        return sorted(os.path.basename(f)[: -len(".nii.gz")] for f in glob.glob(pattern))

    if os.path.isfile(split_path) and not force:
        print("reuse cached split: {}".format(split_path))
        return get_splits(split_path)

    train_all = list_vids("labelsTr")
    test = list_vids("labelsTs")
    assert train_all and test, "no volumes found under {}/{{labelsTr,labelsTs}}".format(data_root)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_all))
    n_val = int(round(len(train_all) * val_ratio))
    val = [train_all[i] for i in perm[:n_val]]
    train = [train_all[i] for i in perm[n_val:]]

    splits = {
        "train": train, "val": val, "test": test,
        "val_ratio": val_ratio, "seed": seed,
        "time": time.asctime(time.gmtime()),
    }
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=1)
    print("written split -> {}".format(split_path))
    return get_splits(split_path)


def preprocess_ptcloud(
        data_root="data/cbai_hip", save_path="data/cbai_hip/pt_preproc",
        hu_thres=200, clip_percentile=(0.5, 99.5), ids=None, split_path=None):
    """Pre-compute ReadNifti -> Reorient -> CT2PointCloud and cache each volume
    as a compressed .npz, so training only pays for a np.load instead of
    decoding a NIfTI per sample. `ReadNpz` reads these back.

    Same pipeline and saving format as `ribsegv2/preproc.py`; see its docstring
    for the layout of the saved arrays. Train/val volumes are read from
    `{data_root}/imagesTr/{vid}_0000.nii.gz` + `{data_root}/labelsTr/{vid}.nii.gz`
    and test volumes from the `imagesTs/labelsTs` folders, per the cached split.

    Args:
        data_root: str, path to the dataset root (contains imagesTr/labelsTr/imagesTs/labelsTs)
        save_path: str, where to write the .npz files
        hu_thres: float, HU threshold for CT2PointCloud (must match the training config)
        clip_percentile: Tuple[float, float], percentile range for intensity clipping
        ids: List[str] = None, which volumes to convert; None = all volumes in the cached split
        split_path: str, path to the split cache json
    """
    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)
    if split_path is None:
        split_path = default_split_path(data_root)
    read = Compose([
        dict(type="ReadNifti", keys=(("intensity", "f4"), ("label", "i4")), meta_key="intensity"),
        # keep the reorientation: it fixes what `voxel_index` means
        dict(type="Reorient", keys=("intensity", "label"), new_ornt="LPS"),
    ])
    to_points = Compose([
        dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("label", "intensity")),
    ])

    splits = make_split(data_root=data_root, split_path=split_path)
    split_of = {v: "Tr" for v in splits["train"] + splits["val"]}
    split_of.update({v: "Ts" for v in splits["test"]})
    if ids is None:
        ids = splits["train"] + splits["val"] + splits["test"]

    for vid in ids:
        out_path = os.path.join(save_path, "{}.npz".format(vid))
        if os.path.isfile(out_path):
            continue
        suffix = split_of.get(vid)
        assert suffix is not None, \
            "volume {} not found in the cached split ({})".format(vid, split_path)
        data_dict = dict(
            intensity=os.path.join(data_root, "images{}".format(suffix), "{}_0000.nii.gz".format(vid)),
            label=os.path.join(data_root, "labels{}".format(suffix), "{}.nii.gz".format(vid)),
        )

        print(vid, end='\r')
        data = read(data_dict)
        # statistics of the percentile-clipped WHOLE volume, taken before
        # sieving so they do not depend on the HU threshold. Cached because
        # `NormalizeIntensityCached` cannot recompute them from the points.
        lo, hi = np.percentile(data["intensity"], clip_percentile)
        clipped = np.clip(data["intensity"], lo, hi)
        mean, std = float(clipped.mean()), float(clipped.std())
        del clipped

        data = to_points(data)
        hu = data["intensity"]
        # ribsegv2 asserts HU is integral (DICOM-derived int pixels) and caches
        # intensity as a narrow integer. The cbai_hip CBCT exports store
        # fractional float HU instead (float32 voxels behind an int16 header
        # tag), so round-tripping them through an integer would lose precision.
        if np.array_equal(hu, np.rint(hu)):
            intensity = np_smallest_dtype(np.rint(hu).astype(np.int64))
        else:
            intensity = hu.astype(np.float32)
        np.savez_compressed(
            out_path,
            affine=data["affine"].astype(np.float64),
            label=np_smallest_dtype(data["label"].astype(np.int64)),
            intensity=intensity,
            intensity_min=float(lo),
            intensity_max=float(hi),
            intensity_mean=mean,
            intensity_std=std,
            voxel_index=np_smallest_dtype(data["voxel_index"].astype(np.int64)),
            nifti_shape=np_smallest_dtype(np.asarray(data["nifti_shape"], dtype=np.int64)),
        )


def grid_stats(
        data_root="data/cbai_hip", save_path='', grid_size_mm=3.0, split="train", n_volumes=0,
        # PointTransformerV3 bound settings, cf. the PTv3 config
        stride=2, n_stages=5, patch_size=1024, split_path=None):
    """How many points survive a grid sub-sampling, and what the PTv3 encoder
    then does with them. Same measurement as `ribsegv2/grid_stats`.

    Read the output as: stage counts should fall by roughly `stride`^3 each
    time, and the last stage should hold fewer points than `patch_size` so that
    the top of the U-Net is a single, globally-connected attention window.
    `max_points` in the config should sit above the p99 of `n_points`.

    Args:
        data_root: str, path to the dataset root (preprocessed .npz under {data_root}/pt_preproc)
        save_path: str = '', where to dump the summary + per-volume numbers as
            json. Defaults to {data_root}/grid-stats-{split}-{grid_size_mm}mm.json;
            pass None to skip writing.
        grid_size_mm: float, grid size in mm (the config stores it divided by `global_radius`)
        split: str, which split to measure, one of train/val/test/all (from the cached split json)
        n_volumes: int = 0, measure a random subset of this size; 0 = all
        stride: int = 2, PTv3 `stride`, i.e. the pooling factor per axis per stage
        n_stages: int = 5, len(enc_depths) of PTv3 cfg
        patch_size: int = 1024, PTv3 `enc_patch_size`, the attention window size
    """
    if split_path is None:
        split_path = default_split_path(data_root)
    splits = make_split(data_root=data_root, split_path=split_path)
    ids = [v for v in splits[split]]
    if n_volumes and n_volumes < len(ids):
        ids = np.random.default_rng(0).choice(ids, n_volumes, replace=False).tolist()

    def n_cells(coord, size):
        gc = np.floor(coord / size).astype(np.int64)
        gc -= gc.min(0)
        span = gc.max(0) + 1
        key = (gc[:, 0] * span[1] + gc[:, 1]) * span[2] + gc[:, 2]
        return int(np.unique(key).size)

    records = []
    for vid in ids:
        f = os.path.join(data_root, "pt_preproc", "{}.npz".format(vid))
        if not os.path.isfile(f):
            continue
        print(vid, end='\r')
        with np.load(f) as npz:
            if "affine" in npz:
                # Match the runtime cache path: reconstruct from the float64
                # affine, then narrow to the historical float32 representation.
                coord = nib.affines.apply_affine(
                    npz["affine"], npz["voxel_index"].astype(np.float32)
                ).astype(np.float32).astype(np.float64)
            else:
                coord = npz["coord"].astype(np.float64)
            label = npz["label"]
        stages = [n_cells(coord, grid_size_mm * (stride ** s)) for s in range(n_stages)]
        records.append({
            "vid": vid,
            "n_points_full": int(coord.shape[0]),
            "n_points": stages[0],
            "fg_ratio_full": float((label > 0).mean()),
            "stage_n_points": stages,
            "stage_cell_mm": [float(grid_size_mm * (stride ** s)) for s in range(n_stages)],
        })

    assert len(records) > 0, "No cached volume found under {}/pt_preproc".format(data_root)
    n_pts = np.asarray([r["n_points"] for r in records], dtype=float)
    stage_n = np.asarray([r["stage_n_points"] for r in records], dtype=float)
    summary = {
        "grid_size_mm": grid_size_mm,
        "split": split,
        "n_volumes": len(records),
        "n_points": {
            "mean": float(n_pts.mean()), "std": float(n_pts.std()),
            "min": float(n_pts.min()), "max": float(n_pts.max()),
            "p50": float(np.percentile(n_pts, 50)), "p99": float(np.percentile(n_pts, 99)),
        },
        "stage_n_points_mean": stage_n.mean(0).round(1).tolist(),
        "stage_reduction_mean": (stage_n[:, :-1] / np.clip(stage_n[:, 1:], 1, None)).mean(0).round(2).tolist(),
        "last_stage_is_global": bool((stage_n[:, -1] <= patch_size).all()),
        "last_stage_patches_max": float(np.ceil(stage_n[:, -1].max() / patch_size)),
        "suggested_max_points": int(np.ceil(np.percentile(n_pts, 99) / 1000.0) * 1000),
    }
    print(json.dumps(summary, indent=1))
    if save_path == '':
        save_path = os.path.join(
            data_root, "grid-stats-{}-{}mm.json".format(split, grid_size_mm)
        )
    if save_path:
        save_path = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({"time": time.asctime(time.gmtime()), "summary": summary, "volumes": records}, f, indent=1)
        print("Written to {}".format(save_path))

    return summary


if "__main__" == __name__:
    # Usage: python -m pointcept.datasets.cbai_hip.preproc [ARGS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/cbai_hip")
    parser.add_argument("--save-path", type=str, default="")

    # separate a validation set from the training set, cache the split json
    parser.add_argument("--split-data", action="store_true", help="(re)build & cache the train/val/test split json.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)

    # convert CT volumes to point clouds & cache as npz
    parser.add_argument("--preproc", action="store_true", help="preprocess & cache data in npz format.")
    parser.add_argument("--hu-thres", type=float, default=200)

    # point count & PTv3 encoder stage sizes at a given grid size
    parser.add_argument("--grid-stats", action="store_true", help="report point counts per encoder stage at a grid size.")
    parser.add_argument("--grid-size-mm", type=float, default=3.0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-volumes", type=int, default=0, help="0 = every volume in the split")
    args = parser.parse_args()

    if args.split_data:
        make_split(data_root=args.data_root, val_ratio=args.val_ratio, seed=args.split_seed, force=True)

    if args.preproc:
        preprocess_ptcloud(
            data_root=args.data_root,
            save_path=args.save_path or os.path.join(args.data_root, "pt_preproc"),
            hu_thres=args.hu_thres,
        )

    if args.grid_stats:
        grid_stats(
            data_root=args.data_root,
            save_path=args.save_path,
            grid_size_mm=args.grid_size_mm,
            split=args.split,
            n_volumes=args.n_volumes,
        )

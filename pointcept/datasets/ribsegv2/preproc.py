import os, json, time, argparse
import numpy as np
import nibabel as nib
from ..transform import Compose, reorient_points, determine_reorient
from ...utils.misc import np_smallest_dtype
from .base import IGNORE_VOLUMES, SPLITS


def preprocess_ptcloud(
    data_root="data/ribsegv2", save_path="data/ribsegv2/pt_preproc",
    hu_thres=200, clip_percentile=(0.5, 99.5), binpred_path='', ids=None):
    """Pre-compute ReadNifti -> Reorient -> CT2PointCloud and
    cache each volume as a compressed .npz, so training only pays for a np.load
    instead of decoding a NIfTI per sample. `ReadNpz` reads these back.

    Saved arrays per volume ({save_path}/{vid}.npz). Every integer field is
    stored in the narrowest dtype that holds it (`np_smallest_dtype`), so the
    exact types depend on the data -- with hu_thres=200 they come out as:
        affine       float64 [4, 4]  voxel-to-physical transform, LPS
        label        uint8   [N]     per-point class id, 0 = background
        intensity    uint16  [N]     raw HU value (signed only if hu_thres < 0)
        voxel_index  uint16  [N, 3]  original voxel indices, to write predictions back
        nifti_shape  uint16  [3]     original volume shape
        intensity_{min|max|mean|std} float, stats of the clipped volume

    Two deliberate choices about `intensity`:

    - It is stored as an integer. CT reconstruction is quantised on output --
      DICOM keeps integer pixel data with an integer RescaleIntercept -- so HU
      is exactly integral, verified here by an assertion rather than assumed.
      The cast happens at save time, not in `ReadNifti`: here the values are
      known to be above the HU threshold, whereas a full volume also holds the
      out-of-FOV padding, and `.astype` truncates and overflows silently.
    - The percentile z-score is NOT cached as an array. The four scalars below
      reproduce it exactly through the `NormalizeIntensityCached` transform, at
      four floats per volume instead of one float32 per point (~22% smaller).
      They have to be cached because they describe the whole volume and cannot
      be recovered once the cloud is sieved to HU > threshold.

    Prefer feeding `intensity` through `WindowIntensity` with a fixed HU range:
    the percentile z-score clips near ~500 HU, so cortical bone saturates, and
    its scale drifts between volumes with the field of view.

    Args:
        data_root: str, path to the dataset root
        save_path: str, where to write the .npz files
        hu_thres: float, HU threshold for CT2PointCloud (must match the training config)
        clip_percentile: Tuple[float, float], percentile range for intensity clipping
        binpred_path: str, if set, also sieve voxels with this stage-1 prediction
            ({binpred_path}/{vid}.nii.gz, {-1: ignored, 0: bg, 1: fg})
        ids: List[int] = None, which volumes to convert; None = all of them.
            Pass a slice of the ids to shard the job across processes.
    """
    if binpred_path:
        binpred_path = os.path.expanduser(binpred_path)
        assert os.path.isdir(binpred_path), "binpred_path ({}) not exists".format(binpred_path)
        read_keys = (("intensity", "f4"), ("label", "i4"), ("sieve_mask", "i1"))
        reorient_keys = ("intensity", "label", "sieve_mask")
    else:
        read_keys = (("intensity", "f4"), ("label", "i4"))
        reorient_keys = ("intensity", "label")

    save_path = os.path.expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)
    read = Compose([
        dict(type="ReadNifti", keys=read_keys, meta_key="intensity"),
        # keep the reorientation: it fixes what `voxel_index` means
        dict(type="Reorient", keys=reorient_keys, new_ornt="LPS"),
    ])
    to_points = Compose([
        dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("label", "intensity")),
    ])

    if ids is None:
        ids = [i for i in range(1, 661) if i not in (452, 485, 490)]
    for vid in ids:
        out_path = os.path.join(save_path, "{}.npz".format(vid))
        if os.path.isfile(out_path):
            continue
        data_dict = dict(
            intensity=os.path.join(data_root, "image", "RibFrac{}-image.nii.gz".format(vid)),
            label=os.path.join(data_root, "label", "RibFrac{}-rib-seg.nii.gz".format(vid)),
        )
        if binpred_path:
            data_dict["sieve_mask"] = os.path.join(binpred_path, "{}.nii.gz".format(vid))
            if not os.path.isfile(data_dict["sieve_mask"]):
                continue

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
        assert np.array_equal(hu, np.rint(hu)), (
            "volume {}: HU values are not integral, so storing them as an integer would "
            "lose precision. Check the NIfTI scl_slope/scl_inter of this scan.".format(vid)
        )
        np.savez_compressed(
            out_path,
            affine=data["affine"].astype(np.float64),
            label=np_smallest_dtype(data["label"].astype(np.int64)),
            intensity=np_smallest_dtype(np.rint(hu).astype(np.int64)),
            intensity_min=float(lo),
            intensity_max=float(hi),
            intensity_mean=mean,
            intensity_std=std,
            voxel_index=np_smallest_dtype(data["voxel_index"].astype(np.int64)),
            nifti_shape=np_smallest_dtype(np.asarray(data["nifti_shape"], dtype=np.int64)),
        )


def grid_stats(
        data_root="data/ribsegv2", save_path='', grid_size_mm=3.0, split="train", n_volumes=0,
        # PointTransformerV3 bound settings, cf. the PTv3 config
        stride=2, n_stages=5, patch_size=1024
):
    """How many points survive a grid sub-sampling, and what the PTv3 encoder
    then does with them.

    Pick `grid_size` with this, not by eye. `grid_size` is not only a
    de-duplication radius: `SerializedPooling` pools by shifting the serialised
    code built from `grid_coord`, so each encoder stage merges `stride`^3 cells
    of the previous one. Set the grid too fine and every stage becomes a no-op,
    the point count never drops, the deepest attention window never covers the
    whole rib cage, and assigning a rib its index 1..24 -- an ordinal task that
    needs global context -- becomes impossible.

    Read the output as: stage counts should fall by roughly `stride`^3 each
    time, and the last stage should hold fewer points than `patch_size` so that
    the top of the U-Net is a single, globally-connected attention window.
    `max_points` in the config should sit above the p99 of `n_points`.

    Args:
        data_root: str, path to the dataset root
        save_path: str = '', where to dump the summary + per-volume numbers as
            json. Defaults to {data_root}/grid-stats-{split}-{grid_size_mm}mm.json
            so a measurement is always kept for later reference; pass None to
            skip writing.
        grid_size_mm: float, grid size in mm (the config stores it divided by `global_radius`)
        split: str, which split to measure, one of train/val/test/all
        n_volumes: int = 0, measure a random subset of this size; 0 = all
        stride: int = 2, PTv3 `stride`, i.e. the pooling factor per axis per stage
        n_stages: int = 5, len(enc_depths) of PTv3 cfg
        patch_size: int = 1024, PTv3 `enc_patch_size`, the attention window size
    """
    ids = [v for v in SPLITS[split] if v not in IGNORE_VOLUMES]
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
            "vid": int(vid),
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
        with open(save_path, 'w') as f:
            json.dump({"time": time.asctime(time.gmtime()), "summary": summary, "volumes": records}, f, indent=1)
        print("Written to {}".format(save_path))

    return summary


def recon_3d_bin_pred(pred_path, save_path, pred_ornt="LPS", data_root="data/ribsegv2"):
    """reconstruct the 1st stage binary segmentation to 3D volumes for all data
    The orientation should be consistent with their original image.
    Reconstructed voxel value: {-1: not predicted, 0: bg, 1: fg}
    Args:
        pred_path: str, path to the original fg-bg prediction (.npz)
        save_path: str, path to save the reconstructed volumes (.nii.gz)
        pred_ornt: str = "LPS", orientation of the predicted point clouds
        data_root: str = "data/ribsegv2"
    """
    import open3d as o3d

    os.makedirs(save_path, exist_ok=True)
    for f in os.listdir(pred_path):
        if not f.endswith(".npz"):
            continue
        vid = int(f[:-4])
        print(vid, end='\r')
        img_f = os.path.join(data_root, "image", "RibFrac{}-image.nii.gz".format(vid))
        img_nii = nib.load(img_f)
        img_ornt = nib.aff2axcodes(img_nii.affine)

        data = np.load(os.path.join(pred_path, f))
        # widen to a signed int: predictions are cached in the narrowest dtype
        # that fits, and the flip in `reorient_points` is subtraction
        xyz = data["voxel_index"].astype(np.int32) # [n_batch, npt, 3] (or [n_batch, 3, npt])
        pred = data["pred"] # [n_batch, npt]
        # print(xyz.shape, pred.shape)
        assert 0 <= pred.min() and pred.max() <= 1, "Invalid binary prediction value range: [{}, {}]".format(pred.min(), pred.max())
        if pred.ndim > 1:
            pred = pred.reshape(-1)
            assert 3 == xyz.ndim, "voxel_index shape: {}".format(xyz.shape)
            if 3 == xyz.shape[1]: # [n_batch, 3, npt]
                xyz = xyz.transpose(0, 2, 1) # -> [n_batch, npt, 3]
            xyz = xyz.reshape(-1, xyz.shape[-1])

        need_reorient, axis_order_pred, _ = determine_reorient(img_ornt, pred_ornt)
        if need_reorient:
            # determine shape in predicted orientation <- used in reorient_points
            shape_pred = tuple(img_nii.shape[axis_order_pred[i]] for i in range(3))
            axes_range_pred = tuple((0, shape_pred[i]-1) for i in range(3))
            # reorient indices to original image orientation
            xyz = reorient_points(xyz, pred_ornt, img_ornt, axes_range_pred).astype(np.int32)

        # remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        xyz = xyz[ind]
        pred = pred[ind]

        # {-1: not predicted, 0: bg, 1: fg}
        pred_3d = np.zeros(img_nii.shape, dtype=np.int8) - 1
        pred_3d[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = pred.astype(np.int8)

        pred_nii = nib.Nifti1Image(pred_3d, img_nii.affine, img_nii.header)
        nib.save(pred_nii, os.path.join(save_path, "{}.nii.gz".format(vid)))


if "__main__" == __name__:
    # Usage: python -m pointcept.datasets.ribsegv2.preproc [ARGS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/ribsegv2")
    parser.add_argument("--save-path", type=str, default="")

    # convert CT volumes to point clouds & cache as npz
    parser.add_argument("--preproc", action="store_true", help="preprocess & cache data in npz format.")
    parser.add_argument("--hu-thres", type=float, default=200)

    # point count & PTv3 encoder stage sizes at a given grid size
    parser.add_argument("--grid-stats", action="store_true", help="report point counts per encoder stage at a grid size.")
    parser.add_argument("--grid-size-mm", type=float, default=3.0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-volumes", type=int, default=0, help="0 = every volume in the split")

    # reconstruct 3D voxel-grids of binary (rib vs. non-rib) predictions from point-wise ones
    parser.add_argument("--recon-3d-bin", action="store_true", help="reconstruct stage-1 binary prediction to 3D volume.")
    parser.add_argument("--bin-pred-path", type=str, default="", help="path to stage-1 binary (rib vs. non-rib) point-wise predictions")
    parser.add_argument("--bin-recon-path", type=str, default="", help="path to save the reconstructed 3D binary voxel-grids")
    parser.add_argument("--ornt", type=str, default="LPS", help="orientation of the point-wise predictions")
    args = parser.parse_args()

    if args.preproc:
        assert args.save_path, "--save-path not specified for preproc"
        preprocess_ptcloud(
            data_root=args.data_root,
            save_path=args.save_path,
            hu_thres=args.hu_thres,
            binpred_path=args.bin_pred_path,
        )

    if args.grid_stats:
        grid_stats(
            data_root=args.data_root,
            save_path=args.save_path,
            grid_size_mm=args.grid_size_mm,
            split=args.split,
            n_volumes=args.n_volumes
        )

    if args.recon_3d_bin:
        recon_3d_bin_pred(
            pred_path=args.bin_pred_path,
            save_path=args.bin_recon_path,
            pred_ornt=args.ornt,
            data_root=args.data_root
        )

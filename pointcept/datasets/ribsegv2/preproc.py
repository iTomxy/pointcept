import os, json, time, argparse
import numpy as np
import nibabel as nib
from ..transform import Compose, reorient_points, determine_reorient, Reorient, remove_statistical_outlier
from ...utils.misc import np_smallest_dtype, staging_dir, publish_dir
from .base import IGNORE_VOLUMES, SPLITS


# Volumes excluded from preprocessing. Deliberately NARROWER than
# `base.IGNORE_VOLUMES`: these three are unreadable / wrongly shaped, whereas
# the extra ids base.py drops are only suspected of bad labels. Caching the
# superset means revisiting that judgement costs no recomputation.
PREPROC_IGNORE_VOLUMES = (452, 485, 490)
ALL_VOLUMES = tuple(range(1, 661))


def _load_sieve(sieve_path, ref_affine, ref_shape, vid):
    """Load a stage-1 sieve onto the grid `ref_affine`/`ref_shape` describe.

    The sieve only means anything voxel-for-voxel against the volume it
    filters, and `recon_3d_bin_pred` can write it in either of two frames: the
    scan's own orientation, or -- under `cache_root` -- the cache's LPS.
    Reorienting by the sieve's OWN axcodes (not the volume's) is what makes
    both line up, and it is why the sieve must NOT be carried through the
    shared `Reorient` alongside the image: that call derives `old_ornt` from
    whatever affine it is given, so a sieve already in LPS would be turned
    OUT of it if it rode along with the image's affine.

    The affine check below is what actually pins the correspondence.
    RibSegv2 volume 491 is RAS where all 659 others are LPS (verified against
    `data/ribsegv2/image` in this repo), and RAS -> LPS is a pure flip: a
    sieve left in the wrong frame keeps its shape, indexes without raising,
    and silently samples the mirrored voxel. Measured on 491, that quietly
    turns 398,965 kept points into 10,463. A shape comparison alone would not
    catch it.

    Also note the 1e-4 tolerance below: a NIfTI stores its srow fields as
    float32, so a round trip through .nii.gz costs a few 1e-7 on translations
    of a few hundred mm; 1e-4 is far below one voxel (~0.7mm here) and far
    above that.

    Input:
        sieve_path: str, path to the stage-1 sieve NIfTI ({-1, 0, 1} int8)
        ref_affine: float[4, 4], affine of the volume this sieve filters
        ref_shape: int[3], nifti_shape of the volume this sieve filters
        vid: int, volume id, used in error messages only
    Output:
        sieve: int8[H, W, L], the sieve reoriented onto `ref_affine`/`ref_shape`
    """
    nii = nib.load(sieve_path)
    # dataobj, not get_fdata: the latter materialises a float64 copy of the
    # whole 512x512xZ volume in order to carry three distinct int8 values.
    arr = np.asanyarray(nii.dataobj).astype(np.int8)
    # Reorient by the sieve's OWN geometry, as a private dict -- not through
    # the `read`/`Reorient` step used for the image, which would reorient by
    # the image's affine instead (see docstring above).
    data = Reorient(keys=("sieve_mask",), new_ornt="LPS")({
        "sieve_mask": arr,
        "affine": nii.affine.copy(),
        "nifti_shape": nii.shape,
    })
    # `Reorient` only rewrites "nifti_shape" when a reorientation is actually
    # needed, so this is valid to read either way.
    assert tuple(data["nifti_shape"]) == tuple(ref_shape), (
        "Volume {}: sieve {} is {} after reorientation but the volume it "
        "filters is {}".format(vid, sieve_path, tuple(data["nifti_shape"]), tuple(ref_shape))
    )
    assert np.allclose(data["affine"], ref_affine, atol=1e-4), (
        "Volume {}: sieve {} does not sit on the same grid as the volume it "
        "filters. Reoriented sieve affine:\n{}\nexpected:\n{}\nThe sieve and "
        "the points it selects must come from the same scan.".format(
            vid, sieve_path, data["affine"], np.asarray(ref_affine)
        )
    )
    return data["sieve_mask"]


def _write_cache_npz(out_path, vid, affine, voxel_index, label, intensity, nifti_shape, stats):
    """Write one cache entry in the layout `preprocess_ptcloud` documents below.

    Shared by both routes into the cache (raw scan or `cache_root`) so that a
    change to the on-disk contract cannot land on one of them and miss the
    other.
    """
    assert np.all(np.isfinite(intensity)), "volume {}: intensity contains non-finite values".format(vid)
    assert np.array_equal(intensity, np.rint(intensity)), (
        "volume {}: HU values are not integral, so storing them as an integer would "
        "lose precision. Check the NIfTI scl_slope/scl_inter of this scan.".format(vid)
    )
    np.savez_compressed(
        out_path,
        affine=np.asarray(affine).astype(np.float64),
        label=np_smallest_dtype(label.astype(np.int64)),
        intensity=np_smallest_dtype(np.rint(intensity).astype(np.int64)),
        voxel_index=np_smallest_dtype(voxel_index.astype(np.int64)),
        nifti_shape=np_smallest_dtype(np.asarray(nifti_shape, dtype=np.int64)),
        **stats,
    )


def preprocess_ptcloud(
    data_root="data/ribsegv2", save_path="data/ribsegv2/pt_preproc",
    hu_thres=200, clip_percentile=(0.5, 99.5), binpred_path='', ids=None,
    min_points=1, cache_root=''):
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

    Usage (stage-2 re-cache, re-filtering an existing point-cloud cache
    through a stage-1 sieve instead of re-reading the raw scans):
        python -m pointcept.datasets.ribsegv2.preproc --preproc \
            --bin-pred-path <stage1_log>/recon-3d-binpred \
            --save-path <stage1_log>/pt_preproc-binpred \
            --cache-root data/ribsegv2/pt_preproc

    Args:
        data_root: str, path to the dataset root
        save_path: str, where to write the .npz files
        hu_thres: float, HU threshold for CT2PointCloud (must match the training config)
        clip_percentile: Tuple[float, float], percentile range for intensity clipping
        binpred_path: str, if set, also sieve voxels with this stage-1 prediction
            ({binpred_path}/{vid}.nii.gz, {-1: ignored, 0: bg, 1: fg})
        ids: List[int] = None, which volumes to convert; None = all of them.
            Pass a slice of the ids to shard the job across processes.
        min_points: int = 1, a volume yielding fewer kept points than this is
            not cached and is reported at the end instead. Caching an empty
            (or near-empty) cloud would push the failure into a stage-2
            dataloader worker, where `GridSample` and `NormalizeCoord` reject
            it far from the cause.
        cache_root: str = '', an existing point-cloud cache to re-filter
            through `binpred_path`, instead of re-reading the raw scans. The
            points, labels and whole-volume statistics are all already in
            those entries, so a second-stage re-cache never touches `image/`
            or `label/`. Requires `binpred_path`, and must not resolve to
            `save_path`: the stage-1 cache is an input here and is never
            written to.

    Raises:
        RuntimeError: if any volume fell under `min_points`. Every other
            volume is still written first, so the run reports the whole list
            at once and a re-run has no work to redo.
    """
    binpred_path = os.path.expanduser(binpred_path) if binpred_path else ''
    if binpred_path:
        assert os.path.isdir(binpred_path), "binpred_path ({}) not exists".format(binpred_path)

    cache_root = os.path.expanduser(cache_root) if cache_root else ''
    if cache_root:
        # cache_root only re-filters an existing cache through a stage-1
        # sieve, so it makes no sense without one.
        assert binpred_path, "cache_root ({}) requires binpred_path to be set".format(cache_root)
        assert os.path.isdir(cache_root), "cache_root ({}) not exists".format(cache_root)

    save_path = os.path.expanduser(save_path)

    if cache_root:
        # Without this check the run is not merely wrong but invisible: every
        # id would find its own entry already at out_path, skip as "done",
        # and report a complete re-cache that is really the untouched
        # one-stage cache. The stage-1 cache is shared by every backbone and
        # repetition, so it must not be somewhere a second-stage run can
        # write -- conventionally <stage-1 log>/pt_preproc-binpred.
        assert os.path.realpath(save_path) != os.path.realpath(cache_root), (
            "save_path and cache_root resolve to the same directory ({}). A second-stage "
            "re-cache reads the stage-1 cache and must be written somewhere else -- "
            "conventionally <stage-1 log>/pt_preproc-binpred.".format(cache_root)
        )
    assert min_points >= 1, "min_points must be at least 1, got {}".format(min_points)

    # `save_path` appears only once every volume is in it, so a consumer can
    # read its existence as completeness instead of counting entries. An
    # explicit `ids` is the exception: that is a shard or a manual subset, the
    # caller decides what "complete" means, and two shards renaming one staging
    # directory onto the same target would collide -- so those write straight
    # through.
    staged = ids is None
    if staged:
        if os.path.isdir(save_path):
            print("Already complete, nothing to do: {}".format(save_path))
            return
        work_path = staging_dir(save_path)
    else:
        work_path = save_path
        os.makedirs(work_path, exist_ok=True)

    # `read`/`to_points` only ever handle `intensity` and `label`: the sieve
    # (when there is one) is loaded separately by `_load_sieve`, below, so
    # that it is reoriented by its OWN affine rather than the image's.
    read = Compose([
        dict(type="ReadNifti", keys=(("intensity", "f4"), ("label", "i4")), meta_key="intensity"),
        # keep the reorientation: it fixes what `voxel_index` means
        dict(type="Reorient", keys=("intensity", "label"), new_ornt="LPS"),
    ])
    to_points = Compose([
        dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("label", "intensity")),
    ])

    underfull = {}
    for vid in (ALL_VOLUMES if ids is None else ids):
        if vid in PREPROC_IGNORE_VOLUMES:
            continue
        out_path = os.path.join(work_path, "{}.npz".format(vid))
        if os.path.isfile(out_path):
            continue

        sieve_path = None
        if binpred_path:
            sieve_path = os.path.join(binpred_path, "{}.nii.gz".format(vid))
            if not os.path.isfile(sieve_path):
                continue

        print(vid, end='\r')
        if cache_root:
            # Second stage. The points, their labels and the full-volume
            # statistics are all already in the stage-1 entry, and its
            # voxel_index indexes the very grid the sieve has to be read
            # onto, so the raw scan is never opened on this route.
            cache_file = os.path.join(cache_root, "{}.npz".format(vid))
            assert os.path.isfile(cache_file), "Volume {}: no cache entry at {}".format(vid, cache_file)
            with np.load(cache_file) as cache_data:
                affine = cache_data["affine"]
                nifti_shape = tuple(cache_data["nifti_shape"].tolist())
                voxel_index = cache_data["voxel_index"]
                intensity = cache_data["intensity"]
                label = cache_data["label"]
                # Whole-volume statistics from before stage 1 thresholded
                # anything. They describe the scan, so the sieve must not
                # change them -- carried through unmodified.
                stats = {k: cache_data[k] for k in ("intensity_min", "intensity_max", "intensity_mean", "intensity_std")}

            grid = _load_sieve(sieve_path, affine, nifti_shape, vid)
            # `>`, matching CT2PointCloud.sieve's `>`: the entry was built at
            # some threshold already, so this only bites when stage 2 raises
            # it, and then the two routes have to agree on the boundary value.
            keep = (intensity > hu_thres) & (
                grid[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] == 1
            )
            voxel_index, intensity, label = voxel_index[keep], intensity[keep], label[keep]
        else:
            data_dict = dict(
                intensity=os.path.join(data_root, "image", "RibFrac{}-image.nii.gz".format(vid)),
                label=os.path.join(data_root, "label", "RibFrac{}-rib-seg.nii.gz".format(vid)),
            )
            data = read(data_dict)
            # statistics of the percentile-clipped WHOLE volume, taken before
            # sieving so they do not depend on the HU threshold. Cached because
            # `NormalizeIntensityCached` cannot recompute them from the points.
            lo, hi = np.percentile(data["intensity"], clip_percentile)
            clipped = np.clip(data["intensity"], lo, hi)
            stats = dict(
                intensity_min=float(lo), intensity_max=float(hi),
                intensity_mean=float(clipped.mean()), intensity_std=float(clipped.std()),
            )
            del clipped

            if sieve_path is not None:
                # Loaded by its OWN affine, not through `read` above (see
                # `_load_sieve`'s docstring for why).
                data["sieve_mask"] = _load_sieve(sieve_path, data["affine"], data["nifti_shape"], vid)

            data = to_points(data)
            affine, nifti_shape = data["affine"], data["nifti_shape"]
            voxel_index, intensity, label = data["voxel_index"], data["intensity"], data["label"]

        num_points = voxel_index.shape[0]
        if num_points < min_points:
            underfull[vid] = num_points
            continue

        _write_cache_npz(out_path, vid, affine, voxel_index, label, intensity, nifti_shape, stats)

    if underfull:
        listed = ", ".join("{} ({} points)".format(vid, count) for vid, count in sorted(underfull.items()))
        reason = (
            "Stage 1 predicted (almost) no foreground for them; drop them from the "
            "stage-2 splits or revisit the stage-1 checkpoint."
            if binpred_path else
            "Check hu_thres and the source volumes."
        )
        # Raised BEFORE publishing, so the staging directory stays unpublished
        # and the next run neither mistakes it for finished nor redoes the
        # volumes it already cached.
        raise RuntimeError(
            "{} volume(s) yielded fewer than {} points and were not cached: {}. {}".format(
                len(underfull), min_points, listed, reason
            )
        )

    if staged:
        publish_dir(work_path, save_path)


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


def recon_3d_bin_pred(pred_path, save_path, pred_ornt="LPS", data_root="data/ribsegv2",
                       cache_root='', nb_neighbors=20, std_ratio=2.0):
    """reconstruct the 1st stage binary segmentation to 3D volumes for all data
    The orientation should be consistent with their original image.
    Reconstructed voxel value: {-1: not predicted, 0: bg, 1: fg}

    Which frame the output lands in depends on where the geometry came from:
    `data_root` puts it in each scan's ORIGINAL orientation, `cache_root` in
    the cache's LPS -- the frame the predictions were already made in, so this
    avoids a round trip. Either is fine downstream, since `_load_sieve`
    reorients by the sieve's own affine and checks the result -- but the two
    are NOT interchangeable within one already-written output directory,
    since existing outputs are skipped rather than rewritten (below).

    Args:
        pred_path: str, path to the original fg-bg prediction (.npz)
        save_path: str, path to save the reconstructed volumes (.nii.gz)
        pred_ornt: str = "LPS", orientation of the predicted point clouds
        data_root: str = "data/ribsegv2"
        cache_root: str = '', take `affine`/`nifti_shape` from this point-cloud
            cache ({cache_root}/{vid}.npz) instead of the raw scan under
            `data_root`, so the raw scan is never opened.
        nb_neighbors, std_ratio: statistical outlier removal parameters, as in
            Open3D's `PointCloud.remove_statistical_outlier` (reimplemented
            without open3d, see `..transform.remove_statistical_outlier`).
    """
    save_path = os.path.expanduser(save_path)
    # Same contract as `preprocess_ptcloud`: the directory appears only once
    # every sieve is in it, so its existence means finished. Interrupted runs
    # resume into the staging directory, since the skip below keeps sieves
    # already written.
    if os.path.isdir(save_path):
        print("Already complete, nothing to do: {}".format(save_path))
        return
    work_path = staging_dir(save_path)
    for f in sorted(os.listdir(pred_path)):
        if not f.endswith(".npz"):
            continue
        vid = int(f[:-4])
        out_path = os.path.join(work_path, "{}.nii.gz".format(vid))
        if os.path.isfile(out_path):
            # so an interrupted run resumes instead of redoing finished volumes
            continue
        print(vid, end='\r')

        if cache_root:
            cache_file = os.path.join(cache_root, "{}.npz".format(vid))
            assert os.path.isfile(cache_file), "Volume {}: no cache entry at {}".format(vid, cache_file)
            with np.load(cache_file) as cache_data:
                affine = cache_data["affine"]
                image_shape = tuple(cache_data["nifti_shape"].tolist())
            image_ornt = nib.aff2axcodes(affine)
        else:
            img_f = os.path.join(data_root, "image", "RibFrac{}-image.nii.gz".format(vid))
            img_nii = nib.load(img_f)
            affine = img_nii.affine
            image_shape = img_nii.shape
            image_ornt = nib.aff2axcodes(affine)

        # `with`: np.load on an .npz keeps the archive's file descriptor open
        # until closed, and this loop runs once per volume (657 in the full
        # dataset) -- leaving it open per iteration risks the soft ulimit -n.
        with np.load(os.path.join(pred_path, f)) as data:
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

        need_reorient, axis_order_pred, _ = determine_reorient(image_ornt, pred_ornt)
        if need_reorient:
            # determine shape in predicted orientation <- used in reorient_points
            shape_pred = tuple(image_shape[axis_order_pred[i]] for i in range(3))
            axes_range_pred = tuple((0, shape_pred[i]-1) for i in range(3))
            # reorient indices to original image orientation
            xyz = reorient_points(xyz, pred_ornt, image_ornt, axes_range_pred).astype(np.int32)

        # remove outliers
        keep = remove_statistical_outlier(xyz, nb_neighbors, std_ratio)
        xyz = xyz[keep]
        pred = pred[keep]

        # {-1: not predicted, 0: bg, 1: fg}
        pred_3d = np.zeros(image_shape, dtype=np.int8) - 1
        pred_3d[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = pred.astype(np.int8)

        # Not `img_nii.header`: reusing the source header keeps its int32
        # datatype -- measured on RibFrac1, that is 1.52 MB per sieve instead
        # of 0.38 MB, i.e. ~1.0 GB instead of ~0.25 GB across the dataset.
        # Values are unaffected.
        pred_nii = nib.Nifti1Image(pred_3d, affine, None)
        nib.save(pred_nii, out_path)

    publish_dir(work_path, save_path)


if "__main__" == __name__:
    # Usage: python -m pointcept.datasets.ribsegv2.preproc [ARGS]
    #   Stage-2 re-cache from a stage-1 point-cloud cache, e.g.:
    #     python -m pointcept.datasets.ribsegv2.preproc --recon-3d-bin \
    #         --bin-pred-path <stage1_log>/result --bin-recon-path <stage1_log>/recon-3d-binpred \
    #         --cache-root data/ribsegv2/pt_preproc
    #     python -m pointcept.datasets.ribsegv2.preproc --preproc \
    #         --bin-pred-path <stage1_log>/recon-3d-binpred --save-path <stage1_log>/pt_preproc-binpred \
    #         --cache-root data/ribsegv2/pt_preproc
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/ribsegv2")
    parser.add_argument("--save-path", type=str, default="")

    # convert CT volumes to point clouds & cache as npz
    parser.add_argument("--preproc", action="store_true", help="preprocess & cache data in npz format.")
    parser.add_argument("--hu-thres", type=float, default=200)
    parser.add_argument("--min-points", type=int, default=1, help="with --preproc, report & skip volumes yielding fewer kept points than this instead of caching them.")
    parser.add_argument("--volume-ids", type=int, nargs="+", default=None, help="with --preproc, process only these volume ids instead of the whole dataset.")

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
    parser.add_argument("--nb-neighbors", type=int, default=20, help="with --recon-3d-bin, statistical outlier removal neighbour count")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="with --recon-3d-bin, statistical outlier removal std ratio")

    parser.add_argument(
        "--cache-root", type=str, default="",
        help="an existing point-cloud cache to source from instead of the raw scans; "
             "with --preproc it re-filters that cache through --bin-pred-path and must not equal --save-path.",
    )
    args = parser.parse_args()

    if args.preproc:
        assert args.save_path, "--save-path not specified for preproc"
        preprocess_ptcloud(
            data_root=args.data_root,
            save_path=args.save_path,
            hu_thres=args.hu_thres,
            binpred_path=args.bin_pred_path,
            ids=args.volume_ids,
            min_points=args.min_points,
            cache_root=args.cache_root,
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
            data_root=args.data_root,
            cache_root=args.cache_root,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )

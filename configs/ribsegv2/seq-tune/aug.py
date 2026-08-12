# Step 2 -- data augmentation, one at a time.
#
# Step 1 found every input-feature variant indistinguishable, so this builds on
# the 1-channel `whu1` baseline. With aug_mode='' the pipeline is byte-identical
# to in_feat-whu1, which already has four seeds -- so the no-aug reference is
# free and this step only needs the eight augmented runs.
#
# Reference for the ranges: ~/codes/tmp.ptcloud/configs/ribsegv2/seq-tune/aug.py
# (the DGCNN sweep on this dataset). Flips are deliberately excluded: ribs 1-12
# are left and 13-24 right, ordered top-down, and RandomFlip only mirrors
# coordinates without remapping class ids, so an L-R flip puts left ribs in
# right-side positions with left labels, and an S-I flip inverts rib order.
#
# Why these eight, given the baseline diagnosis (train dice 0.988 vs test 0.882,
# complete cages 0.916 vs incomplete 0.681):
#   truncrib/droprib  synthesise incomplete cages -- aimed straight at the gap
#   rotz/roty/scale   generic geometric regularisation
#   hushift/hujitter  scanner calibration and noise variation
#
#   TUNE_MODE=truncrib bash scripts/train.sh -d ribsegv2 -c seq-tune/aug ...

import os

_base_ = ["_tune_base.py"]

hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 250000

aug_mode = os.environ.get("TUNE_MODE", "")
assert aug_mode in (
    "",                       # no aug: identical to in_feat-whu1 (4 seeds already run)
    "droprib", "truncrib",    # field of view
    "rotz", "roty",           # rotation
    "scale",                  # other geometry
    "hushift", "hujitter",    # intensity
), "Unsupported TUNE_MODE: {}".format(aug_mode)

# The rib augmentations subset only the keys they are given, and anything left
# at full length is then indexed by GridSample with the shortened index -- no
# error, just silently mismatched rows. `voxel_index` has to travel with them.
fov_keys = ("coord", "intensity", "voxel_index")

# --- field of view: applied on physical coords, before NormalizeCoord, and
# while `segment` still distinguishes rib1..rib24 (they select whole ribs by id)
aug_fov = []
if aug_mode == "droprib":
    aug_fov = [dict(type="RandomDropRibPoint", keys=fov_keys, max_drop_depth=4,
                    begin_from='', allow_single=True, p=0.3)]
elif aug_mode == "truncrib":
    aug_fov = [dict(type="RandomTruncateRibPoint", keys=fov_keys, max_drop_depth=4,
                    begin_from='', pos='', p=0.3, is_axis=2)]

# --- intensity: on RAW HU, before the window turns it into a [0,1] feature.
# The ranges below are in HU, so applying them after windowing would swamp it.
aug_hu = []
if aug_mode == "hushift":
    aug_hu = [dict(type="CTIntensityVariation", intensity_shift_range=(-50, 50),
                   intensity_scale_range=None, gamma_range=None, p=0.5, key="intensity")]
elif aug_mode == "hujitter":
    aug_hu = [dict(type="CTDensityNoise", noise_std=20, p=0.5, key="intensity")]

# --- geometry: after NormalizeCoord (so the magnitudes are in normalised units)
# and before GridSample, so the grid is built on the augmented coordinates
aug_geo = []
if aug_mode in ("rotz", "roty"):
    aug_geo = [dict(type="RandomRotate",
                    angle=(-0.1, 0.1) if aug_mode == "rotz" else (-0.04, 0.04),
                    axis=aug_mode[-1], p=0.5)]
elif aug_mode == "scale":
    aug_geo = [dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True)]
# `jitter` is deliberately absent. PTv3 serialises, pools and sparsifies on
# `grid_coord` -- integer cell indices -- and with enable_rpe=False the only use
# of `coord` is averaging pooled positions, so sub-cell displacement is invisible
# to attention. What jitter before GridSample actually does is push points across
# cell boundaries, inflating the count 5% (sigma 0.04 cells) to 54% (sigma 0.43,
# the DGCNN reference value) and changing the sampling density rather than the
# geometry. That is a confound, not an augmentation. DGCNN had no grid, so it was
# a fair augmentation there. GridSample(mode="train") already re-picks a random
# point per cell each epoch, which is the sub-cell jitter this would have added.

feat_keys = ("whu0",)
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)


def build_pipeline(mode):
    """mode: "train" (augmented) | "val" | "test" (volume-wise)"""
    train = mode == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train:
        pipeline += aug_fov + aug_hu
    pipeline += [
        dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="whu0"),
        dict(type="ExpandDims", key_axes=[("whu0", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
    ]
    if train:
        pipeline += aug_geo
    if mode == "test":
        pipeline.append(dict(type="Copy", keys_dict={
            "segment": "origin_segment", "voxel_index": "origin_voxel_index"}))
    pipeline.append(dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=(mode == "test"),
    ))
    if mode != "test":
        pipeline.append(dict(type="LimitPoint", max_points=max_points))
    pipeline += [
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "inverse",
                  "origin_segment", "origin_voxel_index") if mode == "test"
                 else ("coord", "grid_coord", "segment"),
            feat_keys=feat_keys,
        ),
    ]
    return pipeline


model = dict(backbone=dict(in_channels=1))

data = dict(
    train=dict(transform=build_pipeline("train")),
    val=dict(transform=build_pipeline("val")),
    test=dict(transform=build_pipeline("test")),
)

del build_pipeline, os

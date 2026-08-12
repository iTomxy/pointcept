# Step 2b -- augmentation combinations.
#
# From the singles (delta on val dice_fg vs the no-aug baseline 0.9041, noise
# floor 0.40 pt):
#   scale +1.91 | roty +1.49 | truncrib +1.26 | rotz +0.60 | hushift +0.59
#   hujitter -0.03 (tie) | droprib -0.58 (hurts)
#
# So: build on the five positives, drop `hujitter` (no effect) and `droprib`
# (deletes whole rib pairs mid-cage, an anatomy that never occurs in the data).
#
#   top3     scale + roty + truncrib
#   all5     + rotz + hushift, applied in sequence
#   all5alt  as all5 but the two rotations become one RandomApply choice --
#            applying rotz and roty in sequence compounds them into a larger
#            combined rotation, which is a different (stronger) augmentation
#            than either single, not a combination of them
#
# `max_points` is 200000 here, not the 250000 the singles used, and that is
# deliberate rather than incidental. `find_batch_size` sizes against real
# (unaugmented) samples, whose largest is 223040 -- but `scale` up to 1.1
# inflates the occupied-cell count ~5%, so the batch that actually runs can
# exceed the batch that was measured. On the 32GB V100 that slack was free; on
# this 22GB L4 the measured batch already sits at 97% of the card, so an
# unmeasured 5% would OOM. Capping below the largest real volume makes
# `bs x max_points` an exact bound again, at the cost of clipping the largest
# ~6% of volumes. `aug-comb-scale` re-runs the best single under the same cap so
# the combination is compared against a like-for-like reference.

import os

_base_ = ["_tune_base.py"]

hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 200000

comb_mode = os.environ.get("TUNE_MODE", "top3")
assert comb_mode in ("scale", "top3", "all5", "all5alt"), \
    "Unsupported TUNE_MODE: {}".format(comb_mode)

fov_keys = ("coord", "intensity", "voxel_index")

# field of view -- physical coords, before NormalizeCoord, while `segment` still
# distinguishes rib1..rib24 (RandomTruncateRibPoint selects ribs by class id)
aug_fov = [] if comb_mode == "scale" else [
    dict(type="RandomTruncateRibPoint", keys=fov_keys, max_drop_depth=4,
         begin_from='', pos='', p=0.3, is_axis=2)
]

# intensity -- on RAW HU, before the window maps it to [0,1]
aug_hu = [] if comb_mode not in ("all5", "all5alt") else [
    dict(type="CTIntensityVariation", intensity_shift_range=(-50, 50),
         intensity_scale_range=None, gamma_range=None, p=0.5, key="intensity")
]

# geometry -- after NormalizeCoord, before GridSample
rotz = dict(type="RandomRotate", angle=(-0.1, 0.1), axis='z', p=0.5)
roty = dict(type="RandomRotate", angle=(-0.04, 0.04), axis='y', p=0.5)
if comb_mode == "scale":
    aug_geo = []
elif comb_mode == "top3":
    aug_geo = [roty]
elif comb_mode == "all5":
    aug_geo = [rotz, roty]
else:  # all5alt -- pick one rotation per sample rather than compounding both
    aug_geo = [dict(type="RandomApply", cfgs=[rotz, roty])]
aug_geo = aug_geo + [dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True)]

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

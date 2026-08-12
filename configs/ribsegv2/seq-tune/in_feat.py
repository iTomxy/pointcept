# Step 1 -- input features.
#
# The baseline feeds a single broad HU window (200-1500 HU) as the only channel.
# The DGCNN tuning on this task concluded `nhu+whu` was best, so the same four
# feature sets are tried here:
#
#   whu1     one broad window, 200-1500 HU        (current baseline)  1 ch
#   nhu      per-volume percentile z-score                            1 ch
#   whu3     three narrow windows, 300/450/600 HU, width 100          3 ch
#   nhu+whu3 both                                                     4 ch
#
# The narrow windows are the interesting part: 250-350 / 400-500 / 550-650 HU
# straddle the trabecular-to-cortical range, so they resolve bone density where
# the single 200-1500 window compresses all of it into the top of one channel.
#
# A second sub-step adds position to the feature vector on top of whichever set
# wins -- PointNeXt appends height, DGCNN appends full xyz. PTv3 already uses
# coord for serialisation and its positional encoding, so this asks whether
# repeating it as a feature helps rather than whether position is available:
#
#   +z       append normalised height                                +1 ch
#   +xyz     append normalised coord                                 +3 ch
#
# Mode is taken from the environment so one config file can drive the whole
# sweep without edits; the resolved values are recorded in each run's config.py.
#   TUNE_MODE=whu3 bash scripts/train.sh -d ribsegv2 -c seq-tune/in_feat ...

import os

_base_ = ["_tune_base.py"]

hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 250000
# narrow windows as (level, width) -> (level - width/2, level + width/2)
whu3_windows = [(250.0, 350.0), (400.0, 500.0), (550.0, 650.0)]

in_feat_mode = os.environ.get("TUNE_MODE", "whu1")
assert in_feat_mode in (
    "whu1", "nhu", "whu3", "nhu+whu3",      # sub-step 1a: which HU representation
    "best+z", "best+xyz",                   # sub-step 1b: append position, on the 1a winner
), "Unsupported TUNE_MODE: {}".format(in_feat_mode)

# whichever set wins 1a; edit this line before running 1b
BEST_1A = os.environ.get("TUNE_BEST_1A", "whu1")  # step 1a: all four indistinguishable, take the simplest
hu_mode = BEST_1A if in_feat_mode.startswith("best") else in_feat_mode

# --- feature-building transforms, applied to raw HU before anything derived ---
build_nhu = [dict(type="NormalizeIntensityCached", src_key="intensity", dest_key="nhu")] \
    if "nhu" in hu_mode else []
build_whu = []
if hu_mode == "whu1":
    build_whu = [dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="whu0")]
elif "whu3" in hu_mode:
    build_whu = [
        dict(type="WindowIntensity", window=w, src_key="intensity", dest_key="whu{}".format(i))
        for i, w in enumerate(whu3_windows)
    ]

hu_keys = tuple(["nhu"] if "nhu" in hu_mode else []) + tuple(
    "whu{}".format(i) for i in range(len(build_whu))
)
expand = [dict(type="ExpandDims", key_axes=[(k, 1) for k in hu_keys])]

# position as an extra feature. `coord` is already [N, 3] and normalised, so
# copy it rather than re-deriving; `Copy` runs before GridSample so the copy is
# subsampled along with everything else.
build_pos, pos_keys = [], ()
if in_feat_mode == "best+xyz":
    build_pos = [dict(type="Copy", keys_dict={"coord": "pos"})]
    pos_keys = ("pos",)
elif in_feat_mode == "best+z":
    build_pos = [dict(type="SelectAxis", key="coord", dest_key="pos", axes=(2,))]
    pos_keys = ("pos",)

feat_keys = hu_keys + pos_keys
in_channels = len(hu_keys) + {"best+xyz": 3, "best+z": 1}.get(in_feat_mode, 0)

# GridSample only subsamples keys listed in `index_valid_keys`; the derived
# feature channels have to be declared or they keep their pre-sample length.
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)


def build_pipeline(mode):
    """mode: "train" | "val" | "test" (volume-wise)"""
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
        *build_nhu,
        *build_whu,
        *expand,
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
        *build_pos,
    ]
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


model = dict(backbone=dict(in_channels=in_channels))

data = dict(
    train=dict(transform=build_pipeline("train")),
    val=dict(transform=build_pipeline("val")),
    test=dict(transform=build_pipeline("test")),
)

del build_pipeline, os

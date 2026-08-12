# Step 3 -- grid size x attention window, on the L4 where flash-attention works.
#
# Built on the step-2b winner `all5` (truncrib + hushift + rotz + roty + scale,
# applied in sequence), so the only thing changing here is how the model sees
# the cloud.
#
# `enc_patch_size` was cut from the published (1024,)*5 to (128,128,256,512,1024)
# on the V100 purely because flash-attention needs Ampere+, and the fallback
# materialises the attention matrix (memory linear in patch size, OOM at batch 1
# with the published values). The L4 is SM 8.9, so that constraint is gone --
# measured at 3mm/batch 4, flash with the FULL published windows uses 11.4 GiB
# against 17.8 GiB for the reduced windows without it.
#
# But `enable_flash=True` is not numerically neutral: PTv3 casts qkv to bfloat16
# on that path (point_transformer_v3m1_base.py:209). So flipping it changes
# attention precision AND allows a bigger window at once. Hence the control arm:
#
#   g3p128    3mm, reduced windows, FLASH  -> vs augc-all5 isolates bf16 alone
#   g3p1024   3mm, full windows,    FLASH  -> adds the window effect on top
#   g4p1024   4mm, full windows,    FLASH  -> then the grid effect
#
# The reference is the existing `augc-all5` run (3mm, reduced windows, no flash,
# fp32, max_points=200000) at val dice_fg 0.9336.
#
# Grid choices are constrained by two measurements over the train split:
#
#   grid  max_pts  stage4_max  mem at batch 4   verdict
#   2.5   342964   1042        21.2 GiB (96%)   out: OOM risk AND stage4 > 1024,
#                                               so the deepest stage stops being
#                                               a single global window
#   3.0   232770    636        14.4 GiB (65%)   ok
#   4.0   140835    301         9.6 GiB (43%)   ok
#
# max_points stays 200000 at 3mm to match `augc-all5` exactly (it clips the top
# ~3% of volumes, as that run did); at 4mm 150000 is above the observed maximum
# so it never binds.

import os

_base_ = ["_tune_base.py"]

hu_window = (200.0, 1500.0)
global_radius = 259.16

mode = os.environ.get("TUNE_MODE", "g3p1024")
assert mode in (
    "g3p128", "g3p1024", "g4p1024",   # flash (bf16 attention)
    "g4p128nf",                       # 4mm in fp32: the one-variable grid change
), "Unsupported TUNE_MODE: {}".format(mode)

# Results so far made the follow-up arm obvious. Against the fp32 reference
# (augc-all5, 3mm/reduced windows, 0.9336):
#   g3p128  -0.98  turning flash on at fixed windows -- so bf16 attention alone
#                  costs ~1 pt, matching why enable_amp is off in the base config
#   g3p1024 -0.02  8x larger shallow windows on top -- an exact tie, so the
#                  window cut the V100 forced was never costing anything
#   g4p1024 +0.74  coarser grid, still carrying the bf16 penalty
# So the 4mm gain is confounded with the bf16 loss. `g4p128nf` re-runs 4mm in
# fp32 with the reference's own windows, changing ONLY the grid.
flash = not mode.endswith("nf")
grid_size_mm = 4.0 if mode.startswith("g4") else 3.0
grid_size = grid_size_mm / global_radius
max_points = 150000 if mode.startswith("g4") else 200000
enc_patch_size = (1024,) * 5 if "p1024" in mode else (128, 128, 256, 512, 1024)

model = dict(backbone=dict(
    enable_flash=flash,
    enc_patch_size=list(enc_patch_size),
    dec_patch_size=list(enc_patch_size[:4]),
))

fov_keys = ("coord", "intensity", "voxel_index")
feat_keys = ("whu0",)
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)


def build_pipeline(m):
    """m: "train" (all5 augmentation) | "val" | "test" (volume-wise)"""
    train = m == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train:
        pipeline += [
            dict(type="RandomTruncateRibPoint", keys=fov_keys, max_drop_depth=4,
                 begin_from='', pos='', p=0.3, is_axis=2),
            dict(type="CTIntensityVariation", intensity_shift_range=(-50, 50),
                 intensity_scale_range=None, gamma_range=None, p=0.5, key="intensity"),
        ]
    pipeline += [
        dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="whu0"),
        dict(type="ExpandDims", key_axes=[("whu0", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
    ]
    if train:
        pipeline += [
            dict(type="RandomRotate", angle=(-0.1, 0.1), axis='z', p=0.5),
            dict(type="RandomRotate", angle=(-0.04, 0.04), axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True),
        ]
    if m == "test":
        pipeline.append(dict(type="Copy", keys_dict={
            "segment": "origin_segment", "voxel_index": "origin_voxel_index"}))
    pipeline.append(dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=(m == "test"),
    ))
    if m != "test":
        pipeline.append(dict(type="LimitPoint", max_points=max_points))
    pipeline += [
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "inverse",
                  "origin_segment", "origin_voxel_index") if m == "test"
                 else ("coord", "grid_coord", "segment"),
            feat_keys=feat_keys,
        ),
    ]
    return pipeline


data = dict(
    train=dict(transform=build_pipeline("train")),
    val=dict(transform=build_pipeline("val")),
    test=dict(transform=build_pipeline("test")),
)

del build_pipeline, os

# Step 4 -- optimisation knobs, on the standing best recipe.
#
# Everything about how the model *sees* the data is now fixed by steps 1-3:
#   3mm grid | enc_patch_size (128,128,256,512,1024) | enable_flash=False (fp32)
#   whu1 (single 200-1500 HU window) | all5 augmentation | batch 4 | 50 epochs
# so the pipeline below is `aug-comb.py` with TUNE_MODE=all5 inlined verbatim,
# and the only thing TUNE_MODE selects here is a regularisation knob.
#
# The baseline diagnosis was overfitting (train dice 0.988 vs test 0.882), which
# augmentation attacked from the data side for +2.96 pt. These three attack it
# from the optimisation side:
#
#   ls0.1 / ls0.2    label smoothing 0.1 / 0.2 (currently 0.0). Rib indexing
#                    confuses neighbours -- 99.0% of the model's mislabels are
#                    off-by-one, never cross-side -- so the targets it is being
#                    pushed to fit with full confidence are exactly the ones a
#                    human annotator is least certain of at the rib boundaries.
#                    Applied to the CE term only; LovaszLoss has no such knob
#                    and smoothing its soft-IoU surrogate is not defined.
#   wd5e-3 / wd5e-2  weight decay 10x / 100x (currently 5e-4). 5e-4 is the
#                    Pointcept ScanNet default, carried over untouched, and
#                    ScanNet has ~50x more training scenes than the 320 here.
#   dp0.1 / dp0.5    drop_path 0.1 / 0.5 (currently 0.3, also inherited). Both
#                    directions, since 0.3 is a guess rather than a measurement.
#
# Two arms are controls rather than tuning:
#
#   ref     the standing recipe unchanged, at the standing lr (0.005102, pinned
#           via FIXED_LR). The 0.9336 reference was measured on a different
#           node/GPU and the 0.40 pt noise floor on a third, so without this arm
#           a node offset would shift every comparison below invisibly.
#   lr2x    the same recipe, but at the lr the range test produced when it was
#           re-run here: 0.0115, i.e. 2.25x. Nothing else differs. This is not a
#           knob anyone would tune -- it measures a confound. Steps 2-3 let each
#           arm draw its own lr from a single range-test pass, and that pass is
#           unstable: two byte-identical configs gave 0.0051 and 0.0115 because
#           the loss curve's argmin lands on either side of a transient spike
#           near lr 0.013. Across steps 2-3 the drawn lr spanned 4x. This arm
#           says how much of a dice_fg gap that can account for, which decides
#           whether the step-2b and step-3 rankings need re-running.
#
#   TUNE_MODE=ls0.1 bash scripts/train.sh -d ribsegv2 -c seq-tune/opt ...

import os

_base_ = ["_tune_base.py"]

hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 200000

mode = os.environ.get("TUNE_MODE", "ref")
assert mode in (
    "ref",                # standing recipe at the standing lr -- node control
    "lr2x",               # identical to `ref`; only the lr override differs
    "ls0.1", "ls0.2",     # label smoothing on the CE term
    "wd5e-3", "wd5e-2",   # AdamW weight decay
    "dp0.1", "dp0.5",     # stochastic depth
), "Unsupported TUNE_MODE: {}".format(mode)

label_smoothing = float(mode[2:]) if mode.startswith("ls") else 0.0
weight_decay = float(mode[2:]) if mode.startswith("wd") else 5e-4
drop_path = float(mode[2:]) if mode.startswith("dp") else 0.3

model = dict(
    backbone=dict(in_channels=1, drop_path=drop_path),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1,
             label_smoothing=label_smoothing),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# lr is re-tuned per run by tools/lr_range_test.py and arrives as an `-o`
# override, so only weight_decay is set here. Note the range test runs against
# THIS config, i.e. with this arm's weight decay already in place.
optimizer = dict(type="AdamW", weight_decay=weight_decay)

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

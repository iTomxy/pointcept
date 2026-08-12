# The tuned recipe. NOT a tuning step -- this is the summary of everything
# steps 1-4 concluded, run at full length for the one and only test-split pass.
#
# Two differences from the tuning configs, both deliberate:
#   epoch 100   the tuning steps ran 50 to keep a sweep to hours instead of
#               days. On the baseline, val mIoU at epoch 50 was 0.5 pt below
#               its best (epoch 74) -- enough to matter for a reported number,
#               not enough to change a ranking.
#   split test  every decision above was made on val. The 157-volume test split
#               has been untouched since preprocessing, so this is a genuine
#               held-out measurement, and it happens ONCE.
#
# What each setting rests on (all deltas are val dice_fg at matched lr, against
# a 0.40 pt noise floor):
#
#   whu1, 1 channel      step 1: all four HU representations within 0.33 pt, and
#                        position as a feature gave nothing. Preprocessing
#                        already thresholds at HU > 200, so the bone/background
#                        decision is made before the model sees anything.
#   all5 augmentation    step 2/2b: +2.96 over no augmentation, and the winner
#                        over `scale` alone (+0.67), `all5alt` (+0.79) and
#                        `top3` (+1.07). The only lever that moved the number.
#   grid 3mm             step 3: coarsening to 4mm costs 0.56 pt at matched
#                        precision and windows.
#   enable_flash=False   step 3: bf16 attention costs 0.98 pt. Larger windows
#                        would claw back 0.52, but fp32 cannot have them (no
#                        flash means the attention matrix is materialised, so
#                        memory is linear in patch size and 8x OOMs), and 0.52
#                        does not cover 0.98 anyway.
#   enc_patch_size       the V100-era reduced windows, kept for the same reason.
#   ls 0.0, wd 5e-4,     step 4: every optimisation knob tested came back a tie.
#   drop_path 0.3        See the results table; if one of them had won it would
#                        be set here instead.
#   batch 4, lr 0.005102 pinned, not re-measured. Both carry measurement
#                        variance larger than the effects they would be tuning
#                        -- lr in particular has a cliff just above 0.010 that
#                        costs 2.2 pt. See tmp-ai-tune_hp.sh.
#
# Run it with tmp-ai-final.sh, which trains then runs the single test pass.

_base_ = ["../semseg-pt_v3m1_0_base.py"]

epoch = 100
eval_epoch = epoch

hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 200000

# Selection still happens on val (CheckpointSaver tracks best val mIoU); the
# test pass below is run separately against model_best by tmp-ai-final.sh.
data = dict(
    test=dict(
        split="test",
        add_trainval_incomplete=False,
    ),
)

test = dict(save_pred=False, save_cm=True)

# Pinned, matching every arm of steps 2b-4.
#
# `param_dicts` must be restated here. The base config gives the `block` group a
# LITERAL lr of 0.0006, and the group ratio is computed as that literal over
# `optimizer.lr`. Every tuning run inherited `_tune_base`'s lr of 0.005719 and
# then had 0.005102 applied as an `-o` override *after* the ratio was taken, so
# they all trained the block group at 0.005719 * 0.10491 -> 0.0005353. Writing
# lr=0.005102 directly into this config instead makes the ratio 0.0006/0.005102
# = 0.1176, and the block group -- which is most of the model -- silently trains
# 12% hot. Stating the absolute 0.0005353 keeps the ratio at 0.10491 whether the
# config is run bare or under FIXED_LR.
optimizer = dict(type="AdamW", lr=0.005102, weight_decay=5e-4)
param_dicts = [dict(keyword="block", lr=0.0005353)]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005102, 0.0005353],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

model = dict(
    backbone=dict(in_channels=1, drop_path=0.3),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

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


data["train"] = dict(transform=build_pipeline("train"))
data["val"] = dict(transform=build_pipeline("val"))
data["test"].update(transform=build_pipeline("test"))

del build_pipeline

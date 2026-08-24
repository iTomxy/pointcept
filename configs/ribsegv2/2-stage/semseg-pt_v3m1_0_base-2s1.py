# Stage 1 of the two-stage RibSegv2 pipeline: rib vs. everything else.
# See plans/20260824-2stage.md (A1, A2, A4, A8, A9).
#
# Normally driven by run-2stage.sh rather than run directly:
#
#     bash run-ribseg-ptv3-2stage.sh
#
# Overlay on the TUNED recipe, ../semseg-pt_v3m1_0_base.py, which stays the
# single source of the architecture, the schedule and the augmentation.
#
# NOT built on ../semseg-pt_v3m1_0_base-bin.py, which is superseded (A9). That
# file predates the sequential tuning and restates a pipeline that no longer
# matches the base: `strength` rather than `whu0`, max_points 250000 rather than
# 200000, lr 0.006 with no `param_dicts`, and a `RandomApply` over
# `RandomDropRibPoint` + `RandomTruncateRibPoint` at max_drop_depth=7, p=0.5 --
# an augmentation set the tuning REJECTED (`RandomDropRibPoint` measured -0.58).
# Training stage 1 on it would make its numbers incomparable with the tuned
# 1-stage baseline, which is the whole point of the comparison.
#
# The pipeline is restated rather than inherited because Pointcept's `_base_`
# merges dicts but REPLACES lists wholesale, and the base `del`s its own
# `build_pipeline`, so an overlay cannot call it. That is the repo convention
# (cf. seq-tune/), and it is also exactly how `-bin.py` went stale -- so the
# delta from the base below is kept to the documented minimum, and this file
# should be re-checked whenever the base recipe changes.
#
# What changes, and nothing else:
#   - 2 classes instead of 25, in both places the count is declared (A2);
#   - `BinarizeLabel` in every pipeline, at the one position that satisfies
#     both ordering constraints (A1, below);
#   - `data.test.split="all"` and `save_pred=True`, so every volume gets a
#     sieve for stage 2, not just the test split (A4);
#   - the checkpoint is selected on F2 rather than mIoU (A8).
# Input channels, grid, losses, optimiser and schedule are deliberately
# untouched, so stage-1 numbers stay comparable with the 1-stage baseline.

_base_ = ["../semseg-pt_v3m1_0_base.py"]

fg_classes = tuple(range(1, 24 + 1))

# --- these four must mirror the base; they are only restated because
# `build_pipeline` below needs them in this file's namespace.
hu_window = (200.0, 1500.0)
global_radius = 259.16  # data/ribsegv2/complete-radius.json
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 200000
feat_keys = ("whu0",)
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)

dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"


def build_pipeline(mode):
    """mode: "train" (augmented) | "val" | "test" (volume-wise, full resolution)

    Identical to the base config's builder except for the one `BinarizeLabel`
    marked below. Keep it that way.
    """
    train = mode == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train:
        pipeline += [
            dict(type="RandomTruncateRibPoint", keys=("coord", "intensity", "voxel_index"),
                 max_drop_depth=4, begin_from='', pos='', p=0.3, is_axis=2),
            dict(type="CTIntensityVariation", intensity_shift_range=(-50, 50),
                 intensity_scale_range=None, gamma_range=None, p=0.5, key="intensity"),
        ]
    # --- A1. The ONLY structural difference from the base pipeline, and its
    # position is load-bearing in both directions:
    #   - it must come AFTER `RandomTruncateRibPoint`, which selects whole ribs
    #     by class id and returns early when fewer than 2 rib pairs are present
    #     (transform.py, `n_p < 2`). Binarising first leaves one foreground
    #     class, so that augmentation silently stops firing -- no error, just a
    #     stage-1 run missing the +2.96 pt lever the tuning identified.
    #   - it must come BEFORE the `Copy` to `origin_segment` below, because
    #     `Ribsegv2VolumeTester` scores against `origin_segment`, not the
    #     transformed `segment`. After the copy, stage 1 would compare 2
    #     predicted classes against 25 target ones, and nothing would raise.
    # This position satisfies both at once.
    pipeline.append(dict(type="BinarizeLabel", coi=fg_classes))
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


# A2. Both declarations of the class count. `backbone.in_channels` stays 1.
model = dict(num_classes=1 + 1)

data = dict(
    num_classes=1 + 1,
    bg_class=0,
    names=("background", "rib"),
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        test_mode=False,
        transform=build_pipeline("train"),
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=False,
        transform=build_pipeline("val"),
    ),
    test=dict(
        type="Ribsegv2VolumeDataset",
        # A4. Every volume, so stage 2 has a sieve for train/val/test alike.
        # 649 volumes after IGNORE_VOLUMES; assert that count after step 4.
        split="all",
        data_root=data_root,
        add_trainval_incomplete=False,
        transform=build_pipeline("test"),
    ),
)

# A8. `hooks` is a LIST, so an overlay replaces it outright -- the whole list is
# restated with only `SemSegEvaluator` changed. Losing the unmonitored entries
# would break checkpointing and logging silently.
#
# Why F2: stage 2 only ever sees the foreground stage 1 kept, so a rib voxel
# stage 1 drops is gone for good, while a false positive is still correctable
# downstream. mIoU weighs the two equally, which is right for a one-stage model
# and wrong for a gate. Plain recall is maximised by calling everything
# foreground; F2 weighs recall 4x but still penalises over-prediction.
# Read the value with care -- see A8 in the plan: the F2 of an all-foreground
# predictor is 5p/(4p+1) in the foreground fraction, which on this dataset
# (p ~ 0.216) is 0.58, ABOVE a balanced model's 0.50. It ranks useful models
# correctly (P=.90/R=.95 scores 0.94) but is not an absolute quality score.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", select_metric="fbeta", beta=2.0),
    dict(type="CheckpointSaver", save_freq=None),
]

test = dict(
    # A3/A4. Stage 2 sieves on these; see preproc.recon_3d_bin_pred.
    save_pred=True,
    # `rib_metrics` auto-disables itself below 25 classes: rib-index shift has
    # nothing to shift between when there is one foreground class.
    save_cm=False,
)

del build_pipeline

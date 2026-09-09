"""Shared PTv3 pipeline for RibSegV2 + TotalSegmentator experiments.

This overlays the tuned RibSegV2 recipe.  The source cache and every model
setting stay fixed; only the target label space differs between the coarse and
fine child configs.
"""

_base_ = ["../semseg-pt_v3m1_0_base.py"]

# Register CombineBoneLabel only for these experiments.  The generic dataset
# import path should not load TotalSegmentator's 1,000-line class dictionary.
custom_imports = dict(
    imports=["pointcept.datasets.ribsegv2.plus_totalseg_pred"],
    allow_failed_imports=False,
)

num_bone_classes = 10
bone_class_names = (
    "background",
    "rib",
    "vertebrae",
    "scapula",
    "humerus",
    "clavicula",
    "sternum",
    "costal_cartilages",
    "skull",
    "hip",
)

# Already built by the sibling DGCNN experiment.  Each entry is the ordinary
# RibSegV2 point cache plus one point-aligned ``totalseg_pred`` array.
totalseg_cache_root = "data/ribsegv2/pt_preproc_totalseg"

# These are deliberately restated from the tuned recipe.  Config inheritance
# replaces transform lists wholesale, and CombineBoneLabel has to be inserted
# at an exact point in all three pipelines.
hu_window = (200.0, 1500.0)
global_radius = 259.16
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 200000
feat_keys = ("whu0",)

# ``totalseg_pred`` must be sampled by the same index as every other per-point
# field until CombineBoneLabel consumes it. Keeping it here preserves that
# invariant if another index-operator transform is inserted before combination;
# omitting it from ``fov_keys`` already breaks today's pipeline whenever
# RandomTruncateRibPoint fires. After combination, index operators simply skip
# the now-absent source key.
index_valid_keys = [
    "coord",
    "strength",
    "segment",
    "voxel_index",
    "totalseg_pred",
] + list(feat_keys)
fov_keys = ("coord", "intensity", "voxel_index", "totalseg_pred")


def build_pipeline(mode):
    """Build the tuned PTv3 pipeline with complemented bone targets."""
    train_mode = mode == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train_mode:
        pipeline += [
            # This must see the original RibSegV2 ids 1..24.  In particular,
            # coarse combination cannot run before it or the augmentation's
            # pair counting becomes a no-op.
            dict(
                type="RandomTruncateRibPoint",
                keys=fov_keys,
                max_drop_depth=4,
                begin_from="",
                pos="",
                p=0.3,
                is_axis=2,
            ),
            dict(
                type="CTIntensityVariation",
                intensity_shift_range=(-50, 50),
                intensity_scale_range=None,
                gamma_range=None,
                p=0.5,
                key="intensity",
            ),
        ]

    # ``granularity=None`` reads the coarse/fine selector placed in the sample
    # by the dataset.  This makes the two phases share this exact pipeline.
    pipeline += [
        dict(type="CombineBoneLabel", granularity=None),
        dict(
            type="WindowIntensity",
            window=hu_window,
            src_key="intensity",
            dest_key="whu0",
        ),
        dict(type="ExpandDims", key_axes=[("whu0", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
    ]
    if train_mode:
        pipeline += [
            dict(type="RandomRotate", angle=(-0.1, 0.1), axis="z", p=0.5),
            dict(type="RandomRotate", angle=(-0.04, 0.04), axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True),
        ]
    if mode == "test":
        # Copy the COMBINED label before grid sampling.  The volume tester then
        # scatters grid predictions back to this full-resolution target.
        pipeline.append(
            dict(
                type="Copy",
                keys_dict={
                    "segment": "origin_segment",
                    "voxel_index": "origin_voxel_index",
                },
            )
        )
    pipeline.append(
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=(mode == "test"),
        )
    )
    if mode != "test":
        pipeline.append(dict(type="LimitPoint", max_points=max_points))
    pipeline += [
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=(
                (
                    "coord",
                    "grid_coord",
                    "segment",
                    "inverse",
                    "origin_segment",
                    "origin_voxel_index",
                    "affine",
                )
                if mode == "test"
                else ("coord", "grid_coord", "segment")
            ),
            feat_keys=feat_keys,
        ),
    ]
    return pipeline


model = dict(num_classes=num_bone_classes)

data = dict(
    _delete_=True,
    num_classes=num_bone_classes,
    bg_class=0,
    ignore_index=-1,
    names=bone_class_names,
    train=dict(
        type="Ribsegv2Dataset",
        split="train",
        data_root="data/ribsegv2",
        cache_root=totalseg_cache_root,
        bone_label_granularity="coarse",
        test_mode=False,
        transform=build_pipeline("train"),
    ),
    val=dict(
        type="Ribsegv2Dataset",
        split="val",
        data_root="data/ribsegv2",
        cache_root=totalseg_cache_root,
        bone_label_granularity="coarse",
        test_mode=False,
        transform=build_pipeline("val"),
    ),
    test=dict(
        type="Ribsegv2VolumeDataset",
        split="test",
        data_root="data/ribsegv2",
        cache_root=totalseg_cache_root,
        bone_label_granularity="coarse",
        add_trainval_incomplete=False,
        transform=build_pipeline("test"),
    ),
)

test = dict(
    type="Ribsegv2VolumeTester",
    verbose=True,
    save_pred=False,
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=True,
    # The RibSegV2 shift/label-accuracy metrics hard-code exactly 25 classes.
    # Per-class overlap metrics remain valid for both complemented spaces.
    rib_metrics=False,
)

del build_pipeline

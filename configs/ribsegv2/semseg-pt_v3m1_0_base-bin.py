# iTom, 2025 dec 8
# Stage-1 rib vs. non-rib segmentation with PTv3. Its saved predictions become
# the `sieve_mask` that stage 2 (semseg-pt_v3m1_0_base.py) can restrict itself
# to, so `data.test.split` is "all": every volume needs a prediction, not just
# the test split.

_base_ = ["semseg-pt_v3m1_0_base.py"]

# Sub-sampling is inherited in spirit from the base config (grid-subsample the
# whole rib cage, never a random scatter and never a spatial crop) -- see the
# note there for why the old 2e-3 grid made every PTv3 encoder stage a no-op.
# The pipelines are restated rather than inherited only because `BinarizeLabel`
# and the rib augmentations have to be interleaved at specific points, and
# config list values are replaced wholesale rather than merged.

# model settings
model = dict(
    num_classes=1+1, # fg vs. bg
    backbone=dict(in_channels=1),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        # dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)


# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
hu_thres = 200
hu_window = (200.0, 1500.0)
fg_classes = tuple(range(1, 24+1))
# `ReadNpz` always loads data/ribsegv2/pt_preproc, which is built from the FULL
# scans, so the radius has to be the one measured on those. This used to read
# 254.79 (the foreground-cropped figure) while loading the uncropped cache,
# which normalised the same points differently from every other config.
global_radius = 259.16 # data/ribsegv2/complete-radius.json
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 250000


def build_pipeline(mode):
    """
    mode: str, one of "train" (augmented), "val", or "test" (volume-wise)
    """
    assert mode in ("train", "val", "test")
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        # `intensity` is an integer on disk purely to keep the cache small
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="strength"),
    ]
    if mode == "train":
        # these pick whole ribs out by class id, so they must run while the
        # labels still distinguish rib1..rib24 -- after BinarizeLabel there is
        # only one foreground class left, `n_p < 2`, and both become no-ops
        pipeline.append(dict(type="RandomApply", cfgs=[
            dict(type="RandomDropRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', allow_single=True, p=0.5),
            dict(type="RandomTruncateRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', pos='', p=0.5, is_axis=2),
        ]))
    pipeline += [
        dict(type="BinarizeLabel", coi=fg_classes),
        dict(type="ExpandDims", key_axes=[("strength", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
    ]
    if mode == "train":
        pipeline += [
            dict(type="RandomScale", scale=[0.8, 1.25]),
            dict(type="RandomShift", shift=[[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]]),
        ]
    if mode == "test":
        # keep the full-resolution label & voxel index outside `index_valid_keys`
        # so GridSample leaves them at full length for `inverse` to index into
        pipeline.append(
            dict(type="Copy", keys_dict={"segment": "origin_segment", "voxel_index": "origin_voxel_index"})
        )
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
            keys=("coord", "grid_coord", "segment", "inverse", "origin_segment", "origin_voxel_index")
                 if mode == "test" else ("coord", "grid_coord", "segment"),
            feat_keys=("strength",),
        ),
    ]
    return pipeline


data = dict(
    _delete_=True,
    num_classes=1+1,
    bg_class=0,
    ignore_index=-1,
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
        # every volume, so stage 2 has a sieve mask for train/val/test alike
        split="all",
        data_root=data_root,
        transform=build_pipeline("test"),
    ),
)


test = dict(
    save_pred=True, # stage 2 sieves on this; see preproc.recon_3d_bin_pred
    # the rib-index metrics need the 25-class set, and are auto-disabled here
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=False,
)

# keep the helper out of the parsed config dict
del build_pipeline


_base_ = ["semseg-pt_v3m1_0_base.py"]

# Stage-1 rib vs. non-rib segmentation with SpUNet.
#
# See semseg-spunet_v1m1_0_base.py for why the grid sub-sampling matters even
# more for a sparse CNN than for PTv3: at the old 2e-3 (0.52mm) grid with 15k
# randomly scattered points, only ~2% of points had a neighbour inside the
# 3x3x3 kernel footprint, so every convolution collapsed to a pointwise linear.
#
# The pipelines below are restated rather than inherited only because
# `BinarizeLabel` has to follow cache reconstruction/casting, before any
# label-dependent augmentation. Config list values are replaced wholesale
# rather than merged. They are otherwise identical to the PTv3 base -- keep
# them in sync.

# model settings
model = dict(
    _delete_=True,
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=1,
        num_classes=1+1,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)


optimizer = dict(_delete_=True, type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
# SpUNet has no "block"-keyword parameter group like PTv3, so one group only
param_dicts = None


# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
hu_thres = 200
hu_window = (200.0, 1500.0)
fg_classes = tuple(range(1, 24+1))
global_radius = 259.16 # data/ribsegv2/complete-radius.json
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
max_points = 250000


def build_pipeline(mode):
    """the PTv3 base pipeline plus BinarizeLabel
    mode: str, one of "train" (train/val) or "test"
    """
    assert mode in ("train", "test")
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        # `intensity` is an integer on disk purely to keep the cache small
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="BinarizeLabel", coi=fg_classes),
        dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="strength"),
        dict(type="ExpandDims", key_axes=[("strength", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),
        dict(type="CenterShift", apply_z=True),
    ]
    if mode == "test":
        # stash the full-resolution label & voxel index outside `index_valid_keys`
        # so GridSample leaves them at full length for `inverse` to index back into
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
    if mode == "train":
        pipeline.append(dict(type="LimitPoint", max_points=max_points))
    pipeline += [
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment") if mode == "train" else
                 ("coord", "grid_coord", "segment", "inverse", "origin_segment", "origin_voxel_index"),
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
        transform=build_pipeline("train"),
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
    save_pred=True, # stage-2 sieves on this, see preproc.recon_3d_bin_pred
    # rib-index metrics are meaningless with 2 classes
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=False,
)

# keep the helper out of the parsed config dict
del build_pipeline

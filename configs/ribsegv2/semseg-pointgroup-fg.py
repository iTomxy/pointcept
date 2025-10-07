_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
num_worker = 8
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True
enable_wandb = False # to avoid bug
enable_amp = False # https://github.com/Pointcept/Pointcept/issues/249#issuecomment-2109206794

class_names = [
    "background",
    "rib",
]
num_classes = 2
segment_ignore_index = (-1, 0)

# model settings
model = dict(
    type="PG-semseg",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=1, # channels for features (e.g. HU) other than coord
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_out_channels=96,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=10.0, # radius in clustering (DBScan)
    cluster_closed_points=100,
    cluster_propose_points=100,
    cluster_min_points=50,
)

# scheduler settings
epoch = 100
eval_epoch = epoch
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001)
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type='CTIntensityVariation',
                intensity_shift_range=(-30, 30),  # ±30 HU shift
                intensity_scale_range=(0.98, 1.02),  # ±2% scaling
                gamma_range=(0.95, 1.05),  # subtle gamma correction
                p=0.8
            ),
            dict(type='CTDensityNoise',
                noise_std=8,  # 8 HU standard deviation
                p=0.5
            ),
            dict(type='NormalizeIntensity', min_hu=-1000, max_hu=1000),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=15000, mode="random"),
            dict(type="SamplePoint", npoints=15000),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                ),
                # feat_keys=("color", "normal"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type='NormalizeIntensity', min_hu=-1000, max_hu=1000),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="SamplePoint", npoints=15000),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                ),
                # feat_keys=("color", "normal"),
                feat_keys=("strength",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type='NormalizeIntensity', min_hu=-1000, max_hu=1000),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="SamplePoint", npoints=15000),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                # feat_keys=("color", "normal"),
                feat_keys=("strength",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,  # TODO: design test mode for ins seg, e.g. TTA
    ),  # currently not available
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(
    #     type="InsSegEvaluator",
    #     segment_ignore_index=segment_ignore_index,
    #     instance_ignore_index=-1,
    # ),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Tester
test = dict(
    type="SemSegTester2"
)

_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False # to avoid bug
enable_amp = False # https://github.com/Pointcept/Pointcept/issues/249#issuecomment-2109206794


class_names = [
    "background",
    "rib",
]
num_classes = 2
segment_ignore_index = (-1, 0)
# dataset settings
dataset_type = "Ribsegv2DatasetFG"
data_root = "data/ribsegv2"
# bg_ratio = None
# bg_ratio_rel_fg = 1
npoints = 15000


# model settings
model = dict(
    type="DGCNN_clreg",
    npoints=npoints,
    in_channels=3,
    # (2 Oct 2025) with insseg integration
    semantic_num_classes=1+1,
    instance_ignore_index=-1,
    # voxel_size=0.02,
    segment_ignore_index=(-1, 0),
    cluster_thresh=10.0,
    cluster_closed_points=100,
    cluster_propose_points=100,
    cluster_min_points=50,
)


# scheduler settings
epoch = 100
eval_epoch = epoch
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.0001)
scheduler = dict(type="CosineAnnealingLR")


data = dict(
    num_classes=num_classes,
    ignore_index=0, # background
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
            # dict(
            #     type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            # ),
            dict(type="SamplePoint", npoints=npoints),
            dict(type="Copy", keys_dict={"instance": "origin_instance"}), # before `InstanceParser` which rearrange instance ids
            dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True, also_to=["matched_skeleton"]),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1], also_to=["matched_skeleton"]),
            dict(type="RandomShift", shift=((-20.2, 20.2), (-20.2, 20.2), (-20.2, 20.2)), also_to=["matched_skeleton"]),
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
            dict(type="CenterShift", apply_z=False, also_to=["matched_skeleton"]),
            # dict(type="SphereCrop", point_max=15000, mode="random"),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
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
        # bg_ratio=bg_ratio,
        # bg_ratio_rel_fg=bg_ratio_rel_fg,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type='NormalizeIntensity', min_hu=-1000, max_hu=1000),
            dict(type="SamplePoint", npoints=npoints), # before `origin_instance` copy, otherwise shape mismatch
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True, also_to=["matched_skeleton"]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False, also_to=["matched_skeleton"]),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
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
        # bg_ratio=bg_ratio,
        # bg_ratio_rel_fg=bg_ratio_rel_fg,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type='NormalizeIntensity', min_hu=-1000, max_hu=1000),
            dict(type="SamplePoint", npoints=npoints), # before `origin_instance` copy, otherwise shape mismatch
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True, also_to=["matched_skeleton"]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            # dict(type="SamplePoint", npoints=1000000),
            dict(type="CenterShift", apply_z=False, also_to=["matched_skeleton"]),
            # dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
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
        # bg_ratio=bg_ratio,
        # bg_ratio_rel_fg=bg_ratio_rel_fg,
    ),  # currently not available
)


hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SkeletonRegEvaluator", instance_ignore_index=-1),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
]


# Tester
# test = dict(type="SkeletonTester")
test = dict(
    type="InsSegTester2",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)

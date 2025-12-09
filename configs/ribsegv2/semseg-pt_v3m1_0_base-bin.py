# iTom, 2025 dec 8
# Adapted from:
# - ../matterport3d/semseg-pt-v3m1-0-base.py
# - ./semseg-dgcnn.py

_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False # to avoid bug
enable_amp = False # https://github.com/Pointcept/Pointcept/issues/249#issuecomment-2109206794


# misc custom setting
batch_size = 8  # bs: total bs in all gpus

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=1+1, # fg vs. bg
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=1, # hu
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
eval_epoch = epoch
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
npoints = 15000
segment_ignore_index = (-1, 0)

data = dict(
    num_classes=1+1,
    ignore_index=(-1, 0),
    names=(
        "background",
        "rib",
    ),
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
            dict(type="SamplePoint", npoints=npoints),
            # dict(type="Copy", keys_dict={"instance": "origin_instance"}), # before `InstanceParser` which rearrange instance ids
            # dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True),# also_to=["matched_skeleton"]),
            dict(type="RandomScale", scale=[0.9, 1.1]),# also_to=["matched_skeleton"]),
            dict(type="RandomShift", shift=((-20.2, 20.2), (-20.2, 20.2), (-20.2, 20.2))),# also_to=["matched_skeleton"]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),# also_to=["matched_skeleton"]),
            # dict(
            #     type="InstanceParser",
            #     segment_ignore_index=segment_ignore_index,
            #     instance_ignore_index=-1,
            # ),
            # dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    # "instance",
                    # "instance_centroid",
                    # "bbox",
                ),
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
            # dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True, also_to=["matched_skeleton"]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False, also_to=["matched_skeleton"]),
            # dict(
            #     type="InstanceParser",
            #     segment_ignore_index=segment_ignore_index,
            #     instance_ignore_index=-1,
            # ),
            # dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    # "instance",
                    "origin_coord",
                    "origin_segment",
                    # "origin_instance",
                    # "instance_centroid",
                    # "bbox",
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
                    # "instance": "origin_instance",
                },
            ),
            # dict(type="MatchRibSkeleton"),
            dict(type="CenterShift", apply_z=True),# also_to=["matched_skeleton"]),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            # dict(type="SamplePoint", npoints=1000000),
            dict(type="CenterShift", apply_z=False),# also_to=["matched_skeleton"]),
            # dict(type="NormalizeColor"),
            # dict(
            #     type="InstanceParser",
            #     segment_ignore_index=segment_ignore_index,
            #     instance_ignore_index=-1,
            # ),
            # dict(type="Copy", keys_dict={"matched_skeleton": "instance_centroid"}), # after `InstanceParser`
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    # "instance",
                    "origin_coord",
                    "origin_segment",
                    # "origin_instance",
                    # "instance_centroid",
                    # "bbox",
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


train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester2")


_base_ = ["semseg-pt_v3m1_0_base-bin.py"]

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=1+24,
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

# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
npoints = 15000
hu_thres = 200
preproc = True # use preprocessed data (crop to foreground region) or not
if preproc:
    # data/ribsegv2/complete-radius-preproc.json
    global_radius = 254.79
else:
    # data/ribsegv2/complete-radius.json
    global_radius = 259.16 # in mm, 99.5% percentage of complete volume in physical coordinate


data = dict(
    num_classes=1+24,
    bg_class=0,
    ignore_index=-1,
    names=("background",) + tuple("rib{}".format(i) for i in range(1, 24+1)),
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        test_mode=False,
        transform=[
            dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity"),
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            dict(type="ToPhysicalCoord"),
            # dict(type='CTIntensityVariation',
            #     intensity_shift_range=(-30, 30),  # ±30 HU shift
            #     intensity_scale_range=(0.98, 1.02),  # ±2% scaling
            #     gamma_range=(0.95, 1.05),  # subtle gamma correction
            #     p=0.8
            # ),
            # dict(type='CTDensityNoise',
            #     noise_std=8,  # 8 HU standard deviation
            #     p=0.5
            # ),
            dict(type="RandomApply", cfgs=[ # after ToPhysicalCoord
                dict(type="RandomDropRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', allow_single=True, p=0.5),
                dict(type="RandomTruncateRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', pos='', p=0.5, is_axis=2),
            ]),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before SamplePoint
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.25]),
            dict(type="RandomShift",  shift=[[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SamplePoint", npoints=npoints, keys=["coord", "grid_coord", "segment", "strength"]), # after GridSample
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            # dict(type="Transpose", keys=["coord", "grid_coord", "strength"], axes=[1, 0]), # [npt, 3] -> [3, npt]
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=False,
        transform=[
            dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity"),
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            dict(type="ToPhysicalCoord"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before SamplePoint
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SamplePoint", npoints=npoints, keys=["coord", "grid_coord", "segment", "strength"]), # after GridSample
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            # dict(type="Transpose", keys=["coord", "grid_coord", "strength"], axes=[1, 0]), # [npt, 3] -> [3, npt]
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
    ),
    test=dict(
        type="Ribsegv2VolumeLoader",
        split="test",
        dataset_cls="Ribsegv2Volume",
        npoints=npoints,
        data_root=data_root,
        drop_last_thres=npoints // 4,
        add_trainval_incomplete=True,
        preproc_transform=[ # data reading & preprocessing
            dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity"),
            dict(type="Reorient", keys=("intensity", "segment"), new_ornt="LPS"), # keep this at test cuz it affects voxel_index
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"),
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")),
            dict(type="ToPhysicalCoord"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius),
        ],
        transform=[
            # at test, SamplePoint before GridSample cuz specified sample_idx & batch_size=1
            dict(type="SamplePoint", npoints=npoints, keys=["coord", "segment", "strength", "voxel_index"]),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            # dict(type="Transpose", keys=["coord", "grid_coord", "strength"], axes=[1, 0]), # [npt, 3] -> [3, npt]
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "voxel_index"),
                feat_keys=("strength",),
            ),
        ],
    ),
)


test = dict(type="SemSegVolumeTester", save_pred=True)

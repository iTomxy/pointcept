
_base_ = ["semseg-pt_v3m1_0_base.py"]

# model settings
model = dict(
    _delete_=True,
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=1,
        num_classes=1+24,
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


# dataset settings
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
npoints = 15000
hu_thres = 200
grid_size = 2e-3 # small enough so that the point cloud resolution won't change much
patch_size = (128, 128, 192) # 14 mar 2026, claude suggests
patch_stride = (None, None, None)
preproc = True # use preprocessed data (crop to foreground region) or not
if preproc:
    # data/ribsegv2/complete-radius-preproc.json
    global_radius = 254.79
else:
    # data/ribsegv2/complete-radius.json
    global_radius = 259.16 # in mm, 99.5% percentage of complete volume in physical coordinate


data = dict(
    _delete_=True,
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
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            dict(type="ToPhysicalCoord"),
            # dict(type="RandomApply", cfgs=[ # after ToPhysicalCoord
            #     dict(type="RandomDropRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', allow_single=True, p=0.5),
            #     dict(type="RandomTruncateRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', pos='', p=0.5, is_axis=2),
            # ]),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before SamplePoint
            dict(type="RandomPatchPoint", patch_size=patch_size, keys=("coord", "segment", "strength")),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.25]),
            dict(type="RandomShift",  shift=[[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
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
        type="Ribsegv2VolumeLoader",
        split="val",
        dataset_cls="Ribsegv2VolumePatch",
        npoints=npoints,
        data_root=data_root,
        patch_size=patch_size,
        stride=patch_stride,
        drop_last_thres=npoints // 4,
        preproc_transform=[
            dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity"),
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            dict(type="ToPhysicalCoord"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before SamplePoint
        ],
        transform=[
            dict(type="SamplePoint", npoints=npoints, keys=["coord", "segment", "strength", "voxel_index"]),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "voxel_index"),
                feat_keys=("strength",),
            ),
        ],
    ),
    test=dict(
        type="Ribsegv2VolumeLoader",
        split="test",
        dataset_cls="Ribsegv2VolumePatch",
        npoints=npoints,
        data_root=data_root,
        drop_last_thres=npoints // 4,
        patch_size=patch_size,
        stride=patch_size,
        add_trainval_incomplete=True,
        preproc_transform=[ # data reading & preprocessing
            dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity"),
            dict(type="Reorient", keys=("intensity", "segment"), new_ornt="LPS"), # keep this at test cuz it affects voxel_index
            dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"),
            dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")),
            dict(type="ToPhysicalCoord"),
            # dict(type="DropRibPoint", keys=("coord", "strength"), drop_depth=1, begin_from='i', single=''), # controlled test
            # dict(type="TruncateRibPoint", keys=("coord", "strength"), drop_depth=6, begin_from='i', pos='m', is_axis=2), # controlled test
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius),
        ],
        transform=[
            # at test, SamplePoint before GridSample cuz specified sample_idx & batch_size=1
            dict(type="SamplePoint", npoints=npoints, keys=["coord", "segment", "strength", "voxel_index"]),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
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


hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegVolumeEvaluator"), # suit volume-based evaluation in multi-gpu setting
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(_delete_=True, type="TrainerValLoader")
test = dict(_delete_=True, type="SemSegVolumeTesterOverlap1Gpu", save_pred=True)

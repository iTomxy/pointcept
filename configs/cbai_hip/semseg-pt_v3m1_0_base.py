# PTv3 on CBAI-hip -- first recipe, adapted from the tuned RibSegV2 config.
#
# Differences from configs/ribsegv2/semseg-pt_v3m1_0_base.py (the tuned rib
# recipe this starts from):
#   - 5 classes (bg + left/right femur, left/right hip) instead of 25.
#   - data_root data/cbai_hip; volumes come from the npz cache written by
#     `python -m pointcept.datasets.cbai_hip.preproc --preproc`; splits come
#     from data/cbai_hip/split.json (see make_split).
#   - global_radius = 1790.87 mm: 99.5th percentile of the per-volume p99.5
#     physical radius over all 92 volumes (data/cbai_hip/pt_preproc), the same
#     measurement ribsegv2's complete-radius.json records. These CBCT volumes
#     keep the DICOM corner origin, so the radii are large.
#   - Augmentations: the rib recipe's RandomTruncateRibPoint (a z-truncation
#     modelling a partial rib-cage FOV) is dropped -- it is rib anatomy. The
#     generic set is kept: z/y rotations, anisotropic scale, CTIntensityVariation
#     and the fixed HU window. No flips: left/right are distinct classes.
#   - Test uses the generic volume-wise tester (Ribsegv2VolumeTester, now
#     string-vid-safe; rib-index metrics auto-off for 5 classes).

# ---------------------------------------------------------------- runtime ----
weight = None  # path to model weight
resume = False
evaluate = True
test_only = False

seed = None  # train process will init a random seed and record
save_path = "exp/cbai_hip"
num_worker = 16  # total worker in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
clip_grad = None

sync_bn = True  # multi-GPU; per-GPU batch is 1, so BN stats must be synced
enable_amp = False
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False  # to avoid bug
wandb_project = "pointcept"
wandb_key = None
mix_prob = 0

train = dict(type="DefaultTrainer")
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

# batch size is machine-dependent; `run_cbai_hip.sh` measures it with
# tools/find_batch_size.py and overrides this at launch time.
batch_size = 4  # total across gpus
epoch = 100
eval_epoch = epoch


# ------------------------------------------------------------------ model ----
model = dict(
    type="DefaultSegmentorV2",
    num_classes=1 + 4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=1,  # a single windowed HU channel
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        # Without flash attention the fallback materialises the attention
        # matrix, and memory is linear in patch size; the rib recipe shrank the
        # shallow stages and kept 1024 at the deepest stage, where the cloud is
        # a few hundred points and one patch spans the whole volume.
        enc_patch_size=(128, 128, 256, 512, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 256, 512),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,  # flash path casts qkv to bf16, cost ~1 pt on ribs
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


# -------------------------------------------------------------- scheduler ----
# `param_dicts` gives the `block` group a LITERAL lr, and OneCycleLR's max_lr
# list must carry the same ratio (0.0005353/0.005102 = 0.1049). Change one and
# you must change the other. lr itself is re-measured by run_cbai_hip.sh's
# lr_range_test and overridden at launch.
optimizer = dict(type="AdamW", lr=0.005102, weight_decay=5e-4)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005102, 0.0005353],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0005353)]


# ---------------------------------------------------------------- dataset ----
dataset_type = "CbaiHipDataset"
data_root = "data/cbai_hip"
hu_thres = 200  # foreground threshold applied at preprocessing
# Fixed HU window -> feature in [0, 1], identical for every volume.
hu_window = (200.0, 1500.0)

# 99.5th percentile of the per-volume p99.5 physical radius over all 92 volumes.
global_radius = 1790.87

# Grid sub-sampling of the WHOLE cloud; 3mm as in the rib recipe (see its
# config for the pooling argument). grid_size is normalised by global_radius.
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius
# Cap after GridSample so `batch_size x max_points` stays an exact memory bound.
max_points = 200000

feat_keys = ("whu0",)
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)


def build_pipeline(mode):
    """mode: "train" (augmented) | "val" | "test" (volume-wise, full resolution)"""
    train = mode == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        # `intensity` is float32 on disk (cbai_hip HU is fractional)
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train:
        # HU-space noise acts on the RAW field, before the window maps it to
        # [0, 1]. The rib recipe's z-truncation augmentation is deliberately
        # not carried over -- it models a partial rib-cage field of view.
        pipeline += [
            dict(type="CTIntensityVariation", intensity_shift_range=(-50, 50),
                 intensity_scale_range=None, gamma_range=None, p=0.5, key="intensity"),
        ]
    pipeline += [
        dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="whu0"),
        dict(type="ExpandDims", key_axes=[("whu0", 1)]),
        dict(type="NormalizeCoord", radius=global_radius),  # before GridSample
        dict(type="CenterShift", apply_z=True),
    ]
    if train:
        # Geometry, in normalised units, before GridSample so the grid is built
        # on the augmented coordinates. The two rotations are applied in
        # SEQUENCE (compounding them is worth 0.79 pt over picking one, on ribs).
        pipeline += [
            dict(type="RandomRotate", angle=(-0.1, 0.1), axis='z', p=0.5),
            dict(type="RandomRotate", angle=(-0.04, 0.04), axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True),
        ]
    if mode == "test":
        # stash full-resolution label & voxel index; `origin_*` is outside
        # `index_valid_keys`, so GridSample leaves them at full length
        pipeline.append(dict(type="Copy", keys_dict={
            "segment": "origin_segment", "voxel_index": "origin_voxel_index"}))
    pipeline.append(dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        # maps every original point to its grid representative, so the
        # grid-level prediction scatters back to full resolution
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


data = dict(
    num_classes=1 + 4,
    bg_class=0,
    ignore_index=-1,
    names=("background", "left femur", "right femur", "left hip", "right hip"),
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
        # one item = one whole volume; a single forward pass covers it, and
        # `inverse` scatters the grid-level prediction back to full resolution.
        type="CbaiHipVolumeDataset",
        split="test",
        data_root=data_root,
        transform=build_pipeline("test"),
    ),
)


test = dict(
    type="Ribsegv2VolumeTester",  # generic volume-wise tester; string-vid-safe
    verbose=True,
    save_pred=False,
    # distance metrics (hd/assd/hd95/asd) need a reconstructed 3D grid, not a
    # point list, and are far too slow at ~1e6 points/volume -- overlap only.
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=True,  # confusion matrix: shows where the residual confusion sits
)

del build_pipeline

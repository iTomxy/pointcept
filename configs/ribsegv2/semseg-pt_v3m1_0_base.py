
_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False # to avoid bug
enable_amp = False # https://github.com/Pointcept/Pointcept/issues/249#issuecomment-2109206794
# Measured, not guessed -- `python tools/find_batch_size.py`, which
# runs a real fwd+bwd against the worst batch this pipeline can build
# (batch_size x max_points points, the bound `LimitPoint` enforces).
# On 2x Tesla V100-32GB (saturn14), budget 28.6 GiB at mem_frac=0.9:
#   bs/gpu=1 -> 9.9 GiB reserved (35% of budget)
#   bs/gpu=2 -> 19.2 GiB         (67%)   <- chosen
#   bs/gpu=3 -> 29.3 GiB        (102%)   over
# Needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in run_ribseg.sh)
# to keep fragmentation out of that margin. Re-measure if the GPU changes; the
# figures are dominated by the non-flash attention path, see enc_patch_size.
batch_size = 4  # bs: total bs in all gpus (2 per GPU x 2 GPUs)
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100
eval_epoch = epoch


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
        # Attention-window size per stage. The published (1024,)*5 assumes
        # flash-attention; without it the fallback materialises the attention
        # matrix and OOMs at batch size 1 even on a 32GB card. Memory here is
        # linear in patch size, so shrink the shallow stages (huge N, only
        # local shape needed) and keep 1024 at the deepest stage, where the
        # cloud is ~409 points and one patch therefore spans the whole rib
        # cage -- the global context that rib indexing actually depends on.
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
        enable_flash=False, # flash-attn needs Ampere+; V100 is SM70
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
optimizer = dict(type="AdamW", lr=0.006, weight_decay=5e-4)
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
hu_thres = 200
hu_window = (200.0, 1500.0) # fixed HU window -> feature in [0, 1], identical for every volume
# single_stage = True # use sieve_mask or not
preproc = False # use preprocessed data (crop to foreground region) or not
if preproc:
    # data/ribsegv2/complete-radius-preproc.json
    global_radius = 254.79
else:
    # data/ribsegv2/complete-radius.json
    global_radius = 259.16 # in mm, 99.5% percentage of complete volume in physical coordinate

# Sub-sampling: grid-subsample the WHOLE rib cage rather than randomly scattering
# a fixed number of points over it.
#   - `grid_size` is also what PTv3 pools on (SerializedPooling shifts the code
#     built from `grid_coord`), so a grid far below the voxel pitch makes every
#     encoder stage a no-op: 15000 -> 14943 -> 14556 -> 12532 -> 7201, and the
#     deepest attention window then only spans ~100mm.
#   - At 3mm the stages average 139k -> 35k -> 9.4k -> 2.2k -> 409 over the
#     train split (reduction 4.0/3.8/4.3/5.2x), so the last stage holds fewer
#     points than enc_patch_size (1024) for every volume and attends globally.
#     Assigning a rib its index 1..24 is an ordinal task that needs exactly that.
#   - Measured over the training set this costs ~0.21x the FLOPs and ~0.61x the
#     activation memory of the old 15k-random setup, at ~10x the points.
# Never crop a spatial patch here: a 128mm patch makes the rib index genuinely
# unidentifiable from the input, i.e. it turns the labels into noise.
# Re-measure any of this with:
#   python -m pointcept.datasets.ribsegv2.preproc --grid-stats --grid-size-mm 3.0
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius # config coords are normalised by `global_radius`
# Outlier guard only, above the observed max of 233k (train split: mean 139k,
# std 37k, p99 224k), so it never binds and never biases the density. It still
# bounds the worst-case batch for `find_batch_size`; lowering it below ~233k to
# buy a bigger batch does not pay here, since batch 3/GPU does not fit either
# way, and clipping would bias the point density for nothing.
max_points = 250000


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
            # dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity") if single_stage else dict(
            #     type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4"), ("sieve_mask", "i1")), meta_key="intensity"),
            # dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            # dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            # dict(type="ToPhysicalCoord"),
            dict(type="ReadNpz", rename_keys={"label": "segment"}), # load preprocessed data
            dict(type="ToPhysicalCoord"),
            # `intensity` is an integer on disk purely to keep the cache small
            dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
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
            # dict(type="RandomApply", cfgs=[ # after ToPhysicalCoord
            #     dict(type="RandomDropRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', allow_single=True, p=0.5),
            #     dict(type="RandomTruncateRibPoint", keys=("coord", "strength"), max_drop_depth=7, begin_from='', pos='', p=0.5, is_axis=2),
            # ]),
            # `intensity` is raw HU in [200, ~1600]; feed a fixed window instead of
            # the raw value or the per-volume percentile z-score cached as `norm_intensity`
            # (the latter clips at the 99.5th percentile, so all cortical bone saturates).
            dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="strength"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before GridSample
            dict(type="CenterShift", apply_z=True),
            # dict(type="RandomScale", scale=[0.8, 1.25]),
            # dict(type="RandomShift",  shift=[[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]]),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="LimitPoint", max_points=max_points), # after GridSample, so the drop stays uniform
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
            # dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity") if single_stage else dict(
            #     type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4"), ("sieve_mask", "i1")), meta_key="intensity"),
            # dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"), # won't affect `intensity`
            # dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")), # so here `intensity` is still usable
            # dict(type="ToPhysicalCoord"),
            dict(type="ReadNpz", rename_keys={"label": "segment"}), # load preprocessed data
            dict(type="ToPhysicalCoord"),
            # `intensity` is an integer on disk purely to keep the cache small
            dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
            dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="strength"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius), # before GridSample
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="LimitPoint", max_points=max_points), # after GridSample, so the drop stays uniform
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
        # one item = one whole volume; a single forward pass covers it, and
        # `inverse` scatters the grid-level prediction back to full resolution.
        type="Ribsegv2VolumeDataset",
        split="test",
        data_root=data_root,
        add_trainval_incomplete=True,
        transform=[
            # dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")), meta_key="intensity") if single_stage else dict(
            #     type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4"), ("sieve_mask", "i1")), meta_key="intensity"),
            # dict(type="Reorient", keys=("intensity", "segment"), new_ornt="LPS"), # keep this at test cuz it affects voxel_index
            # dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")),
            # dict(type="ToPhysicalCoord"),
            dict(type="ReadNpz", rename_keys={"label": "segment"}), # load preprocessed data
            dict(type="ToPhysicalCoord"),
            # `intensity` is an integer on disk purely to keep the cache small
            dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
            # dict(type="DropRibPoint", keys=("coord", "strength"), drop_depth=1, begin_from='i', single=''), # controlled test
            # dict(type="TruncateRibPoint", keys=("coord", "strength"), drop_depth=6, begin_from='i', pos='m', is_axis=2), # controlled test
            dict(type="WindowIntensity", window=hu_window, src_key="intensity", dest_key="strength"),
            dict(type="ExpandDims", key_axes=[("strength", 1)]),
            dict(type="NormalizeCoord", radius=global_radius),
            dict(type="CenterShift", apply_z=True),
            # stash full-resolution label & voxel index; `origin_*` is outside
            # `index_valid_keys`, so GridSample leaves them at full length
            dict(type="Copy", keys_dict={"segment": "origin_segment", "voxel_index": "origin_voxel_index"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True, # maps every original point to its grid representative
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "inverse", "origin_segment", "origin_voxel_index"),
                feat_keys=("strength",),
            ),
        ],
    ),
)


test = dict(
    type="Ribsegv2VolumeTester",
    save_pred=False,
    # distance metrics (hd/assd/hd95/asd) need a reconstructed 3D grid, not a
    # point list, and are far too slow at ~2M points/volume -- overlap only here.
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=True, # confusion matrix: an off-by-one band means missing global context
)


hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

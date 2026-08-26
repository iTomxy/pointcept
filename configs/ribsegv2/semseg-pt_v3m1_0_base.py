# PTv3 on RibSegV2 -- the tuned recipe.
#
# Standalone: no `_base_`. The generic runtime block below is what
# `configs/_base_/default_runtime.py` used to supply, minus the entries this
# config overrode anyway, so the whole recipe is readable in one file.
#
# TEST RESULT (157 held-out volumes, read once, `exp/ribsegv2/final/`):
#
#   metric         before tuning   tuned     delta
#   dice_fg        0.8819          0.9264    +4.44
#   iou_fg         0.8289          0.8796    +5.07
#   precision_fg   0.9010          0.9414    +4.04
#   recall_fg      0.8770          0.9214    +4.44
#   label_acc      0.9097          0.9611    +5.15
#
# Mislabelled rib points fell 1,895,622 -> 206,294 (-89%), still 95% off-by-one
# and 0% cross-side. Val for the same checkpoint was 0.9306, i.e. 0.42 pt above
# test -- about the noise floor, so selecting on val across ~35 runs did not
# overfit it.
#
# Provenance: plans/20260802-tune_ptv3.md carries a per-run table (config, log
# path, the batch/lr each run actually trained at, val metrics);
# plans/20260802-tune_ptv3-status.md explains the decision rule and the two
# measurement traps that cost the most time. What the sweep found:
#
#   augmentation      +2.96 pt   the only lever that moved the number
#   grid / precision   ~1 pt     and this config already sits on the best cell
#   input features     nothing   4 HU representations within 0.33 pt
#   optimisation       nothing   label smoothing, weight decay (100x sweep) and
#                                drop_path all inside the noise floor
#
# On 320 training volumes the binding constraint is data variety, not
# regularisation. Decisions were made on `Ribsegv2VolumeTester` over **val**
# (full-resolution, averaged per volume, foreground-only), NOT on the
# `Currently Best mIoU` in train.log -- the hook micro-averages over
# grid-subsampled points across batches, which is a different quantity and
# ranked `drop_path` the opposite way. **Noise floor: 0.40 pt.**


# ---------------------------------------------------------------- runtime ----
weight = None  # path to model weight
resume = False
evaluate = True
test_only = False

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
clip_grad = None

sync_bn = False
enable_amp = False  # https://github.com/Pointcept/Pointcept/issues/249#issuecomment-2109206794
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

# Pinned, not re-measured per run -- and that is the point. Both numbers carry
# measurement variance larger than the effects they would be tuning:
#   batch  memory-determined, ~5-10% run-to-run; near the ceiling that variance
#          silently halves the batch for one arm and not another. It happened
#          twice mid-sweep and cost 2.5 and 6.5 pt.
#   lr     `tools/lr_range_test.py` returned 0.0051 and 0.0115 for two
#          byte-identical configs, because a transient loss spike moves the
#          curve's argmin across it. There is a cliff just above 0.010 worth
#          -2.22 pt (measured, `exp/ribsegv2/opt-lr2x/`).
# Re-measure only when the model or the card changes enough that these may not
# even be stable; then pin the new values with FIXED_BS / FIXED_LR
# (tools/, tmp-ai-tune_hp.sh) for every arm of the comparison.
# Measured on 2x V100-32GB at mem_frac=0.9 against the worst batch this pipeline
# can build (batch_size x max_points points): 2/gpu -> 19.2 GiB (67% of budget),
# 3/gpu -> 29.3 GiB (over). Also fits 2x RTX A5500 24GB and 1x L4 22GB.
batch_size = 4  # total across gpus (2 per GPU x 2 GPUs)
epoch = 100
eval_epoch = epoch


# ------------------------------------------------------------------ model ----
model = dict(
    type="DefaultSegmentorV2",
    num_classes=1 + 24,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=1,  # a single windowed HU channel
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        # Attention-window size per stage. The published (1024,)*5 assumes
        # flash-attention; without it the fallback materialises the attention
        # matrix and OOMs at batch size 1 even on a 32GB card. Memory here is
        # linear in patch size, so shrink the shallow stages (huge N, only local
        # shape needed) and keep 1024 at the deepest stage, where the cloud is
        # ~409 points and one patch therefore spans the whole rib cage -- the
        # global context that rib indexing actually depends on.
        #
        # Measured, on an Ampere+ card where the full windows are reachable:
        # larger windows are worth +0.52 pt. That does NOT make them the better
        # choice here, because getting them requires enable_flash=True, which
        # costs 0.98 pt (below). The two are not independent knobs.
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
        # Tested at 0.1 and 0.5; both inside the noise floor. 0.5 first looked
        # like a +0.36 pt win with every metric agreeing, but a second seed put
        # its two runs 0.41 pt apart -- more spread than the effect. 0.3 also
        # reproduced itself to 0.02 pt across seeds, so it is the steadier pick.
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        # Keep False even on Ampere+. It is not just an availability flag: the
        # flash path casts qkv to bfloat16 (point_transformer_v3m1_base.py:209),
        # and that costs 0.98 pt at this grid size. Corroborates enable_amp=False
        # above. Turning it on buys ~36% memory headroom and nothing else.
        enable_flash=False,
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
        # label_smoothing was tested at 0.1 and 0.2 and does nothing -- 99% of
        # this model's mislabels are off-by-one, so softening the boundary
        # targets looked promising, but LovaszLoss carries half the loss weight
        # and takes no such argument, so only half the hard-target pressure can
        # be removed.
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)


# -------------------------------------------------------------- scheduler ----
# `param_dicts` gives the `block` group a LITERAL lr, and OneCycleLR's max_lr
# list must carry the same ratio (0.0005353/0.005102 = 0.1049). Change one and
# you must change the other -- writing a new `optimizer.lr` alone silently
# retunes the block group, which is most of the model.
# weight_decay was swept 5e-4 -> 5e-2 (100x) and moves the fourth decimal.
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
dataset_type = "Ribsegv2Dataset"
data_root = "data/ribsegv2"
hu_thres = 200  # foreground threshold applied at preprocessing
# Fixed HU window -> feature in [0, 1], identical for every volume. Preferred
# over raw HU and over the per-volume percentile z-score cached as
# `norm_intensity` (that one clips at the 99.5th percentile, so all cortical
# bone saturates). Three richer alternatives -- a 3-window stack, the z-score,
# and the z-score plus 3 windows -- all landed within 0.33 pt of this one,
# i.e. inside the noise. Preprocessing already thresholds at HU > 200, so the
# bone/background decision is made before the model sees anything.
hu_window = (200.0, 1500.0)
preproc = False  # use preprocessed data (cropped to the foreground region)
if preproc:
    global_radius = 254.79  # data/ribsegv2/complete-radius-preproc.json
else:
    # in mm, 99.5th percentile of complete volumes in physical coordinates
    global_radius = 259.16  # data/ribsegv2/complete-radius.json

# Sub-sampling: grid-subsample the WHOLE rib cage rather than randomly
# scattering a fixed number of points over it.
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
#
# 3mm is tuned, not assumed: 4mm costs 0.56 pt at matched precision and windows.
# 2.5mm is excluded on measurement rather than accuracy -- 96% of a 22GB card,
# AND its deepest stage reaches 1042 points > 1024, which loses the single
# global attention window that makes rib indexing work at all.
# Re-measure any of this with:
#   python -m pointcept.datasets.ribsegv2.preproc --grid-stats --grid-size-mm 3.0
grid_size_mm = 3.0
grid_size = grid_size_mm / global_radius  # config coords are normalised by `global_radius`
# Applied after GridSample, so dropping stays uniform and never biases density.
# 200000 sits just under the largest real volume (223k grid points): `RandomScale`
# up to 1.1 inflates the occupied-cell count ~5%, so without a cap the batch that
# actually runs can exceed the batch `find_batch_size` measured. Capping here
# makes `batch_size x max_points` an exact bound again, at the cost of clipping
# the top ~3% of volumes.
max_points = 200000

# Feature key. `whu0` (windowed HU) is what the tuned runs used; `index_valid_keys`
# must list it, or GridSample will not subset it alongside `coord`/`segment`.
# `strength` is kept in the list for the legacy CT2PointCloud path below.
feat_keys = ("whu0",)
index_valid_keys = ["coord", "strength", "segment", "voxel_index"] + list(feat_keys)


def build_pipeline(mode):
    """mode: "train" (augmented) | "val" | "test" (volume-wise, full resolution)

    One builder rather than three literal lists: train/val/test must share an
    identical preprocessing prefix, and a copy-paste triple is exactly where
    that silently stops being true.

    To read raw NIfTI instead of the preprocessed npz cache, replace `ReadNpz`
    with:
        dict(type="ReadNifti", keys=(("intensity", "f4"), ("segment", "i4")),
             meta_key="intensity"),
        dict(type="NormalizeIntensity", clip_percentile=(0.5, 99.5), dest_key="strength"),
        dict(type="CT2PointCloud", hu_thres=hu_thres, keys=("segment", "strength")),
    and keep `Reorient(new_ornt="LPS")` for the test split, since it affects
    `voxel_index`.
    """
    train = mode == "train"
    pipeline = [
        dict(type="ReadNpz", rename_keys={"label": "segment"}),
        dict(type="ToPhysicalCoord"),
        # `intensity` is an integer on disk purely to keep the cache small
        dict(type="TypeCast", key_type={"coord": "f4", "intensity": "f4"}),
        dict(type="Update", keys_dict={"index_valid_keys": index_valid_keys}),
    ]
    if train:
        # --- augmentation: the +2.96 pt, and the only thing that moved.
        # Order matters. These two act on physical coords and RAW HU, before
        # NormalizeCoord and before the window maps HU into [0, 1] -- the shift
        # range is in HU, so applying it after windowing would swamp the signal.
        # `RandomTruncateRibPoint` cuts along z like a real partial field of
        # view; its sibling `RandomDropRibPoint`, which deletes whole rib pairs
        # mid-cage, was tested and HURTS (-0.58) because that anatomy never
        # occurs. Flips are excluded by construction: ribs 1-12 are left and
        # 13-24 right, ordered top-down, and RandomFlip mirrors coordinates
        # without remapping class ids.
        pipeline += [
            dict(type="RandomTruncateRibPoint", keys=("coord", "intensity", "voxel_index"),
                 max_drop_depth=4, begin_from='', pos='', p=0.3, is_axis=2),
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
        # Geometry, in normalised units, and before GridSample so the grid is
        # built on the augmented coordinates. The two rotations are applied in
        # SEQUENCE, not as a RandomApply choose-one: compounding them into a
        # larger combined rotation is worth 0.79 pt over picking one.
        # No `jitter`: PTv3 serialises, pools and sparsifies on `grid_coord`, and
        # with enable_rpe=False `coord` is only averaged at pooling, so sub-cell
        # displacement is invisible to attention. All it does is push points
        # across cell boundaries, inflating the count 5-54% -- a density
        # confound, not an augmentation. GridSample(mode="train") already
        # re-picks a random point per cell each epoch, which is the sub-cell
        # jitter this would have added.
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
        # controlled ablations, e.g.
        # dict(type="DropRibPoint", keys=("coord", "intensity", "voxel_index"),
        #      drop_depth=1, begin_from='i', single=''),
        # dict(type="TruncateRibPoint", keys=("coord", "intensity", "voxel_index"),
        #      drop_depth=6, begin_from='i', pos='m', is_axis=2),
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
                  "origin_segment", "origin_voxel_index", "affine") if mode == "test"
                 else ("coord", "grid_coord", "segment"),
            feat_keys=feat_keys,
        ),
    ]
    return pipeline


data = dict(
    num_classes=1 + 24,
    bg_class=0,
    ignore_index=-1,
    names=("background",) + tuple("rib{}".format(i) for i in range(1, 24 + 1)),
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
        type="Ribsegv2VolumeDataset",
        split="test",
        data_root=data_root,
        # Must stay False. Enabled, it unions in every incomplete volume from
        # train and val, which puts training data into the evaluation set. The
        # 0.9264 above is with False, as is the 0.8819 it is compared against.
        add_trainval_incomplete=False,
        transform=build_pipeline("test"),
    ),
)


test = dict(
    type="Ribsegv2VolumeTester",
    verbose=True,
    save_pred=False,
    # distance metrics (hd/assd/hd95/asd) need a reconstructed 3D grid, not a
    # point list, and are far too slow at ~2M points/volume -- overlap only here.
    metrics=("dice", "iou", "precision", "recall", "specificity", "accuracy"),
    save_cm=True,  # confusion matrix: an off-by-one band means missing global context
)

del build_pipeline

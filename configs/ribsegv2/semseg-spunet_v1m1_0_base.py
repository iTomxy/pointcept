
_base_ = ["semseg-pt_v3m1_0_base.py"]

# NOTE on sub-sampling: the whole `data` block is inherited from the PTv3 config
# on purpose. SpUNet consumes exactly the same tensors (coord, grid_coord,
# segment, feat, offset), and it needs the grid sub-sampling even more sharply
# than PTv3 does.
#
# A sparse convolution only sees points that land inside its 3x3x3 kernel
# footprint, i.e. within one grid cell along each axis. Under the old setup --
# 15k points scattered at random over the volume on a 2e-3 (0.52mm) grid -- only
# 1.5-2.2% of points had ANY neighbour in that footprint, with a median
# neighbour count of 0. Every convolution therefore degenerated to a 1x1x1
# pointwise linear and the "UNet" was a per-point MLP with no spatial structure
# at all. Grid-sampling the whole rib cage at 3mm raises that to 98-99.5%, with
# a median of 16 of the 26 possible neighbours occupied.
#
# The 4 stride-2 stages then take the cell size 3 -> 6 -> 12 -> 24 -> 48mm, and
# the 6 blocks at the coarsest stage give a receptive field that spans the whole
# rib cage -- which is what assigning a rib its index 1..24 requires.
#
# Re-measure with:
#   python -m pointcept.datasets.ribsegv2.preproc --grid-stats --grid-size-mm 3.0

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
# the base gives PTv3's "block" parameters their own lr, hence its two-element
# max_lr; SpUNet has no counterpart, so drop back to a single parameter group
param_dicts = None


test = dict(save_pred=True)

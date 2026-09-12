_base_ = ["../semseg-pt_v3m1_0_base.py"]

model = dict(
    type="ContrastiveSegmentorV2",
    projection_hidden_channels=128,
    projection_out_channels=64,
    contrastive=dict(
        type="InstanceContrastiveLoss",
        loss_weight=0.1,
        temperature=0.1,
        anchors_per_class=4,
        positives_per_anchor=2,
        negatives_per_anchor=16,
        candidate_chunk_size=16384,
        include_background_negatives=False,
        bg_class=0,
        ignore_index=-1,
    ),
)

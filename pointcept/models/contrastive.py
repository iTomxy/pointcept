"""Training-only contrastive supervision for decoded segmentation features."""

import torch
import torch.nn as nn

from .builder import MODELS
from .default import DefaultSegmentorV2
from .losses import LOSSES


@MODELS.register_module()
class ContrastiveSegmentorV2(DefaultSegmentorV2):
    """DefaultSegmentorV2 with a training-only instance contrastive head."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        projection_hidden_channels=128,
        projection_out_channels=64,
        contrastive=None,
    ):
        if num_classes < 2:
            raise ValueError(
                "contrastive segmentation requires at least two classes")
        if min(backbone_out_channels, projection_hidden_channels,
               projection_out_channels) < 1:
            raise ValueError(
                "feature and projection dimensions must be positive")
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
            freeze_backbone=freeze_backbone,
        )
        self.backbone_out_channels = backbone_out_channels
        cfg = dict(contrastive or {})
        cfg.setdefault("type", "InstanceContrastiveLoss")
        cfg.setdefault("num_classes", num_classes)
        if cfg["num_classes"] != num_classes:
            raise ValueError(
                "contrastive num_classes must match the segmentation head")
        self.contrastive = LOSSES.build(cfg)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_channels, projection_hidden_channels),
            nn.LayerNorm(projection_hidden_channels),
            nn.GELU(),
            nn.Linear(projection_hidden_channels, projection_out_channels),
        )

    def forward(self, input_dict, return_point=False):
        # Evaluation must retain the canonical segmentor contract and avoid mining.
        if not self.training:
            return super().forward(input_dict, return_point=return_point)
        if input_dict["coord"].shape[0] == 0:
            raise ValueError(
                "contrastive segmentation requires a nonempty point cloud")
        if "offset" not in input_dict:
            raise KeyError("contrastive training requires input offset")
        out = super().forward(input_dict, return_point=True)
        point = out.pop("point")
        if not hasattr(point, "feat"):
            raise TypeError(
                "contrastive segmentation requires a decoded Point backbone")
        feat = point.feat
        if feat.shape != (input_dict["segment"].numel(),
                          self.backbone_out_channels):
            raise RuntimeError(
                "decoded features are not aligned with input labels")
        if point.coord.shape != input_dict["coord"].shape:
            raise RuntimeError(
                "decoded coordinates are not aligned with input coordinates")
        # PTv3's decoder restores the input order; the real-backbone preflight
        # verifies values too, without synchronizing GPU tensors every step.
        mined = self.contrastive.mine(input_dict["coord"],
                                      input_dict["segment"],
                                      input_dict["offset"])
        if mined.selected_indices.numel():
            selected = feat[mined.selected_indices]
        else:
            # Keep projection parameters in the graph for empty-pair batches.
            selected = feat[:1]
        projected = self.projection(selected)
        loss_con = self.contrastive(projected, mined)
        loss_seg = out["loss"]
        out["loss_seg"] = loss_seg.detach()
        out["loss_instance_contrastive"] = loss_con.detach()
        counts = {
            "contrastive_anchors": mined.anchor_indices.numel(),
            "contrastive_foreground_points": mined.num_foreground,
            "contrastive_selected_points": mined.selected_indices.numel(),
            "contrastive_clouds": mined.num_clouds,
        }
        out.update({
            key: feat.new_tensor(value)
            for key, value in counts.items()
        })
        out["loss"] = loss_seg + loss_con
        if return_point:
            out["point"] = point
        return out

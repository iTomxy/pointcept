"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p, bfs_cluster = None, None

from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.losses import build_criteria
from pointcept.models.builder import MODELS, build_model


@MODELS.register_module("PG-v1m1")
class PointGroup(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        self.ce_criteria = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    def forward(self, data_dict):
        coord = data_dict["coord"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"] # [bs], [npt, 2*npt, ..., bs*npt]

        feat = self.backbone(data_dict)
        bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)

        # compute loss
        if "segment" in data_dict.keys() and "instance" in data_dict.keys():
            segment = data_dict["segment"]
            instance = data_dict["instance"]
            seg_loss = self.ce_criteria(logit_pred, segment)

            mask = (instance != self.instance_ignore_index).float()
            bias_gt = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
            )

            loss = seg_loss + bias_l1_loss + bias_cosine_loss
            return_dict = dict(
                loss=loss,
                seg_loss=seg_loss,
                bias_l1_loss=bias_l1_loss,
                bias_cosine_loss=bias_cosine_loss,
            )
        else:
            return_dict = dict()

        if not self.training:
            center_pred = coord + bias_pred
            center_pred /= self.voxel_size
            seg_logits = logit_pred.detach() # for sem seg validation
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            # cluster
            mask = ( # points predicted to be not in `segment_ignore_index`
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            # mask.sum(): n_fg_pt, #{points not ignored}, i.e. foreground points
            if mask.sum() == 0:
                # all points are to be ignored
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                # offset2batch(offset): [bs*npt], [[0]*npt, ..., [bs-1]*npt]
                batch_ = offset2batch(offset)[mask]
                # batch2offset(batch_): [0, n1, n1+n2, ..., bs*npt]
                # nn.ConstantPad1d: prepend a `0` to the list
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                # idx: [n_all_neighbor_pt], neighbor point indices of each fg point
                # start_len: [n_fg_pt, 2], start index and length of each fg point's neighbors in `idx`
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                # proposals_idx: [n_fg_pt, 2], which point belongs to which cluster
                #   - proposals_idx[:, 0]: cluster id
                #   - proposals_idx[:, 1]: point index
                # proposals_offset: [n_cluster+1], cluster start and end index in proposals_idx
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                # map (FG) point index back to original (ALL) indices
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # get proposal
            # proposals_pred: int[n_cluster, npt], in {0, 1}, cluster assignment matrix
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            # instance_pred: int[n_cluster], in {0, ..., semantic_num_classes-1}, object class for each cluster
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            proposals_point_num = proposals_pred.sum(1) # [n_cluster], #points of each cluster
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module("PG-semseg")
class PointGroupSemseg(nn.Module):
    """(28 Sept 2025, iTom) debug the sem seg branch"""
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
    ):
        super().__init__()
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        # self.bias_head = nn.Sequential(
        #     nn.Linear(backbone_out_channels, backbone_out_channels),
        #     norm_fn(backbone_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(backbone_out_channels, 3),
        # )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        # self.ce_criteria = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)
        self.criteria = build_criteria([
            dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1),
            dict(type="DiceLoss", loss_weight=1.0, ignore_index=-1),
            dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        ])

    def forward(self, data_dict):
        feat = self.backbone(data_dict)
        # bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)
        segment = data_dict["segment"]
        seg_loss = self.criteria(logit_pred, segment)

        if self.training:
            return dict(loss=seg_loss)
        else:
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            return dict(seg_logits=logit_pred, loss=seg_loss)

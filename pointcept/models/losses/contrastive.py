"""Within-cloud pair mining and the instance contrastive objective."""

from dataclasses import dataclass
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES


@dataclass
class MinedPairs:
    # selected_indices index full decoder features; other indices address the compact embedding.
    selected_indices: torch.Tensor
    anchor_indices: torch.Tensor
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    anchor_cloud: torch.Tensor
    anchor_class: torch.Tensor
    num_clouds: int
    num_foreground: int


@LOSSES.register_module()
class InstanceContrastiveLoss(nn.Module):

    def __init__(self,
                 loss_weight=.1,
                 temperature=.1,
                 anchors_per_class=4,
                 positives_per_anchor=2,
                 negatives_per_anchor=16,
                 candidate_chunk_size=16384,
                 include_background_negatives=False,
                 bg_class=0,
                 ignore_index=-1,
                 num_classes=25):
        super().__init__()
        integer_values = (anchors_per_class, positives_per_anchor,
                          negatives_per_anchor, candidate_chunk_size, bg_class,
                          ignore_index, num_classes)
        if any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                for value in integer_values):
            raise ValueError(
                "selection budgets and class IDs must be integers")
        if (not isinstance(loss_weight, numbers.Real)
                or not isinstance(temperature, numbers.Real)
                or not math.isfinite(loss_weight)
                or not math.isfinite(temperature)):
            raise ValueError("loss_weight and temperature must be finite")
        if loss_weight < 0 or temperature <= 0 or anchors_per_class <= 0:
            raise ValueError("invalid contrastive loss configuration")
        if positives_per_anchor <= 0 or negatives_per_anchor <= 0 or candidate_chunk_size <= 0:
            raise ValueError("selection budgets must be positive")
        if num_classes <= 0 or not 0 <= bg_class < num_classes:
            raise ValueError(
                "num_classes and bg_class must define a valid background ID")
        self.loss_weight, self.temperature = loss_weight, temperature
        self.anchors_per_class, self.positives_per_anchor = anchors_per_class, positives_per_anchor
        self.negatives_per_anchor, self.candidate_chunk_size = negatives_per_anchor, candidate_chunk_size
        self.include_background_negatives, self.bg_class = include_background_negatives, bg_class
        self.ignore_index, self.num_classes = ignore_index, num_classes

    @staticmethod
    def _update_topk(values, indices, new_values, new_indices, k, largest):
        values = torch.cat((values, new_values), dim=1)
        indices = torch.cat(
            (indices, new_indices.expand(new_values.shape[0], -1)), dim=1)
        width = min(k, values.shape[1])
        values, order = torch.topk(values,
                                   width,
                                   dim=1,
                                   largest=largest,
                                   sorted=True)
        return values, indices.gather(1, order)

    def _stream_pairs(self, coord, labels, start, anchors, cls):
        """Exact top-k using only len(anchors) x chunk_size distances."""
        device, count = coord.device, labels.numel()
        pv = torch.empty((anchors.numel(), 0),
                         device=device,
                         dtype=torch.float32)
        nv = torch.empty_like(pv)
        pi = torch.empty((anchors.numel(), 0), device=device, dtype=torch.long)
        ni = torch.empty_like(pi)
        foreground = ((labels >= 1) & (labels < self.num_classes)
                      & (labels != self.bg_class) &
                      (labels != self.ignore_index))
        for begin in range(0, count, self.candidate_chunk_size):
            end = min(begin + self.candidate_chunk_size, count)
            local = torch.arange(begin, end, device=device)
            candidate = local + start
            distance = (coord[anchors].float()[:, None, :] -
                        coord[candidate].float()[None, :, :]).square().sum(-1)
            block_labels = labels[begin:end]
            positive = ((block_labels == cls) &
                        (block_labels != self.ignore_index))[None, :]
            positive = positive & (candidate[None, :] != anchors[:, None])
            negative = foreground[begin:end][None, :] & (block_labels[None, :]
                                                         != cls)
            if self.include_background_negatives:
                negative |= ((block_labels == self.bg_class) &
                             (block_labels != self.ignore_index))[None, :]
            pv, pi = self._update_topk(
                pv, pi, distance.masked_fill(~positive, float("-inf")),
                candidate, self.positives_per_anchor, True)
            nv, ni = self._update_topk(
                nv, ni, distance.masked_fill(~negative, float("inf")),
                candidate, self.negatives_per_anchor, False)

        def pad(indices, mask, width):
            padded_indices = torch.zeros((anchors.numel(), width),
                                         dtype=torch.long,
                                         device=device)
            padded_mask = torch.zeros((anchors.numel(), width),
                                      dtype=torch.bool,
                                      device=device)
            padded_indices[:, :indices.shape[1]] = indices
            padded_mask[:, :mask.shape[1]] = mask
            padded_indices.masked_fill_(~padded_mask, 0)
            return padded_indices, padded_mask

        pi, pm = pad(pi, torch.isfinite(pv), self.positives_per_anchor)
        ni, nm = pad(ni, torch.isfinite(nv), self.negatives_per_anchor)
        return pi, pm, ni, nm

    @torch.no_grad()
    def mine(self, coord, target, offset):
        """Mine compact pairs from Pointcept cumulative end offsets ([n1, n1+n2, ...])."""
        integral_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64,
                           torch.uint8)
        if coord.ndim != 2 or coord.shape[
                1] != 3 or target.ndim != 1 or coord.shape[0] != target.numel(
                ):
            raise ValueError("coord must be [N, 3] and target [N]")
        if coord.device != target.device or coord.device != offset.device:
            raise ValueError("coord, target, and offset must share a device")
        if target.dtype not in integral_dtypes or offset.dtype not in integral_dtypes:
            raise ValueError("target and offset must have integral dtypes")
        if not torch.isfinite(coord).all():
            raise ValueError("coord must be finite")
        if offset.ndim != 1 or offset.numel() == 0 or offset[-1].item(
        ) != coord.shape[0]:
            raise ValueError(
                "offset must be non-empty cumulative ends ending at N")
        if torch.any(offset[1:] < offset[:-1]) or torch.any(offset < 0):
            raise ValueError("invalid offset boundaries")
        device = coord.device
        anchors, positives, negatives, pms, nms, clouds, classes = [], [], [], [], [], [], []
        foreground_count, start = 0, 0
        for cloud, end in enumerate(offset.tolist()):
            labels = target[start:end]
            foreground = ((labels >= 1) & (labels < self.num_classes)
                          & (labels != self.bg_class) &
                          (labels != self.ignore_index))
            foreground_count += int(foreground.sum())
            for cls in torch.unique(labels[foreground]).tolist():
                class_local = torch.nonzero(labels == cls,
                                            as_tuple=False).flatten()
                if class_local.numel() < 2:
                    continue
                take = min(self.anchors_per_class, class_local.numel())
                sampled = class_local[torch.randperm(
                    class_local.numel(), device=device)[:take]] + start
                pi, pm, ni, nm = self._stream_pairs(coord, labels, start,
                                                    sampled, cls)
                keep = pm.any(1) & nm.any(1)
                if not keep.any():
                    continue
                anchors.append(sampled[keep])
                positives.append(pi[keep])
                negatives.append(ni[keep])
                pms.append(pm[keep])
                nms.append(nm[keep])
                clouds.append(
                    torch.full((int(keep.sum()), ),
                               cloud,
                               device=device,
                               dtype=torch.long))
                classes.append(
                    torch.full((int(keep.sum()), ),
                               cls,
                               device=device,
                               dtype=torch.long))
            start = end
        num_clouds = offset.numel()
        if not anchors:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return MinedPairs(
                empty, empty,
                torch.empty((0, self.positives_per_anchor),
                            dtype=torch.long,
                            device=device),
                torch.empty((0, self.negatives_per_anchor),
                            dtype=torch.long,
                            device=device),
                torch.empty((0, self.positives_per_anchor),
                            dtype=torch.bool,
                            device=device),
                torch.empty((0, self.negatives_per_anchor),
                            dtype=torch.bool,
                            device=device), empty, empty, num_clouds,
                foreground_count)
        anchors, positives, negatives = torch.cat(anchors), torch.cat(
            positives), torch.cat(negatives)
        pms, nms = torch.cat(pms), torch.cat(nms)
        selected = torch.unique(
            torch.cat((anchors, positives[pms], negatives[nms]))).sort().values

        def remap(indices, mask=None):
            result = torch.zeros_like(indices)
            if mask is None:
                return torch.searchsorted(selected, indices)
            result[mask] = torch.searchsorted(selected, indices[mask])
            return result

        return MinedPairs(selected, remap(anchors), remap(positives, pms),
                          remap(negatives, nms), pms, nms, torch.cat(clouds),
                          torch.cat(classes), num_clouds, foreground_count)

    def forward(self, embedding, mined):
        """FP32 per-positive InfoNCE, averaged anchors -> ribs -> clouds."""
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            zero = embedding.float().sum() * 0.0
            if mined.anchor_indices.numel() == 0:
                return zero
            z = F.normalize(embedding.float(), dim=-1)
            anchor = z[mined.anchor_indices]
            positive = (anchor[:, None] *
                        z[mined.positive_indices]).sum(-1) / self.temperature
            negative = (anchor[:, None] *
                        z[mined.negative_indices]).sum(-1) / self.temperature
            neg_lse = torch.logsumexp(negative.masked_fill(
                ~mined.negative_mask, float("-inf")),
                                      dim=1)
            each_positive = torch.logaddexp(positive, neg_lse[:,
                                                              None]) - positive
            per_anchor = (each_positive * mined.positive_mask
                          ).sum(1) / mined.positive_mask.sum(1).clamp_min(1)
            cloud_losses = []
            for cloud in range(mined.num_clouds):
                in_cloud = mined.anchor_cloud == cloud
                if not in_cloud.any():
                    cloud_losses.append(zero)
                    continue
                rib_losses = [
                    per_anchor[in_cloud & (mined.anchor_class == cls)].mean()
                    for cls in torch.unique(
                        mined.anchor_class[in_cloud]).tolist()
                ]
                cloud_losses.append(torch.stack(rib_losses).mean())
            return torch.stack(cloud_losses).mean() * self.loss_weight

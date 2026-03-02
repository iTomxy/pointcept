import math
import numpy as np
import torch


def clswise_cm_metrics_dist(tp, tn, fp, fn):
    """class-wise Confusion Matrix based metrics & counting
    Args:
        tp, tn, fp, fn: int[#classes], torch.LongTensor
    Returns:
        metrics: dict, {metric<str>: {
            'sum': torch.FloatTensor[#classes],
            'count': torch.FloatTensor[#classes]
        }}. Invalid entries are set to 0 so that they are directly summable.
    """
    tp = tp.to(torch.float64)
    tn = tn.to(torch.float64)
    fp = fp.to(torch.float64)
    fn = fn.to(torch.float64)
    zero = torch.zeros_like(tp, dtype=torch.float64)
    metrics = {}
    # iou
    denom = tp + fp + fn
    metrics["iou"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # dice = F1
    denom = 2 * tp + fp + fn
    metrics["dice"] = {
        "sum": torch.where(denom > 0, (2 * tp) / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # sensitivity = recall
    denom = tp + fn
    metrics["sensitivity"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # precision
    denom = tp + fp
    metrics["precision"] = {
        "sum": torch.where(denom > 0, tp / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # specificity
    denom = tn + fp
    metrics["specificity"] = {
        "sum": torch.where(denom > 0, tn / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    # accuracy
    denom = tp + tn + fp + fn
    metrics["accuracy"] = {
        "sum": torch.where(denom > 0, (tp + tn) / torch.clamp(denom, 1, None), zero),
        "count": (denom > 0).to(torch.float64)
    }
    return metrics


def confusion_matrix(pred, y, num_classes, ignore_index=-1):
    """Compute confusion matrix (TP, TN, FP, FN) for multi-class classification/segmentation,
    based on PyTorch tensors, can be used together with `torch.distributed.all_reduce` for distributed evaluation.
    Related metrics include: IoU, dice, accuracy
    Example:
    ```python
    background_cls = 0
    for x, y in loader:
        with torch.no_grad():
            logits = model(x) # [B, C, H, W]
        pred = logits.argmax(1) # [B, H, W]
        tp, tn, fp, fn = confusion_matrix(pred, y, num_classes, background_cls)
        if ddp_enabled:
            dist.all_reduce(tp), dist.all_reduce(tn), dist.all_reduce(fp), dist.all_reduce(fn)
    ```
    Input:
        pred: prediction mask of shape [N] or [N, L] or [N, H, W], int
        y: ground-truth segmentation mask, same shape as pred
        num_classes: int, #classes
        ignore_index: Union[int, List[int]] = -1, class ID/s to ignore in computation
    Output:
        tp: int[num_classes], True Positive
        tn: int[num_classes], True Negative
        fp: int[num_classes], False Positive
        fn: int[num_classes], False Negative
    """
    assert pred.dim() in [1, 2, 3]
    assert pred.shape == y.shape
    assert num_classes > int(pred.max().item()) and num_classes > int(y.max().item())

    pred = pred.view(-1)
    y = y.view(-1)
    ignore_index = torch.as_tensor(ignore_index, device=pred.device).flatten()
    valid_mask = ~ torch.isin(y, ignore_index)
    total_valid_pixels = valid_mask.sum().item()

    p_pred = torch.bincount(pred[valid_mask], minlength=num_classes)
    p_y = torch.bincount(y[valid_mask], minlength=num_classes)
    correct_mask = (pred == y) & valid_mask

    tp = torch.bincount(y[correct_mask], minlength=num_classes)
    fp = p_pred - tp
    fn = p_y - tp
    tn = total_valid_pixels - tp - fp - fn

    return tp.long(), tn.long(), fp.long(), fn.long()


def calc_cm_metrics(tp, tn, fp, fn, class_set, ignore_cls=[]):
    """calculate Confusion Matrix based metrics
    Input:
        tp, tn, fp, fn: int[#classes]
        class_set: int or List[int]
            - int: #classes, the class ID set will be {0, ..., n_classes - 1}
            - List[int]: ordered class ID set in the same order as tp, tn, fp & fn.
                Can be useful in part segmentation?
        ignore_cls: List[int] = [], classes to ignore at calculation, e.g. background
    Output:
        metrics: dict, {metric<str>: float}
    """
    ignore_cls = np.asarray([ignore_cls]).flatten()
    class_set = np.arange(class_set) if isinstance(class_set, int) else np.asarray(class_set)
    mask = ~ np.isin(class_set, ignore_cls)
    metrics = {}

    # class-wise: value of all classes are kept, including those to be ignored
    metrics["iou_class"] = tp / np.clip(tp + fp + fn, 1, None)
    metrics["dice_class"] = (2 * tp) / np.clip((2 * tp + fp + fn), 1, None)
    metrics["sens_class"] = tp / np.clip(tp + fn, 1, None) # recall = sensitivity
    metrics["prec_class"] = tp / np.clip(tp + fp, 1, None)
    metrics["spec_class"] = tn / np.clip(tn + fp, 1, None) # specificity = recall for negative class
    # metrics["f1_class"] = (2 * tp) / np.clip(2 * tp + fp + fn, 1, None)
    metrics["acc_class"] = (tp + tn) / np.clip(tp + tn + fp + fn, 1, None)

    # overall average: value of ignored classes are excluded
    metrics["iou"] = float(np.mean(metrics["iou_class"][mask]))
    metrics["dice"] = float(np.mean(metrics["dice_class"][mask]))
    metrics["precision"] = float(np.mean(metrics["prec_class"][mask]))
    metrics["sensitivity"] = float(np.mean(metrics["sens_class"][mask]))
    metrics["specificity"] = float(np.mean(metrics["spec_class"][mask]))
    # metrics["f1"] = float(np.mean(metrics["f1_class"][mask]))
    metrics["acc_macro"] = float(np.mean(metrics["acc_class"][mask]))
    metrics["acc_micro"] = float((tp + tn)[mask].sum() / max(1.0, (tp + tn + fp + fn)[mask].sum()))

    for k, v in metrics.items():
        if isinstance(v, float):
            # these metrics should be within [0, 1]
            assert -0.01 < v < 1.01, "Error value range of {}: {}".format(k, v)
        else:
            # class-wise list -> convert to list for json compatibility
            metrics[k] = v.tolist()

    return metrics


def error_rate(pred, y, bg_class):
    """calculate error rates (binary or multi-class)
    - False Negative Rate: p(pred \in BG | y \in FG)
    - False Positive Rate: p(pred \in FG | y \in BG)
    - Wrong Positive Rate: p(pred != y | pred \in FG, y \in FG)
        In multi-class, both prediction and ground-truth are fore-ground, but mismatched.
    - Correct Rate: p(pred = y)
    - Wrong Rate: p(pred != y)
    - FNR/FPR/WPR_conditional: p(. | pred != y)

    Args:
        pred: int numpy.ndarray, predicted class id
        y: same shape as pred, ground-truth class id
        bg_class: int or List[int], class id/s of background

    Returns:
        dict with fields:
            fnr: float in [0, 1], false-negative rate
            fpr: float in [0, 1], false-positive rate
            wpr: float in [0, 1], wrong-positive rate
    """
    bg_class = np.asarray([bg_class]).flatten()
    mask_bg = np.isin(y, bg_class)
    mask_bg_pred = np.isin(pred, bg_class)
    mask_fn = ~mask_bg & mask_bg_pred
    mask_fp = mask_bg & ~mask_bg_pred
    both_p = ~mask_bg & ~mask_bg_pred
    mask_wrong = pred != y
    n_p_fg = (~mask_bg_pred).sum()
    n_y_fg = (~mask_bg).sum()
    return {
        "fnr": float((mask_fn).sum() / (~mask_bg).sum()) if (~mask_bg).sum() > 0 else math.nan,
        "fpr": float((mask_fp).sum() / mask_bg.sum()) if mask_bg.sum() > 0 else math.nan,
        "wpr": float((mask_wrong & both_p).sum() / both_p.sum()) if both_p.sum() > 0 else math.nan,
        "cr": float((~mask_wrong).sum() / y.size),
        "wr": float(mask_wrong.sum() / y.size),
        "fnr_conditional": float(mask_fn.sum() / mask_wrong.sum()) if mask_wrong.sum() > 0 else math.nan,
        "fpr_conditional": float(mask_fp.sum() / mask_wrong.sum()) if mask_wrong.sum() > 0 else math.nan,
        "wpr_conditional": float((mask_wrong & both_p).sum() / mask_wrong.sum()) if mask_wrong.sum() > 0 else math.nan,
        "pred_fg_over_gt_fg": float(n_p_fg / n_y_fg) if n_y_fg > 0 else math.nan,
    }


def error_rate_rib(pred, y, bg_class=0):
    """error rate specific for ribsegv2 dataset:
    - Neighbouring-rib Shift Rate: p(|pred - y| == 1 | pred \in FG, y \in FG)
    - Far-rib Shift Rate: p(|pred - y| > 1 | pred \in FG, y \in FG)
    """
    bg_class = np.asarray([bg_class]).flatten()
    mask_bg = np.isin(y, bg_class)
    mask_bg_pred = np.isin(pred, bg_class)
    both_p = ~mask_bg & ~mask_bg_pred
    mask_wrong_pos = both_p & (pred != y)
    shift = np.abs(pred - y)
    mask_cross_side = ((12 == pred) & (13 == y)) | ((13 == pred) & (12 == y))
    mask_ns = both_p & (shift == 1) & ~mask_cross_side
    mask_fs = both_p & ((shift > 1) | mask_cross_side)
    return {
        # unconditioned
        "nsr": float(mask_ns.sum() / both_p.sum()) if both_p.sum() > 0 else math.nan,
        "fsr": float(mask_fs.sum() / both_p.sum()) if both_p.sum() > 0 else math.nan,
        # conditioned on wrong positive, i.e. split wrong positive into these two types
        "nsr_conditional": float((mask_ns & mask_wrong_pos).sum() / mask_wrong_pos.sum()) if mask_wrong_pos.sum() > 0 else math.nan,
        "fsr_conditional": float((mask_fs & mask_wrong_pos).sum() / mask_wrong_pos.sum()) if mask_wrong_pos.sum() > 0 else math.nan,
    }


def rib_id_shift_rate(pred, y):
    """several error rates in rib semantic segmentation
    - NSR: Neighbouring-rib Shift Rate, predicted and ground-truth class id shift by 1 -> confusing neighbouring ribs
    - FSR: Far-rib Shift Rate, shift more than 1 -> confusing far ribs
    - FNR: false-negative rate
    - FPR: false-positive rate
    The class set is hard-coded: {bg(0), rib1 - rib24}, see RibSegV2 dataset.
    Args:
        pred: int numpy.ndarray, predicted class id
        y: same shape as pred, ground-truth class id
        bg_class: int = 0, class id of background
    Returns:
        nsr: float in [0, 1], Neighbouring-rib Shift Rate
        fsr: float in [0, 1], Far-rib Shift Rate
        fnr: float in [0, 1], false-negative rate
        fpr: float in [0, 1], false-positive rate
        n_consensus_pos: int, num of consensus rib points (predicted as rib & ground-truth is rib) in this batch
    """
    bg_class = 0
    num_classes = 24 + 1
    mask = (bg_class != y) & (bg_class != pred) # consensus positive
    # ignore corss-side rib confusion
    mask &= ~ ((12 == pred) & (13 == y)) # cross side
    mask &= ~ ((13 == pred) & (12 == y)) # cross side
    shift = np.abs(pred - y)
    # left & right id shifting (similar to global shift)
    left_mask = (1 <= y) & (y <= 12) & mask
    right_mask = (13 <= y) & (y <= 24) & mask
    # class-wise shifting
    cw_nsr = np.zeros(num_classes, dtype=float)
    cw_fsr = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        _m = (c == y) & mask
        _s = shift[_m]
        _d = _m.sum()
        cw_nsr[c] = np.divide((1 == _s).sum(), _d) # 0/0 = NaN
        cw_fsr[c] = np.divide((_s > 1).sum(), _d)

    oracle_neg = (y == bg_class)
    oracle_pos = (y != bg_class)
    n_neg = oracle_neg.sum()
    n_pos = oracle_pos.sum()

    res = {
        "n_consensus_pos": int(mask.sum()),
        "bg_ratio": float(n_neg / y.size),
        # neighbour/far shift rate: global, left/right, class-wise
        "nsr": float(np.divide( (1 == shift[mask]).sum(), mask.sum() )), # 0/0 = NaN
        "nsr_left": float(np.divide( (1 == shift[left_mask]).sum(), left_mask.sum() )),
        "nsr_right": float(np.divide( (1 == shift[right_mask]).sum(), right_mask.sum() )),
        "nsr_cw": np.round(cw_nsr, 4).tolist(),
        "fsr": float(np.divide( (shift[mask] > 1).sum(), mask.sum() )),
        "fsr_left": float(np.divide( (shift[left_mask] > 1).sum(), left_mask.sum() )),
        "fsr_right": float(np.divide( (shift[right_mask] > 1).sum(), right_mask.sum() )),
        "fsr_cw": np.round(cw_fsr, 4).tolist(),
        # FP, FN
        "fnr": float(np.divide( ((pred == bg_class) & oracle_pos).sum(), n_pos )),
        "fpr": float(np.divide( ((pred != bg_class) & oracle_neg).sum(), n_neg )),
    }
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v, 4)

    return res


def clswise_nsr_dist(pred, label, num_classes=25):
    """Class-wise Neighbouring-rib Shift Rate (NSR) intermediate results,
    compatible with the sum/count aggregation pattern used in clswise_cm_metrics_dist.

    NSR for class c = (# consensus-positive points of class c where |pred - label| == 1)
                    / (# consensus-positive points of class c)

    'Consensus positive' means both pred != bg AND label != bg, with cross-side
    rib confusion (classes 12 <-> 13) excluded, mirroring rib_id_shift_rate logic.

    Args:
        pred:  torch.LongTensor[N], predicted class ids
        label: torch.LongTensor[N], ground-truth class ids
        num_classes: int, total number of classes including background (default 25 = bg + rib1..24)

    Returns:
        dict with keys:
            "nsr": {
                "sum":   torch.FloatTensor[num_classes],  # per-class count of neighbour-shift points
                "count": torch.FloatTensor[num_classes],  # per-class count of consensus-positive points
            }
            "nsr_cw" is just an alias / same data as "nsr"; keep one key to stay minimal.
        Divide sum/count (clamped) after aggregation to get per-class NSR.
        Invalid (zero-count) entries are 0 so they are directly summable across processes.
    """
    bg_class = 0

    # Consensus-positive mask (same logic as rib_id_shift_rate)
    mask = (pred != bg_class) & (label != bg_class)
    # Exclude cross-side rib confusion (12 <-> 13)
    mask &= ~((pred == 12) & (label == 13))
    mask &= ~((pred == 13) & (label == 12))

    shift = (pred - label).abs()

    nsr_sum   = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)
    nsr_count = torch.zeros(num_classes, dtype=torch.float64, device=pred.device)

    for c in range(num_classes):
        cls_mask = (label == c) & mask          # consensus-positive points of this class
        nsr_count[c] = cls_mask.sum()
        nsr_sum[c]   = (shift[cls_mask] == 1).sum()

    return {
        "nsr": {
            "sum":   nsr_sum,    # summable across volumes / GPUs
            "count": nsr_count,  # summable across volumes / GPUs
        }
    }


def compute_prediction_integrity(pred: np.ndarray, y: np.ndarray) -> dict:
    """
    Quantify the fractional/broken prediction level for rib semantic segmentation.

    For each ground-truth rib, checks how many distinct predicted labels its points
    received. A perfectly intact prediction means all points of a rib are assigned
    a single predicted class. Fragmentation occurs when points of one rib are split
    across multiple predicted classes.

    Args:
        pred: np.ndarray of shape [N], predicted point-wise class IDs
        y:    np.ndarray of shape [N], ground-truth point-wise class IDs

    Returns:
        dict with keys:
          - 'per_class_integrity': dict mapping gt_class -> integrity score in [0, 1]
                1.0 = all points predicted as a single class (perfectly intact)
                0.0 = maximally fragmented
          - 'per_class_dominant_pred': dict mapping gt_class -> most common predicted label
          - 'per_class_fragment_counts': dict mapping gt_class -> number of distinct predicted labels
          - 'per_class_pred_distribution': dict mapping gt_class -> {pred_label: point_count}
          - 'mean_integrity': float, mean integrity across all gt classes present (excl. background)
          - 'weighted_mean_integrity': float, point-count-weighted mean integrity (excl. background)
          - 'global_integrity': float, fraction of points correctly assigned to the dominant
                predicted label within each gt class (excl. background)
    """
    assert pred.shape == y.shape, "pred and y must have the same shape"

    gt_classes = np.unique(y)
    rib_classes = gt_classes[gt_classes != 0]  # exclude background (class 0)

    per_class_integrity = {}
    per_class_dominant_pred = {}
    per_class_fragment_counts = {}
    per_class_pred_distribution = {}

    total_rib_points = 0
    weighted_integrity_sum = 0.0
    dominant_correct_points = 0

    for gt_cls in rib_classes:
        mask = y == gt_cls
        preds_for_cls = pred[mask]
        n_points = mask.sum()

        unique_preds, counts = np.unique(preds_for_cls, return_counts=True)
        n_fragments = len(unique_preds)

        dominant_pred = unique_preds[np.argmax(counts)]
        dominant_count = counts.max()

        # Integrity: fraction of points assigned to the single dominant predicted label.
        # 1.0 = no fragmentation; lower = more broken/fractional.
        integrity = dominant_count / n_points

        per_class_integrity[int(gt_cls)] = float(integrity)
        per_class_dominant_pred[int(gt_cls)] = int(dominant_pred)
        per_class_fragment_counts[int(gt_cls)] = int(n_fragments)
        per_class_pred_distribution[int(gt_cls)] = {
            int(p): int(c) for p, c in zip(unique_preds, counts)
        }

        weighted_integrity_sum += integrity * n_points
        dominant_correct_points += dominant_count
        total_rib_points += n_points

    mean_integrity = (
        float(np.mean(list(per_class_integrity.values()))) if per_class_integrity else 0.0
    )
    weighted_mean_integrity = (
        weighted_integrity_sum / total_rib_points if total_rib_points > 0 else 0.0
    )
    global_integrity = (
        dominant_correct_points / total_rib_points if total_rib_points > 0 else 0.0
    )

    return {
        "per_class_integrity": per_class_integrity,
        "per_class_dominant_pred": per_class_dominant_pred,
        "per_class_fragment_counts": per_class_fragment_counts,
        "per_class_pred_distribution": per_class_pred_distribution,
        "mean_integrity": mean_integrity,
        "weighted_mean_integrity": weighted_mean_integrity,
        "global_integrity": global_integrity,
    }

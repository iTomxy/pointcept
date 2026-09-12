import math, warnings
import numpy as np
import medpy.metric.binary as mmb
import torch

from pointcept.utils.misc import quiet_nan, nanmean, calc_stat, natural_sort_key


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


def compute_prediction_integrity(
    pred: np.ndarray, y: np.ndarray, bg_class: int = 0
) -> dict:
    """
    Quantify the fractional/broken prediction level for rib semantic segmentation.

    For each ground-truth rib, checks how many distinct non-background predicted
    labels its recognized points received. Background predictions are excluded so
    foreground misses do not count as label fragmentation. A perfectly intact
    prediction means all recognized points of a rib are assigned a single rib
    class. Fragmentation occurs when they are split across multiple rib classes.

    Args:
        pred: np.ndarray of shape [N], predicted point-wise class IDs
        y:    np.ndarray of shape [N], ground-truth point-wise class IDs
        bg_class: int, background class excluded from the integrity calculation

    Returns:
        dict with keys:
          - 'per_class_integrity': dict mapping gt_class -> integrity score in [0, 1]
                1.0 = all recognized points predicted as one rib class
                NaN = the entire GT rib was predicted as background
          - 'per_class_dominant_pred': dict mapping gt_class -> most common predicted label
          - 'per_class_fragment_counts': dict mapping gt_class -> number of distinct predicted labels
          - 'per_class_pred_distribution': dict mapping gt_class -> non-background
                {pred_label: point_count}
          - 'per_class_recognized_points': dict mapping gt_class -> number of
                non-background predictions used as the integrity denominator
          - 'mean_integrity': float, mean across GT classes with recognized points
          - 'weighted_mean_integrity': float, recognized-point-weighted mean
          - 'global_integrity': float, fraction of recognized GT-rib points assigned
                to each GT class's dominant predicted rib label
    """
    assert pred.shape == y.shape, "pred and y must have the same shape"

    gt_classes = np.unique(y)
    rib_classes = gt_classes[gt_classes != bg_class]

    per_class_integrity = {}
    per_class_dominant_pred = {}
    per_class_fragment_counts = {}
    per_class_pred_distribution = {}
    per_class_recognized_points = {}

    valid_integrities = []
    total_recognized_points = 0
    total_dominant_points = 0

    for gt_cls in rib_classes:
        mask = y == gt_cls
        preds_for_cls = pred[mask]
        recognized_preds = preds_for_cls[preds_for_cls != bg_class]
        n_recognized = recognized_preds.size
        per_class_recognized_points[int(gt_cls)] = int(n_recognized)

        if n_recognized == 0:
            per_class_integrity[int(gt_cls)] = float("nan")
            per_class_dominant_pred[int(gt_cls)] = None
            per_class_fragment_counts[int(gt_cls)] = 0
            per_class_pred_distribution[int(gt_cls)] = {}
            continue

        unique_preds, counts = np.unique(recognized_preds, return_counts=True)
        n_fragments = len(unique_preds)

        dominant_pred = unique_preds[np.argmax(counts)]
        dominant_count = counts.max()

        # Integrity is conditioned on recognized points; FN-background points
        # are measured separately and do not count as label fragmentation.
        integrity = dominant_count / n_recognized

        per_class_integrity[int(gt_cls)] = float(integrity)
        per_class_dominant_pred[int(gt_cls)] = int(dominant_pred)
        per_class_fragment_counts[int(gt_cls)] = int(n_fragments)
        per_class_pred_distribution[int(gt_cls)] = {
            int(p): int(c) for p, c in zip(unique_preds, counts)
        }

        valid_integrities.append(integrity)
        total_dominant_points += dominant_count
        total_recognized_points += n_recognized

    mean_integrity = (
        float(np.mean(valid_integrities)) if valid_integrities else float("nan")
    )
    weighted_mean_integrity = (
        total_dominant_points / total_recognized_points
        if total_recognized_points > 0
        else float("nan")
    )
    global_integrity = (
        total_dominant_points / total_recognized_points
        if total_recognized_points > 0
        else float("nan")
    )

    return {
        "per_class_integrity": per_class_integrity,
        "per_class_dominant_pred": per_class_dominant_pred,
        "per_class_fragment_counts": per_class_fragment_counts,
        "per_class_pred_distribution": per_class_pred_distribution,
        "per_class_recognized_points": per_class_recognized_points,
        "mean_integrity": mean_integrity,
        "weighted_mean_integrity": weighted_mean_integrity,
        "global_integrity": global_integrity,
    }


def foreground_confusion_metrics(conf_mat):
    """Return per-rib integrity statistics from one full-resolution matrix.

    ``conf_mat`` has ground-truth classes on rows and predicted classes on
    columns.  This public helper deliberately only accepts RibSegV2's 25-class
    convention (background at index zero), so callers cannot accidentally
    attach anatomical-rib semantics to a generic segmentation task.  The four
    returned lists are class aligned and JSON-safe: their background entry is
    ``None`` and absent/unrecognized ribs retain the distinctions described in
    the field names.
    """
    conf_mat = np.asarray(conf_mat)
    if conf_mat.shape != (25, 25):
        raise ValueError(
            "foreground_confusion_metrics requires a 25x25 confusion matrix "
            "with background at index 0, got {}".format(conf_mat.shape)
        )
    if conf_mat.dtype.kind not in "iuf" or not np.all(np.isfinite(conf_mat)):
        raise ValueError("confusion matrix counts must be finite real numbers")
    if np.any(conf_mat < 0) or np.any(conf_mat != np.floor(conf_mat)):
        raise ValueError("confusion matrix counts must be non-negative integers")

    fg = conf_mat[1:, 1:]
    purity = [None] * 25
    recall = [None] * 25
    fragments = [None] * 25
    coverage = [None] * 25
    for cls in range(1, 25):
        row_total = conf_mat[cls].sum()
        if row_total == 0:
            continue  # anatomically absent in this volume
        recognized = fg[cls - 1].sum()
        coverage[cls] = float(recognized / row_total)
        if recognized == 0:
            continue  # present, but entirely predicted as background
        fg_row = fg[cls - 1]
        purity[cls] = float(fg_row.max() / recognized)
        recall[cls] = float(fg_row[cls - 1] / recognized)
        fragments[cls] = int(np.count_nonzero(fg_row / recognized >= 0.05))
    return {
        "fg_purity_cw": purity,
        "fg_recall_cw": recall,
        "fg_fragment_count_5pct_cw": fragments,
        "fg_recognition_coverage_cw": coverage,
    }


def summarize_foreground_confusion(records, purity_threshold=0.9):
    """Summarize class-aligned foreground metrics over ``(volume, rib)`` pairs.

    This intentionally consumes per-volume rows rather than a pooled confusion
    matrix: purity order statistics and fragment rates give every present rib
    in every volume equal weight.  Rows from older JSONL logs that lack these
    fields are ignored rather than treated as missing/failed ribs.

    ``fg_purity_tail_*`` describes only scored volume/rib pairs whose purity is
    strictly below ``purity_threshold`` (a fraction, default 0.9). Filter the
    individual pairs before computing order statistics; do not average by rib
    or volume first. An empty subset has count zero and null order statistics.
    The existing unfiltered summaries retain their definitions.
    """
    if not np.isfinite(purity_threshold) or not 0 <= purity_threshold <= 1:
        raise ValueError("purity_threshold must be a finite fraction in [0, 1]")
    keys = (
        "fg_purity_mean", "fg_purity_min", "fg_purity_median", "fg_purity_max",
        "fg_fragment_count_5pct_mean", "fg_fragment_gt1_fraction",
        "fg_fragment_eq0_fraction", "fg_scored_pair_count", "fg_absent_pair_count",
        "fg_unrecognized_pair_count", "fg_purity_lt95_fraction",
        "fg_purity_lt95_volume_count", "fg_recognition_coverage_mean",
        "fg_purity_tail_median", "fg_purity_tail_max", "fg_purity_tail_min",
    )
    result = {key: None for key in keys}
    result["fg_purity_tail_threshold"] = float(purity_threshold)
    result["fg_purity_tail_count"] = 0
    purity_values, fragment_values, coverage_values = [], [], []
    absent_count = unrecognized_count = tail_volume_count = 0
    for record in records.values() if isinstance(records, dict) else records:
        purity = record.get("fg_purity_cw")
        coverage = record.get("fg_recognition_coverage_cw")
        fragments = record.get("fg_fragment_count_5pct_cw")
        if not (isinstance(purity, (list, tuple)) and len(purity) == 25 and
                isinstance(coverage, (list, tuple)) and len(coverage) == 25 and
                isinstance(fragments, (list, tuple)) and len(fragments) == 25):
            continue
        has_tail = False
        for cls in range(1, 25):
            p, cov, frag = purity[cls], coverage[cls], fragments[cls]
            if cov is None:
                absent_count += 1
            elif p is None:
                unrecognized_count += 1
                coverage_values.append(float(cov))
            else:
                # A scored pair always has all three fields.  Ignore malformed
                # historical/manual rows rather than inventing a statistic.
                if frag is None:
                    continue
                p, frag, cov = float(p), int(frag), float(cov)
                purity_values.append(p)
                fragment_values.append(frag)
                coverage_values.append(cov)
                has_tail |= p < 0.95
        tail_volume_count += int(has_tail)

    result["fg_scored_pair_count"] = len(purity_values)
    result["fg_absent_pair_count"] = absent_count
    result["fg_unrecognized_pair_count"] = unrecognized_count
    result["fg_purity_lt95_volume_count"] = tail_volume_count
    if purity_values:
        p = np.asarray(purity_values, dtype=float)
        tail = p[p < purity_threshold]
        result["fg_purity_tail_count"] = int(tail.size)
        if tail.size:
            result.update({
                "fg_purity_tail_median": float(np.median(tail)),
                "fg_purity_tail_max": float(np.max(tail)),
                "fg_purity_tail_min": float(np.min(tail)),
            })
        fragments = np.asarray(fragment_values, dtype=float)
        result.update({
            "fg_purity_mean": float(np.mean(p)),
            "fg_purity_min": float(np.min(p)),
            "fg_purity_median": float(np.median(p)),
            "fg_purity_max": float(np.max(p)),
            "fg_fragment_count_5pct_mean": float(np.mean(fragments)),
            "fg_fragment_gt1_fraction": float(np.mean(fragments > 1)),
            "fg_fragment_eq0_fraction": float(np.mean(fragments == 0)),
            "fg_purity_lt95_fraction": float(np.mean(p < 0.95)),
        })
    if coverage_values:
        result["fg_recognition_coverage_mean"] = float(np.mean(coverage_values))
    return result


def rib_label_acc(pred, y, recall_thres=0.7):
    """label accuracy defined in Sec. V.A.1 in RibSegv2 (https://arxiv.org/abs/2210.09309)
    Args:
        pred: int numpy.ndarray, point-wise prediction
        y: same size of pred, ground-truth point-wise label
        recall_thres: float = 0.7, within [0, 1]
    Returns:
        dict with fields:
            - label_acc: float, accuracy over all present rib classes (1-24)
            - label_acc_first: float, accuracy for first rib pair (classes 1, 13)
            - label_acc_inter: float, accuracy for intermediate pairs (2-11, 14-23)
            - label_acc_last: float, accuracy for 12th rib pair (classes 12, 24)
            - recall_cw: List[float], per-class recall, length 25; index 0 (bg) = NaN;
                         NaN for ribs absent from y (incomplete rib cage cases)
            - correctly_labeled_cw: List[bool|None], per-class flag; None if absent from y
    """
    pred = pred.flatten()
    y = y.flatten()

    num_classes = 25  # class 0 (bg) + classes 1..24 (ribs)
    recall_cw = np.full(num_classes, np.nan)
    correctly_labeled_cw = [None] * num_classes

    for c in range(1, num_classes):
        n_gt = int((y == c).sum())
        if n_gt == 0:
            continue  # rib absent from this scan (incomplete rib cage)
        tp = int(((pred == c) & (y == c)).sum())
        recall_cw[c] = tp / n_gt
        # cast to a built-in bool: the numpy comparison yields np.bool_, which
        # json.dump rejects, and this dict usually ends up in a json log
        correctly_labeled_cw[c] = bool(recall_cw[c] > recall_thres)

    def _group_acc(classes):
        present = [c for c in classes if correctly_labeled_cw[c] is not None]
        if len(present) == 0:
            return math.nan
        return float(sum(correctly_labeled_cw[c] for c in present)) / len(present)

    all_ribs   = list(range(1, 25))
    first_ribs = [1, 13]
    inter_ribs = list(range(2, 12)) + list(range(14, 24))
    last_ribs  = [12, 24]

    return {
        "label_acc":            _group_acc(all_ribs),
        "label_acc_first":      _group_acc(first_ribs),
        "label_acc_inter":      _group_acc(inter_ribs),
        "label_acc_last":       _group_acc(last_ribs),
        "label_acc_recall_cw":  recall_cw.tolist(),
        "correctly_labeled_cw": correctly_labeled_cw,
    }


class SemSegEvaluator:
    """numpy.ndarray based segmentation evaluation for semantic segmentation.

    NOTE: Remember to reconstruct to original data structure when calculating
    distance-based metrics like HD and ASSD, e.g. when running for pointclouds
    converted from CT scans (voxel-grids), restore the [#points]-shape prediction
    back to a [H, W, L] shape 3D voxel-grid shape prediction volume before
    feeding to this class for evaluation.
    But overlapping metrics like mIoU and precision are fine.
    """

    METRICS = {
        "dice": mmb.dc, # = F1
        "iou": mmb.jc,
        "accuracy": lambda _B1, _B2: (_B1 == _B2).sum() / _B1.size,
        "precision": mmb.precision,
        "recall": mmb.recall, # = sensitivity, true_positive_rate
        "specificity": mmb.specificity, # = true_negative_rate
        "hd": mmb.hd,
        "assd": mmb.assd,
        "hd95": mmb.hd95,
        "asd": mmb.asd
    }
    DISTANCE_BASED = ("hd", "assd", "hd95", "asd")

    def __init__(self, n_classes, bg_classes=[], ignore_classes=[], select=[]):
        """
        Input:
            n_classes: int, length of the softmax logit vector.
                For semantic/instance segmentation, this is the number of all classes.
                For part segmentation, this is the total number of all part categories from all object classes.
            bg_classes: int or List[int], class ID of the background class/es
                (or similar classes for all uncategorised classes).
                Typically, it is class 0.
            ignore_classes: int or List[int], ID of class/es to be ignored in evaluation.
            select: List[str], name list of metrics of interest
                Provide if you only want to evaluate on these selected metrics
                instead of all supported (see METRICS).
        """
        self.n_classes = n_classes
        if bg_classes is None:
            bg_classes = []
        if isinstance(bg_classes, int):
            bg_classes = (bg_classes,)
        self.bg_classes = bg_classes

        if ignore_classes is None:
            ignore_classes = []
        if isinstance(ignore_classes, int):
            ignore_classes = (ignore_classes,)
        self.ignore_classes = ignore_classes

        if select is None:
            select = []
        if len(select) == 0:
            self.metrics = self.METRICS
        else:
            self.metrics = {}
            for m in select:
                ml = m.lower()
                assert ml in self.METRICS, "Not supported metric: {}".format(m)
                self.metrics[ml] = self.METRICS[ml]

        self.reset()

    def reset(self):
        # records:
        #  - records[metr][c][i] = <metr> score of i-th datum on c-th class, or
        #  - records[metr][c] = # of NaN caused by empty pred/label
        self.records = {}
        for metr in self.metrics:
            # self.records[metr] = [[]] * self.n_classes # wrong
            self.records[metr] = [[] for _ in range(self.n_classes)]
        for metr in self.DISTANCE_BASED + ("recall", "sensitivity", "precision"):
            if metr in self.metrics:
                self.records[f"empty_gt_{metr}"] = [0] * self.n_classes
                self.records[f"empty_pred_{metr}"] = [0] * self.n_classes

    def __call__(self, *, pred, y, spacing=None):
        """evaluates 1 prediction
        Input:
            pred: int numpy.ndarray, prediction (class ID after argmax) of one datum, not a batch
            y: same as `pred`, label (ground-truth class ID) of this datum
            spacing: float[] = None, len(spacing) = pred.ndim
        """
        for c in range(self.n_classes):
            B_pred_c = (pred == c).astype(np.int64)
            B_c      = (y == c).astype(np.int64)
            pred_l0, pred_inv_l0, gt_l0, gt_inv_l0 = B_pred_c.sum(), (1 - B_pred_c).sum(), B_c.sum(), (1 - B_c).sum()
            for metr, fn in self.metrics.items():
                is_distance_metr = metr in self.DISTANCE_BASED
                # if 0 == c and (self.ignore_bg or is_distance_metr):
                if c in self.ignore_classes or (is_distance_metr and c in self.bg_classes):
                    # always ignore bg for distance metrics
                    a = np.nan
                # elif 0 == gt_l0 and 0 == pred_l0 and metr in ("dice", "iou", "recall", "precision", "sensitivity"):
                #     a = 1
                elif 0 == gt_l0 and 0 == pred_l0 and not is_distance_metr:
                    # Ignore for dice, IoU, recall/sensitivity, precision, specificity and accuracy in this case.
                    # NOTE specificity and accuracy are defined in this cased (=1) but still ignored
                    # so that their averaging protocol/class set is consistent with other metrics.
                    a = np.nan
                elif 0 == gt_l0 and metr in ("recall", "sensitivity"):
                    a = np.nan
                    self.records["empty_gt_recall"][c] += 1
                elif 0 == pred_l0 and "precision" == metr:
                    a = np.nan
                    self.records["empty_pred_precision"][c] += 1
                elif 0 == gt_inv_l0 and 0 == pred_inv_l0 and "specificity" == metr:
                    a = 1
                elif is_distance_metr and pred_l0 * gt_l0 == 0: # at least one party is all 0
                    if 0 == pred_l0 and 0 == gt_l0: # both are all 0
                        # nips23a&d, xmed-lab/GenericSSL
                        a = 0
                    else: # only one party is all 0
                        a = np.nan
                        if 0 == pred_l0:
                            self.records[f"empty_pred_{metr}"][c] += 1
                        else: # 0 == gt_l0
                            self.records[f"empty_gt_{metr}"][c] += 1
                else: # normal cases or that medpy can solve well
                    # try:
                    if is_distance_metr:
                        a = fn(B_pred_c, B_c, voxelspacing=spacing)
                    else:
                        a = fn(B_pred_c, B_c)
                    # except:
                    #     a = np.nan

                self.records[metr][c].append(a)

    def load_from_dict(self, vw_dict):
        """Useful when aggregating volume-wise results to an overall one.
        Assumes the dict structure to be as follows:
        {
            "<METRIC>_cw": List[float]
            "<other keys>": Any
        }
        Only keys of format `<METRIC>_cw` are used, while other keys are ignored.
        """
        for metr in vw_dict:
            if not metr.endswith("_cw") or metr.startswith("empty_"): # only use class-wise records
                continue
            cw_list = vw_dict[metr]
            assert len(cw_list) == self.n_classes
            metr = metr[:-3] # remove "_cw"
            if metr not in self.metrics:
                continue
            for c, v in enumerate(cw_list):
                if c in self.ignore_classes or (metr in self.DISTANCE_BASED and c in self.bg_classes):
                    # always ignore bg for distance metrics
                    self.records[metr][c].append(np.nan)
                else:
                    self.records[metr][c].append(np.nan if v is None else float(v))

    def reduce(self, prec=4):
        """calculate class-wise & overall average
        Input:
            prec: int, decimal precision
        Output:
            res: dict
                - res[<metr>]: float, overall average
                - res[<metr>_cw]: List[float], class-wise average of each class
                - res[empty_pred|gt_<metr>]: int, overall #NaN caused by empty pred/label
                - res[empty_pred|gt_<metr>_cw]: List[int], class-wise #NaN
        """
        res = {}
        for metr in self.records:
            if metr.startswith("empty_"):
                res[metr+"_cw"] = self.records[metr]
                res[metr] = int(np.sum(self.records[metr]))
            else:
                CxN = np.asarray(self.records[metr], dtype=float)

                with warnings.catch_warnings():
                    # An all-NaN class or datum is the expected "not evaluated"
                    # case here, so the empty-slice warning is not informative.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    # class-wise average
                    cls_avg = np.nanmean(CxN, axis=1)  # [c]
                    # overall average
                    ins_avg = np.nanmean(CxN, axis=0)  # [n]
                    avg = np.nanmean(ins_avg)  # overall average across instances

                res[f"{metr}_cw"] = np.round(cls_avg, prec).tolist()
                res[metr] = float(np.round(avg, prec))

        return res


# Default metric set for the RibSegv2 volume-wise reports. Lives here rather than
# on `Ribsegv2VolumeTester` so a CPU-only post-processing script (e.g.
# tools/ribsegv2/combine_2stage.py) can score with the same set without importing
# pointcept.engines.test, which pulls in the `pointops` CUDA extension.
RIBSEG_DEFAULT_METRICS = ("dice", "iou", "precision", "recall", "specificity", "accuracy")


def eval_volume(pred, label, n_grid, num_classes, bg_class, metrics, rib_metrics=False,
                conf_mat=None):
    """all metrics of a single volume, at full resolution

    Extracted from `Ribsegv2VolumeTester.eval_volume` (A10) so it can be reused
    from a plain script (e.g. `tools/ribsegv2/combine_2stage.py`) without a CUDA model
    or a dataloader behind it.
    Args:
        pred: int numpy.ndarray[n_points], point-wise predicted class id
        label: same shape as `pred`, ground-truth class id
        n_grid: int, how many points the model actually saw
        num_classes: int, #classes
        bg_class: int, class id of background
        metrics: List[str], subset of SemSegEvaluator.METRICS to compute
        rib_metrics: bool = False, also report the rib-index metrics (shift
            rates, label accuracy, off-by-one confusion). They hard-code the
            25-class rib set; only set this for that class count.
        conf_mat: optional full-resolution [num_classes, num_classes] matrix
            with GT rows and prediction columns.  Reusing the tester's matrix
            avoids a second full-array bincount.  It is required only to
            persist the RibSegV2 integrity matrix; standalone callers may omit
            it and it will be computed once here.
    Returns:
        dict, `<metric>_cw` holds per-class values (NaN where the class is
        absent from this volume) and the rest are volume-level scalars.
    """
    # a throw-away evaluator per volume: its `records[metr][c]` is a 1-element
    # list, which keeps NaN intact. `reduce()` would turn NaN into 0.0 here.
    evaluator = SemSegEvaluator(
        n_classes=num_classes, bg_classes=[bg_class], select=metrics
    )
    with quiet_nan():
        evaluator(pred=pred, y=label)
        row = {
            "{}_cw".format(m): [float(evaluator.records[m][c][0]) for c in range(num_classes)]
            for m in evaluator.metrics
        }
        row["n_points"] = int(label.size)
        row["n_grid_points"] = int(n_grid)
        row.update(error_rate(pred, label, bg_class))
        integrity = compute_prediction_integrity(pred, label, bg_class=bg_class)
        row.update({k: integrity[k] for k in ("mean_integrity", "weighted_mean_integrity", "global_integrity")})
        is_rib_task = rib_metrics and num_classes == 25 and bg_class == 0
        if is_rib_task:
            row.update(error_rate_rib(pred, label, bg_class))
            row.update(rib_label_acc(pred, label))
            # left/right and class-wise breakdown of the rib-id shift
            shift = rib_id_shift_rate(pred, label)
            row.update({k: shift[k] for k in ("nsr_left", "nsr_right", "fsr_left", "fsr_right", "nsr_cw", "fsr_cw")})
            if conf_mat is None:
                conf_mat = np.bincount(
                    label.astype(np.int64) * num_classes + pred.astype(np.int64),
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
            else:
                conf_mat = np.asarray(conf_mat)
                if conf_mat.shape != (num_classes, num_classes):
                    raise ValueError("conf_mat shape {} does not match {} classes".format(
                        conf_mat.shape, num_classes
                    ))
            row["confusion"] = conf_mat.tolist()
            row.update(foreground_confusion_metrics(conf_mat))
    return row


def reduce_records(records, conf_mat, num_classes, bg_class, metrics, rib_metrics=False):
    """average every per-volume record across volumes

    Extracted from `Ribsegv2VolumeTester.reduce` (A10); see `eval_volume` above
    for why.
    Args:
        records: dict, volume id -> the dict `eval_volume` returned for it
        conf_mat: int numpy.ndarray[num_classes, num_classes], summed across volumes
        num_classes: int, #classes
        bg_class: int, class id of background
        metrics: List[str], subset of SemSegEvaluator.METRICS `records` was built with
        rib_metrics: bool = False, also reduce the rib-index metrics; must match
            what `eval_volume` was called with, since it decides which keys are present.
    """
    n_cls = num_classes
    order = sorted(records, key=natural_sort_key)
    fg = [c for c in range(n_cls) if c != bg_class]

    # class-wise overlap metrics, via SemSegEvaluator's own aggregation path
    evaluator = SemSegEvaluator(
        n_classes=n_cls, bg_classes=[bg_class], select=metrics
    )
    for name in order:
        evaluator.load_from_dict(records[name])
    with quiet_nan():
        summary = evaluator.reduce()

    # `reduce()` averages over every class including background, which the
    # background dominates. Redo the overall number over foreground only,
    # still averaging per volume first.
    for m in metrics:
        cls_by_vol = np.asarray(
            [[records[n]["{}_cw".format(m)][c] for n in order] for c in fg], dtype=float
        )  # [n_fg_classes, n_volumes]
        per_volume = nanmean(cls_by_vol, axis=0)
        summary["{}_fg".format(m)] = nanmean(per_volume)
        summary["{}_fg_vol".format(m)] = calc_stat(per_volume, percentages=[5, 25, 50, 75, 95], prec=4)

    # volume-level scalars: error rates, label accuracy, integrity, sizes
    for k, v in records[order[0]].items():
        if k.endswith("_cw") or not isinstance(v, (int, float)):
            continue
        vals = np.asarray([records[n].get(k, np.nan) for n in order], dtype=float)
        summary[k] = nanmean(vals)
        summary[k + "_vol"] = calc_stat(vals, percentages=[5, 25, 50, 75, 95], prec=4)

    is_rib_task = rib_metrics and num_classes == 25 and bg_class == 0
    if not is_rib_task:
        return summary

    # These are volume/rib-pair statistics, never statistics from the pooled
    # matrix below.  Do not manufacture them when reducing older records.
    if any("fg_purity_cw" in records[name] for name in order):
        summary.update(summarize_foreground_confusion(records))

    # Is the residual rib-vs-rib confusion concentrated on adjacent ribs?
    # A dominant off-by-one band means the model cannot count ribs from the
    # top, i.e. it is missing global context rather than local shape.
    rib = np.arange(1, n_cls)
    fg_cm = conf_mat[1:, 1:].astype(np.float64)
    offset = np.abs(rib[:, None] - rib[None, :])
    cross_side = ((rib[:, None] == 12) & (rib[None, :] == 13)) | \
                 ((rib[:, None] == 13) & (rib[None, :] == 12))
    wrong = fg_cm.sum() - np.trace(fg_cm)
    summary["fg_confusion"] = {
        "n_wrong_rib_points": int(wrong),
        "off_by_1_share": float(fg_cm[(offset == 1) & ~cross_side].sum() / max(wrong, 1.0)),
        "off_by_2_share": float(fg_cm[(offset == 2) & ~cross_side].sum() / max(wrong, 1.0)),
        "cross_side_share": float(fg_cm[cross_side].sum() / max(wrong, 1.0)),
    }
    return summary

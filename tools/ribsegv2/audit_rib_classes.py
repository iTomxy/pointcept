#!/usr/bin/env python3
"""Rank per-volume rib-label failures for a stratified visual audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_CLASSES = (12, 24)
RIB_LABEL_MIN = 1
RIB_LABEL_MAX = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate directional per-class rib metrics and rank volumes by "
            "auditing stratum."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("exp/ribsegv2/final"),
        help="Experiment directory containing result/ and the per-volume JSONL.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Prediction NPZ directory (default: <experiment-dir>/result).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to audit (default: test).",
    )
    parser.add_argument(
        "--volume-log",
        type=Path,
        default=None,
        help=(
            "Per-volume JSONL used to select the latest run "
            "(default: <experiment-dir>/<split>-Ribsegv2VolumeTester-per_volume.jsonl)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory "
            "(default: <experiment-dir>/reports/rib_class_audit/<split>)."
        ),
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Rib classes to audit (default: 12 24).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of volumes retained per class and stratum (default: 10).",
    )
    parser.add_argument(
        "--min-gt-points",
        type=int,
        default=1,
        help="Minimum GT points for GT-conditioned rankings (default: 1).",
    )
    parser.add_argument(
        "--min-pred-points",
        type=int,
        default=1,
        help="Minimum predicted points for prediction-conditioned rankings (default: 1).",
    )
    parser.add_argument(
        "--control-threshold",
        type=float,
        default=0.9,
        help="Minimum recall, precision, and integrity for good controls (default: 0.9).",
    )
    parser.add_argument(
        "--integrity-threshold",
        type=float,
        default=0.95,
        help=(
            "Maximum integrity for the low-integrity stratum. Values close to "
            "1 are minor rib-label spill and are excluded (default: 0.95)."
        ),
    )
    parser.add_argument(
        "--pred-key", default="pred", help="Prediction array key in each NPZ."
    )
    parser.add_argument(
        "--label-key", default="label", help="GT-label array key in each NPZ."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-class ranked lists in stdout.",
    )
    return parser.parse_args()


def rib_neighbors(class_id: int) -> tuple[int, ...]:
    """Return adjacent rib IDs on the same side of the rib cage."""
    neighbors = []
    for candidate in (class_id - 1, class_id + 1):
        if not RIB_LABEL_MIN <= candidate <= RIB_LABEL_MAX:
            continue
        if {class_id, candidate} == {12, 13}:
            continue
        neighbors.append(candidate)
    return tuple(neighbors)


def load_latest_volume_ids(path: Path) -> tuple[list[str], str | None]:
    """Read IDs from the final timestamp-delimited run in a per-volume JSONL."""
    volume_ids: list[str] = []
    run_time = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "time" in record:
                volume_ids = []
                run_time = str(record["time"])
            elif "vid" in record:
                volume_ids.append(str(record["vid"]))

    if not volume_ids:
        raise ValueError(f"No volume records found in {path}")
    if len(volume_ids) != len(set(volume_ids)):
        raise ValueError(f"Latest run in {path} contains duplicate volume IDs")
    return volume_ids, run_time


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_confusion_counts(
    volume_id: str, pred: np.ndarray, label: np.ndarray
) -> np.ndarray:
    if pred.shape != label.shape:
        raise ValueError(
            f"Volume {volume_id}: pred shape {pred.shape} != label shape {label.shape}"
        )
    if pred.size and (
        pred.min() < 0
        or pred.max() > RIB_LABEL_MAX
        or label.min() < 0
        or label.max() > RIB_LABEL_MAX
    ):
        raise ValueError(f"Volume {volume_id}: labels must be in 0..24")

    encoded = label.astype(np.int64)
    encoded *= RIB_LABEL_MAX + 1
    encoded += pred
    return np.bincount(
        encoded, minlength=(RIB_LABEL_MAX + 1) ** 2
    ).reshape(RIB_LABEL_MAX + 1, RIB_LABEL_MAX + 1)


def audit_class(
    volume_id: str, confusion: np.ndarray, class_id: int
) -> dict[str, Any]:
    gt_counts = confusion[class_id]
    pred_counts = confusion[:, class_id]
    gt_points = int(gt_counts.sum())
    pred_points = int(pred_counts.sum())
    tp = int(confusion[class_id, class_id])
    neighbors = rib_neighbors(class_id)

    bg_miss_count = int(gt_counts[0])
    neighbor_out_count = int(gt_counts[list(neighbors)].sum()) if neighbors else 0
    other_rib_out_count = int(gt_counts[1:].sum()) - tp - neighbor_out_count

    bg_contamination_count = int(pred_counts[0])
    neighbor_in_count = int(pred_counts[list(neighbors)].sum()) if neighbors else 0
    other_rib_in_count = int(pred_counts[1:].sum()) - tp - neighbor_in_count

    if gt_points:
        full_distribution = {
            str(predicted_label): int(count)
            for predicted_label, count in enumerate(gt_counts)
            if count
        }
    else:
        full_distribution = {}

    recognized_counts = gt_counts[1:]
    recognized_points = int(recognized_counts.sum())
    if recognized_points:
        dominant_prediction = int(np.argmax(recognized_counts) + 1)
        dominant_count = int(gt_counts[dominant_prediction])
        integrity = float(dominant_count / recognized_points)
        integrity_distribution = {
            str(predicted_label): int(count)
            for predicted_label, count in enumerate(gt_counts[1:], start=1)
            if count
        }
    else:
        dominant_prediction = None
        integrity = None
        integrity_distribution = {}

    return {
        "volume_id": int(volume_id) if volume_id.isdigit() else volume_id,
        "class_id": class_id,
        "neighbor_labels": "|".join(str(label_id) for label_id in neighbors),
        "gt_points": gt_points,
        "pred_points": pred_points,
        "tp_count": tp,
        "fn_count": gt_points - tp,
        "fp_count": pred_points - tp,
        "recall": safe_rate(tp, gt_points),
        "precision": safe_rate(tp, pred_points),
        "bg_miss_count": bg_miss_count,
        "bg_miss": safe_rate(bg_miss_count, gt_points),
        "neighbor_out_count": neighbor_out_count,
        "neighbor_out": safe_rate(neighbor_out_count, gt_points),
        "other_rib_out_count": other_rib_out_count,
        "other_rib_out": safe_rate(other_rib_out_count, gt_points),
        "bg_contamination_count": bg_contamination_count,
        "bg_contamination": safe_rate(bg_contamination_count, pred_points),
        "neighbor_in_count": neighbor_in_count,
        "neighbor_in": safe_rate(neighbor_in_count, pred_points),
        "other_rib_in_count": other_rib_in_count,
        "other_rib_in": safe_rate(other_rib_in_count, pred_points),
        "integrity_recognized_points": recognized_points,
        "integrity": integrity,
        "dominant_prediction": dominant_prediction,
        "prediction_fragment_count": (
            len(integrity_distribution) if gt_points else None
        ),
        "gt_prediction_distribution": full_distribution,
        "integrity_prediction_distribution": integrity_distribution,
    }


def volume_sort_key(volume_id: Any) -> tuple[int, Any]:
    if isinstance(volume_id, int):
        return (0, volume_id)
    return (1, str(volume_id))


def rank_rows(
    rows: list[dict[str, Any]],
    classes: list[int],
    top_k: int,
    predicate: Callable[[dict[str, Any]], bool],
    metric: str,
    descending: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for class_id in classes:
        candidates = [
            row
            for row in rows
            if row["class_id"] == class_id and predicate(row)
        ]
        candidates.sort(key=lambda row: volume_sort_key(row["volume_id"]))
        candidates.sort(key=lambda row: row[metric], reverse=descending)
        for rank, row in enumerate(candidates[:top_k], start=1):
            ranked.append({"rank": rank, **row})
    return ranked


def select_strata(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    gt_ok = lambda row: row["gt_points"] >= args.min_gt_points
    pred_ok = lambda row: row["pred_points"] >= args.min_pred_points
    present_both = lambda row: gt_ok(row) and pred_ok(row)

    controls = [
        row
        for row in rows
        if present_both(row)
        and row["integrity"] is not None
        and row["recall"] >= args.control_threshold
        and row["precision"] >= args.control_threshold
        and row["integrity"] >= args.control_threshold
    ]
    controls_by_class = []
    for class_id in args.classes:
        candidates = [row for row in controls if row["class_id"] == class_id]
        candidates.sort(
            key=lambda row: (row["gt_points"], volume_sort_key(row["volume_id"]))
        )
        controls_by_class.extend(
            {"rank": rank, **row}
            for rank, row in enumerate(candidates[: args.top_k], start=1)
        )

    return {
        "bg_miss": rank_rows(
            rows,
            args.classes,
            args.top_k,
            lambda row: gt_ok(row) and row["bg_miss"] > 0,
            "bg_miss",
            True,
        ),
        "bg_contamination": rank_rows(
            rows,
            args.classes,
            args.top_k,
            lambda row: present_both(row) and row["bg_contamination"] > 0,
            "bg_contamination",
            True,
        ),
        "neighbor_out": rank_rows(
            rows,
            args.classes,
            args.top_k,
            lambda row: gt_ok(row) and row["neighbor_out"] > 0,
            "neighbor_out",
            True,
        ),
        "low_integrity": rank_rows(
            rows,
            args.classes,
            args.top_k,
            lambda row: (
                gt_ok(row)
                and row["integrity"] is not None
                and row["integrity"] < args.integrity_threshold
            ),
            "integrity",
            False,
        ),
        "gt_absent_predicted": rank_rows(
            rows,
            args.classes,
            args.top_k,
            lambda row: row["gt_points"] == 0 and pred_ok(row),
            "pred_points",
            True,
        ),
        "small_good_control": controls_by_class,
    }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def analyze_strata_relationships(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    relationship = {}
    for class_id in args.classes:
        class_rows = [row for row in rows if row["class_id"] == class_id]
        low_integrity = {
            row["volume_id"]
            for row in class_rows
            if row["integrity"] is not None
            and row["integrity"] < args.integrity_threshold
        }
        neighbor_out = {
            row["volume_id"]
            for row in class_rows
            if row["neighbor_out"] is not None and row["neighbor_out"] > 0
        }
        overlap = low_integrity & neighbor_out
        low_only = low_integrity - neighbor_out
        relationship[str(class_id)] = {
            "low_integrity_count": len(low_integrity),
            "neighbor_out_count": len(neighbor_out),
            "overlap_count": len(overlap),
            "low_integrity_overlap_fraction": (
                len(overlap) / len(low_integrity) if low_integrity else None
            ),
            "low_integrity_without_neighbor_out": sorted(
                low_only, key=volume_sort_key
            ),
        }
    return relationship


def collect_all_low_integrity(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = [
        row
        for row in rows
        if row["integrity"] is not None
        and row["integrity"] < args.integrity_threshold
    ]
    cases.sort(key=lambda row: volume_sort_key(row["volume_id"]))
    cases.sort(key=lambda row: row["class_id"])
    cases.sort(key=lambda row: row["integrity"])
    ranked_cases = [
        {"severity_rank": rank, **row}
        for rank, row in enumerate(cases, start=1)
    ]

    by_volume: dict[Any, list[dict[str, Any]]] = {}
    for row in cases:
        by_volume.setdefault(row["volume_id"], []).append(row)

    volume_rows = []
    for volume_id, volume_cases in by_volume.items():
        volume_cases.sort(key=lambda row: row["class_id"])
        min_integrity = min(row["integrity"] for row in volume_cases)
        worst_classes = [
            row["class_id"]
            for row in volume_cases
            if np.isclose(row["integrity"], min_integrity)
        ]
        neighbor_out_classes = [
            row["class_id"] for row in volume_cases if row["neighbor_out"] > 0
        ]
        other_rib_out_classes = [
            row["class_id"] for row in volume_cases if row["other_rib_out"] > 0
        ]
        volume_rows.append(
            {
                "volume_id": volume_id,
                "affected_class_count": len(volume_cases),
                "affected_classes": "|".join(
                    str(row["class_id"]) for row in volume_cases
                ),
                "min_integrity": min_integrity,
                "worst_classes": "|".join(str(c) for c in worst_classes),
                "class_integrities": {
                    str(row["class_id"]): row["integrity"] for row in volume_cases
                },
                "dominant_predictions": {
                    str(row["class_id"]): row["dominant_prediction"]
                    for row in volume_cases
                },
                "neighbor_out_classes": "|".join(
                    str(c) for c in neighbor_out_classes
                ),
                "other_rib_out_classes": "|".join(
                    str(c) for c in other_rib_out_classes
                ),
                "all_low_integrity_have_neighbor_out": (
                    len(neighbor_out_classes) == len(volume_cases)
                ),
            }
        )

    volume_rows.sort(key=lambda row: volume_sort_key(row["volume_id"]))
    volume_rows.sort(key=lambda row: row["min_integrity"])
    ranked_volumes = [
        {"rank": rank, **row}
        for rank, row in enumerate(volume_rows, start=1)
    ]
    return ranked_cases, ranked_volumes


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if not 0 <= args.control_threshold <= 1:
        raise ValueError("--control-threshold must be in [0, 1]")
    if not 0 <= args.integrity_threshold <= 1:
        raise ValueError("--integrity-threshold must be in [0, 1]")
    invalid_classes = [
        class_id
        for class_id in args.classes
        if not RIB_LABEL_MIN <= class_id <= RIB_LABEL_MAX
    ]
    if invalid_classes:
        raise ValueError(f"Rib classes must be in 1..24: {invalid_classes}")

    result_dir = args.result_dir or args.experiment_dir / "result"
    volume_log = args.volume_log or (
        args.experiment_dir
        / f"{args.split}-Ribsegv2VolumeTester-per_volume.jsonl"
    )
    output_dir = args.output_dir or (
        args.experiment_dir / "reports" / "rib_class_audit" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_ids, run_time = load_latest_volume_ids(volume_log)
    rows = []
    for volume_id in volume_ids:
        result_path = result_dir / f"{volume_id}.npz"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result for volume {volume_id}: {result_path}")
        with np.load(result_path) as result:
            if args.pred_key not in result or args.label_key not in result:
                raise KeyError(
                    f"{result_path} must contain {args.pred_key!r} and {args.label_key!r}"
                )
            pred = result[args.pred_key]
            label = result[args.label_key]
            confusion = build_confusion_counts(volume_id, pred, label)
            for class_id in args.classes:
                rows.append(audit_class(volume_id, confusion, class_id))

    rows.sort(key=lambda row: (row["class_id"], volume_sort_key(row["volume_id"])))
    strata = select_strata(rows, args)
    strata_relationships = analyze_strata_relationships(rows, args)
    all_low_integrity, low_integrity_volumes = collect_all_low_integrity(rows, args)

    metric_fields = list(rows[0])
    write_csv(output_dir / "per_class_metrics.csv", rows, metric_fields)
    write_csv(
        output_dir / "low_integrity_all.csv",
        all_low_integrity,
        ["severity_rank", *metric_fields],
    )
    low_integrity_volume_fields = [
        "rank",
        "volume_id",
        "affected_class_count",
        "affected_classes",
        "min_integrity",
        "worst_classes",
        "class_integrities",
        "dominant_predictions",
        "neighbor_out_classes",
        "other_rib_out_classes",
        "all_low_integrity_have_neighbor_out",
    ]
    write_csv(
        output_dir / "low_integrity_volumes.csv",
        low_integrity_volumes,
        low_integrity_volume_fields,
    )
    for stratum, selected_rows in strata.items():
        write_csv(
            output_dir / f"{stratum}.csv",
            selected_rows,
            ["rank", *metric_fields],
        )

    report = {
        "run_time": run_time,
        "split": args.split,
        "volume_log": str(volume_log),
        "result_dir": str(result_dir),
        "volume_count": len(volume_ids),
        "classes": args.classes,
        "neighbor_definition": {
            str(class_id): list(rib_neighbors(class_id)) for class_id in args.classes
        },
        "parameters": {
            "top_k": args.top_k,
            "min_gt_points": args.min_gt_points,
            "min_pred_points": args.min_pred_points,
            "control_threshold": args.control_threshold,
            "integrity_threshold": args.integrity_threshold,
        },
        "strata": strata,
        "strata_relationships": {
            "low_integrity_vs_neighbor_out": strata_relationships
        },
        "all_low_integrity": {
            "class_case_count": len(all_low_integrity),
            "volume_count": len(low_integrity_volumes),
            "case_file": "low_integrity_all.csv",
            "volume_file": "low_integrity_volumes.csv",
        },
    }
    with (output_dir / "audit_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Audited {len(volume_ids)} {args.split} volumes from run: {run_time}")
    print(f"Wrote per-class metrics and ranked strata to {output_dir}")
    if not args.quiet:
        for stratum, selected_rows in strata.items():
            summary = []
            for class_id in args.classes:
                volume_list = [
                    str(row["volume_id"])
                    for row in selected_rows
                    if row["class_id"] == class_id
                ]
                summary.append(
                    f"class {class_id}: {', '.join(volume_list) or 'none'}"
                )
            print(f"  {stratum}: " + "; ".join(summary))
        overlap_summary = []
        for class_id in args.classes:
            relationship = strata_relationships[str(class_id)]
            overlap_summary.append(
                f"class {class_id}: {relationship['overlap_count']}/"
                f"{relationship['low_integrity_count']}"
            )
        print(
            "  low-integrity cases also in neighbor-out: "
            + "; ".join(overlap_summary)
        )
    print(
        f"  all low-integrity cases: {len(all_low_integrity)} class cases in "
        f"{len(low_integrity_volumes)} volumes"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect volumes with low integrity on any rib class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_rib_classes import (
    RIB_LABEL_MAX,
    RIB_LABEL_MIN,
    audit_class,
    build_confusion_counts,
    collect_all_low_integrity,
    load_latest_volume_ids,
    rib_neighbors,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List train/test volumes with low integrity on any rib class."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("exp/ribsegv2/final"),
        help="Experiment directory containing result/ and per-volume JSONL files.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Prediction NPZ directory (default: <experiment-dir>/result).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <experiment-dir>/reports/rib_integrity_all_ribs).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "test"],
        help="Dataset splits to scan (default: train test).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(RIB_LABEL_MIN, RIB_LABEL_MAX + 1)),
        help="Rib classes to scan (default: all 1..24).",
    )
    parser.add_argument(
        "--integrity-threshold",
        type=float,
        default=0.95,
        help="Collect classes with integrity below this value (default: 0.95).",
    )
    parser.add_argument("--pred-key", default="pred")
    parser.add_argument("--label-key", default="label")
    return parser.parse_args()


def scan_split(
    split: str,
    result_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    volume_log = (
        args.experiment_dir
        / f"{split}-Ribsegv2VolumeTester-per_volume.jsonl"
    )
    volume_ids, run_time = load_latest_volume_ids(volume_log)
    cases = []

    for volume_id in volume_ids:
        result_path = result_dir / f"{volume_id}.npz"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result for volume {volume_id}: {result_path}")
        with np.load(result_path) as result:
            if args.pred_key not in result or args.label_key not in result:
                raise KeyError(
                    f"{result_path} must contain {args.pred_key!r} and {args.label_key!r}"
                )
            confusion = build_confusion_counts(
                volume_id, result[args.pred_key], result[args.label_key]
            )
            for class_id in args.classes:
                row = audit_class(volume_id, confusion, class_id)
                if (
                    row["integrity"] is not None
                    and row["integrity"] < args.integrity_threshold
                ):
                    cases.append(row)

    ranked_cases, volume_rows = collect_all_low_integrity(cases, args)
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = list(audit_class("", np.zeros((25, 25), dtype=np.int64), 1))
    write_csv(
        output_dir / "low_integrity_all.csv",
        ranked_cases,
        ["severity_rank", *schema],
    )
    volume_fields = [
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
        volume_rows,
        volume_fields,
    )

    neighbor_cases = sum(row["neighbor_out"] > 0 for row in cases)
    other_rib_cases = sum(row["other_rib_out"] > 0 for row in cases)
    class_counts = {
        str(class_id): sum(row["class_id"] == class_id for row in cases)
        for class_id in args.classes
    }
    report = {
        "split": split,
        "run_time": run_time,
        "volume_log": str(volume_log),
        "result_dir": str(result_dir),
        "volume_count": len(volume_ids),
        "integrity_threshold": args.integrity_threshold,
        "classes": args.classes,
        "neighbor_definition": {
            str(class_id): list(rib_neighbors(class_id))
            for class_id in args.classes
        },
        "low_integrity_class_case_count": len(cases),
        "low_integrity_volume_count": len(volume_rows),
        "low_integrity_class_counts": class_counts,
        "neighbor_out_overlap_count": neighbor_cases,
        "other_rib_out_overlap_count": other_rib_cases,
        "all_cases_have_neighbor_out": neighbor_cases == len(cases),
        "volume_ids": [row["volume_id"] for row in volume_rows],
    }
    with (output_dir / "integrity_report.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print(
        f"{split}: {len(cases)} low-integrity class cases in "
        f"{len(volume_rows)}/{len(volume_ids)} volumes"
    )
    print(
        "  volumes: "
        + (", ".join(str(row["volume_id"]) for row in volume_rows) or "none")
    )
    return report


def main() -> None:
    args = parse_args()
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
    output_root = args.output_dir or (
        args.experiment_dir / "reports" / "rib_integrity_all_ribs"
    )
    reports = {
        split: scan_split(split, result_dir, output_root, args)
        for split in args.splits
    }
    with (output_root / "integrity_report.json").open("w") as handle:
        json.dump(reports, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

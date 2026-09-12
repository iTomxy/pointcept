#!/usr/bin/env python3
"""Extract table-ready foreground purity from matched RibSegV2 evaluations.

Standard-library only; reads reports and writes stdout, never runs inference or
changes files. Thresholds and JSON purity values are fractions; Markdown cells
are percentages in the fixed order val / test.
Use --splits val to record validation before test; unrequested cells show an
em dash while a measured empty subset has count zero.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


SPLITS = ("val", "test")
STATISTICS = ("median", "max", "min")


def latest_records(path):
    """Read the final timestamp-delimited block, including an incomplete one."""
    timestamp, records = None, {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if "time" in row:
                if set(row) != {"time"} or not isinstance(row["time"], str):
                    raise ValueError(f"{path}:{line_number}: invalid run header")
                timestamp, records = row["time"], {}
                continue
            if timestamp is None or "vid" not in row or not isinstance(row.get("metrics"), dict):
                raise ValueError(f"{path}:{line_number}: expected timestamp followed by volume metrics")
            vid = str(row["vid"])
            if vid in records:
                raise ValueError(f"{path}:{line_number}: duplicate volume {vid}")
            records[vid] = row["metrics"]
    if timestamp is None or not records:
        raise ValueError(f"{path}: latest evaluation is empty or missing")
    return timestamp, records


def extract_split(log_path, split, expected_volumes, threshold):
    prefix = Path(log_path) / f"{split}-Ribsegv2VolumeTester"
    summary_path = Path(str(prefix) + ".json")
    volume_path = Path(str(prefix) + "-per_volume.jsonl")
    summary = json.loads(summary_path.read_text())
    args = summary["args"]
    data = args["data"]
    if data["num_classes"] != 25 or data["bg_class"] != 0:
        raise ValueError(f"{summary_path}: expected 25 classes with background 0")
    if data["test"]["split"] != split or data["test"].get("add_trainval_incomplete") is not False:
        raise ValueError(f"{summary_path}: wrong split or add_trainval_incomplete is not False")
    if not isinstance(args.get("weight"), str) or not args["weight"]:
        raise ValueError(f"{summary_path}: missing checkpoint path")
    if type(summary.get("epoch")) is not int or type(args.get("seed")) is not int:
        raise ValueError(f"{summary_path}: checkpoint epoch and evaluation seed must be recorded integers")
    timestamp, records = latest_records(volume_path)
    if timestamp != summary["time"]:
        raise ValueError(f"{volume_path}: latest timestamp does not match {summary_path}")
    if len(records) != expected_volumes or summary["n_volumes"] != expected_volumes:
        raise ValueError(
            f"{log_path} {split}: expected {expected_volumes} volumes; "
            f"JSONL has {len(records)}, summary has {summary['n_volumes']}"
        )

    qualifying = []
    scored = 0
    for vid in sorted(records):
        values = records[vid].get("fg_purity_cw")
        if not isinstance(values, list) or len(values) != 25 or values[0] is not None:
            raise ValueError(f"{volume_path}: volume {vid} needs 25 fg_purity_cw entries, background null")
        for rib, value in enumerate(values[1:], 1):
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{volume_path}: volume {vid}, rib {rib}: invalid purity {value!r}")
            scored += 1
            if value < threshold:  # Filter unrounded individual pairs first.
                qualifying.append({"volume": vid, "rib": rib, "purity": float(value)})
    values = [pair["purity"] for pair in qualifying]
    stats = {
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }
    return {
        "summary_path": str(summary_path),
        "volume_log": str(volume_path),
        "time": timestamp,
        "epoch": summary["epoch"],
        "checkpoint": args["weight"],
        "eval_seed": args["seed"],
        "n_volumes": len(records),
        "volume_ids": sorted(records),
        "scored_pairs": scored,
        "qualifying_pairs": len(values),
        "statistics": stats,
        "pairs": qualifying,
    }


def extract(log_paths, threshold=0.9, expected_val=73, expected_test=157, splits=SPLITS):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite fraction in [0, 1], e.g. 0.90, not 90")
    if expected_val < 1 or expected_test < 1:
        raise ValueError("expected volume counts must be positive")
    if not log_paths:
        raise ValueError("provide at least one experiment log path")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError("splits must be a nonempty unique selection of val and test")
    if tuple(split for split in SPLITS if split in splits) != splits:
        raise ValueError("splits must be in canonical order: val before test")
    if len({Path(path).resolve() for path in log_paths}) != len(log_paths):
        raise ValueError("duplicate experiment log paths")
    rows = []
    for log_path in log_paths:
        selected = {
            split: extract_split(log_path, split, count, threshold)
            for split, count in (("val", expected_val), ("test", expected_test))
            if split in splits
        }
        if len(selected) == 2:
            for key in ("epoch", "checkpoint", "eval_seed"):
                if selected["val"][key] != selected["test"][key]:
                    raise ValueError(f"{log_path}: val/test {key} mismatch")
            if set(selected["val"]["volume_ids"]) & set(selected["test"]["volume_ids"]):
                raise ValueError(f"{log_path}: val and test volume IDs overlap")
        for split in splits:
            if rows:
                reference = rows[0]["splits"][split]
                if selected[split]["volume_ids"] != reference["volume_ids"]:
                    raise ValueError(f"{log_path}: {split} volume IDs differ between experiments")
                if selected[split]["eval_seed"] != reference["eval_seed"]:
                    raise ValueError(f"{log_path}: {split} evaluation seed differs between experiments")
        rows.append({"log_path": str(log_path), "splits": selected})
    return {
        "threshold": threshold,
        "comparison": "fg_purity_cw < threshold, individual (volume, rib) pairs",
        "json_units": "fraction",
        "markdown_units": "percent",
        "split_order": list(SPLITS),
        "evaluated_splits": list(splits),
        "rows": rows,
    }


def markdown(report):
    lines = [
        f"Evaluated splits: {' / '.join(report['evaluated_splits'])}; "
        f"FG purity strictly < {report['threshold'] * 100:g}%; order: val / test. "
        "Purity values are percentages; pair counts are integers.",
        "",
        "| log path | fg purity median | fg purity max | fg purity min | "
        f"fg purity <{report['threshold'] * 100:g}% pairs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        cells = []
        for stat in STATISTICS:
            values = [row["splits"].get(split, {}).get("statistics", {}).get(stat) for split in SPLITS]
            cells.append(" / ".join("—" if value is None else f"{value * 100:.2f}" for value in values))
        counts = " / ".join("—" if split not in row["splits"] else str(row["splits"][split]["qualifying_pairs"]) for split in SPLITS)
        path = row["log_path"].replace("|", "\\|")
        lines.append("| " + " | ".join([path, *cells, counts]) + " |")
    lines.extend(["", "Verified sources:", ""])
    for row in report["rows"]:
        for split in report["evaluated_splits"]:
            source = row["splits"][split]
            lines.append(
                f"- {source['summary_path']}: time={source['time']}; "
                f"volumes={source['n_volumes']}; epoch={source['epoch']}; "
                f"eval_seed={source['eval_seed']}; checkpoint={source['checkpoint']}"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_paths", type=Path, nargs="+", help="Experiment directories, in desired table-row order")
    parser.add_argument("--threshold", type=float, default=0.9, help="Strict purity cutoff as a fraction (default: 0.90)")
    parser.add_argument("--expected-val", type=int, default=73)
    parser.add_argument("--expected-test", type=int, default=157)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="Splits to read, in val/test order (default: both); unrequested cells print an em dash",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    try:
        report = extract(args.log_paths, args.threshold, args.expected_val, args.expected_test, args.splits)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    # Validate every experiment before emitting any results.
    print(json.dumps(report, indent=2, allow_nan=False) if args.format == "json" else markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

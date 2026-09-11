#!/usr/bin/env python3
"""Plot train/val/test distributions for the rib-class audit strata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter


SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
COLORS = {"train": "#3274a1", "val": "#e1812c", "test": "#3a923a"}

STRATA = (
    {
        "name": "bg_miss",
        "title": "Terminal-rib background misses",
        "xlabel": "GT terminal rib predicted as background",
        "kind": "rate",
    },
    {
        "name": "bg_contamination",
        "title": "Terminal-label background contamination",
        "xlabel": "Terminal-label prediction contributed by background",
        "kind": "rate",
    },
    {
        "name": "neighbor_out",
        "title": "Terminal ribs predicted as neighboring ribs",
        "xlabel": "GT terminal rib predicted as a neighboring rib",
        "kind": "rate",
    },
    {
        "name": "low_integrity",
        "title": "Terminal-rib prediction fragmentation",
        "xlabel": "Fragmentation severity (1 - integrity)",
        "kind": "rate",
    },
    {
        "name": "gt_absent_predicted",
        "title": "Predicted terminal labels when GT is absent",
        "xlabel": "Predicted terminal-label points (log scale)",
        "kind": "log1p_count",
    },
    {
        "name": "small_good_control",
        "title": "Small correctly recognized terminal-rib controls",
        "xlabel": "GT terminal-rib points (log scale)",
        "kind": "log_count",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw one grouped histogram per rib-audit stratum, comparing "
            "train, validation, and test side by side."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("exp/ribsegv2/final/reports/rib_class_audit"),
        help="Directory containing <split>/per_class_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot directory (default: <report-dir>/plots).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_LABELS),
        default=list(SPLIT_LABELS),
        help="Splits to compare (default: train val test).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[12, 24],
        help="Rib classes shown as panels (default: 12 24).",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Histogram bin count (default: 10)."
    )
    parser.add_argument(
        "--min-gt-points",
        type=int,
        default=1,
        help="Minimum GT points for GT-conditioned strata (default: 1).",
    )
    parser.add_argument(
        "--min-pred-points",
        type=int,
        default=1,
        help="Minimum predicted points for prediction-conditioned strata (default: 1).",
    )
    parser.add_argument(
        "--control-threshold",
        type=float,
        default=0.9,
        help="Recall, precision, and integrity threshold for controls (default: 0.9).",
    )
    parser.add_argument(
        "--integrity-threshold",
        type=float,
        default=0.95,
        help="Maximum integrity included in the low-integrity chart (default: 0.95).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Raster DPI (default: 180).")
    return parser.parse_args()


def parse_number(value: str) -> int | float | None:
    if value == "":
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with tools/ribsegv2/audit_rib_classes.py."
        )
    rows = []
    with path.open(newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = {
                key: value
                if key
                in {
                    "neighbor_labels",
                    "gt_prediction_distribution",
                    "integrity_prediction_distribution",
                }
                else parse_number(value)
                for key, value in raw_row.items()
            }
            rows.append(row)
    return rows


def eligible(row: dict[str, Any], stratum: str, args: argparse.Namespace) -> bool:
    gt_ok = row["gt_points"] >= args.min_gt_points
    pred_ok = row["pred_points"] >= args.min_pred_points
    if stratum in {"bg_miss", "neighbor_out"}:
        return gt_ok
    if stratum == "low_integrity":
        return gt_ok and row["integrity"] is not None
    if stratum in {"bg_contamination", "small_good_control"}:
        return gt_ok and pred_ok
    if stratum == "gt_absent_predicted":
        return row["gt_points"] == 0
    raise KeyError(stratum)


def stratum_value(
    row: dict[str, Any], stratum: str, args: argparse.Namespace
) -> float | None:
    if stratum == "bg_miss":
        value = row["bg_miss"]
        return value if value > 0 else None
    if stratum == "bg_contamination":
        value = row["bg_contamination"]
        return value if value > 0 else None
    if stratum == "neighbor_out":
        value = row["neighbor_out"]
        return value if value > 0 else None
    if stratum == "low_integrity":
        severity = 1 - row["integrity"]
        return severity if row["integrity"] < args.integrity_threshold else None
    if stratum == "gt_absent_predicted":
        return float(row["pred_points"]) if row["pred_points"] >= args.min_pred_points else None
    if stratum == "small_good_control":
        is_good = (
            row["integrity"] is not None
            and row["recall"] >= args.control_threshold
            and row["precision"] >= args.control_threshold
            and row["integrity"] >= args.control_threshold
        )
        return float(row["gt_points"]) if is_good else None
    raise KeyError(stratum)


def transform_values(values: list[float], kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if kind == "log1p_count":
        return np.log10(array + 1)
    if kind == "log_count":
        return np.log10(array)
    return array


def calculate_bins(all_values: list[float], kind: str, n_bins: int) -> np.ndarray:
    if kind == "rate":
        return np.linspace(0, 1, n_bins + 1)
    transformed = transform_values(all_values, kind)
    if transformed.size == 0:
        return np.linspace(0, 1, n_bins + 1)
    lower = float(transformed.min())
    upper = float(transformed.max())
    if np.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    margin = 0.02 * (upper - lower)
    return np.linspace(lower - margin, upper + margin, n_bins + 1)


def format_count_axis(kind: str) -> FuncFormatter:
    offset = 1 if kind == "log1p_count" else 0

    def formatter(value: float, _position: int) -> str:
        count = max(0, 10**value - offset)
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        if count >= 1_000:
            return f"{count / 1_000:.1f}k"
        return f"{count:.0f}"

    return FuncFormatter(formatter)


def plot_stratum(
    spec: dict[str, str],
    metrics: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = spec["name"]
    selected: dict[int, dict[str, list[float]]] = {}
    eligible_counts: dict[int, dict[str, int]] = {}
    all_values = []

    for class_id in args.classes:
        selected[class_id] = {}
        eligible_counts[class_id] = {}
        for split in args.splits:
            class_rows = [
                row for row in metrics[split] if row["class_id"] == class_id
            ]
            eligible_rows = [
                row for row in class_rows if eligible(row, name, args)
            ]
            values = [
                value
                for row in eligible_rows
                if (value := stratum_value(row, name, args)) is not None
            ]
            selected[class_id][split] = values
            eligible_counts[class_id][split] = len(eligible_rows)
            all_values.extend(values)

    bins = calculate_bins(all_values, spec["kind"], args.bins)
    figure, axes = plt.subplots(
        1,
        len(args.classes),
        figsize=(6.3 * len(args.classes), 4.8),
        sharey=True,
        squeeze=False,
    )
    bin_widths = np.diff(bins)
    centers = bins[:-1] + bin_widths / 2
    bar_widths = bin_widths * 0.82 / len(args.splits)

    chart_data = {"bins": bins.tolist(), "classes": {}}
    for panel, class_id in enumerate(args.classes):
        axis = axes[0, panel]
        class_chart_data = {}
        for split_index, split in enumerate(args.splits):
            values = transform_values(selected[class_id][split], spec["kind"])
            counts, _ = np.histogram(values, bins=bins)
            denominator = eligible_counts[class_id][split]
            percentages = counts * 100 / denominator if denominator else counts.astype(float)
            offsets = (split_index - (len(args.splits) - 1) / 2) * bar_widths
            axis.bar(
                centers + offsets,
                percentages,
                width=bar_widths,
                color=COLORS[split],
                label=SPLIT_LABELS[split],
                alpha=0.9,
            )
            class_chart_data[split] = {
                "selected_count": len(values),
                "eligible_count": denominator,
                "bin_counts": counts.tolist(),
                "bin_percentages": percentages.tolist(),
            }

        neighbor_labels = "11" if class_id == 12 else "23" if class_id == 24 else ""
        neighbor_suffix = f"; neighbor {neighbor_labels}" if neighbor_labels else ""
        axis.set_title(f"Class {class_id}{neighbor_suffix}", fontsize=11)
        axis.set_xlabel(spec["xlabel"])
        axis.set_xlim(bins[0], bins[-1])
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        counts_text = "  |  ".join(
            f"{SPLIT_LABELS[split]} {len(selected[class_id][split])}/"
            f"{eligible_counts[class_id][split]}"
            for split in args.splits
        )
        axis.text(
            0.5,
            -0.23,
            f"Selected / eligible: {counts_text}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
        if spec["kind"] == "rate":
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        else:
            axis.xaxis.set_major_formatter(format_count_axis(spec["kind"]))
        chart_data["classes"][str(class_id)] = class_chart_data

    axes[0, 0].set_ylabel("Eligible volumes in bin (%)")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    figure.suptitle(spec["title"], fontsize=15, fontweight="bold", y=0.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(args.splits),
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.87))

    output_path = output_dir / f"{name}.{args.format}"
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    chart_data["output"] = str(output_path)
    return chart_data


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if not 0 <= args.control_threshold <= 1:
        raise ValueError("--control-threshold must be in [0, 1]")
    if not 0 <= args.integrity_threshold <= 1:
        raise ValueError("--integrity-threshold must be in [0, 1]")

    output_dir = args.output_dir or args.report_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        split: read_metrics(args.report_dir / split / "per_class_metrics.csv")
        for split in args.splits
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#777777",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    plot_data = {
        "report_dir": str(args.report_dir),
        "splits": args.splits,
        "classes": args.classes,
        "bins": args.bins,
        "control_threshold": args.control_threshold,
        "integrity_threshold": args.integrity_threshold,
        "strata": {},
    }
    for spec in STRATA:
        plot_data["strata"][spec["name"]] = plot_stratum(
            spec, metrics, output_dir, args
        )

    with (output_dir / "plot_data.json").open("w") as handle:
        json.dump(plot_data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(STRATA)} grouped stratum charts to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a fresh, provenance-labelled RibSegV2 integrity report offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_CLASSES = 25


def _eval_helpers():
    # Keep the reporting definitions in one place.  The deferred import also
    # lets ``--help`` work in a lightweight environment.
    from pointcept.utils.eval_cm import foreground_confusion_metrics, summarize_foreground_confusion
    return foreground_confusion_metrics, summarize_foreground_confusion


def latest_run(volume_log: Path) -> tuple[str, dict]:
    """Read only the final timestamp-delimited JSONL evaluation run."""
    current_time, current = None, {}
    selected_time, selected = None, None
    with volume_log.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} is not valid JSON".format(
                    volume_log, line_no)) from exc
            if set(row) == {"time"}:
                if current_time is not None:
                    selected_time, selected = current_time, current
                current_time, current = row["time"], {}
                continue
            if current_time is None:
                raise ValueError("{}:{} appears before a run timestamp".format(
                    volume_log, line_no))
            if "vid" not in row or "metrics" not in row or not isinstance(
                    row["metrics"], dict):
                raise ValueError("{}:{} must contain vid and metrics".format(
                    volume_log, line_no))
            vid = str(row["vid"])
            if vid in current:
                raise ValueError("duplicate volume {} in run {}".format(
                    vid, current_time))
            current[vid] = row["metrics"]
    if current_time is not None:
        selected_time, selected = current_time, current
    if selected_time is None or not selected:
        raise ValueError(
            "{} has no complete timestamp-delimited run".format(volume_log))
    return selected_time, selected


def _valid_confusion(value, source: str) -> np.ndarray:
    cm = np.asarray(value)
    if cm.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(
            "{} confusion must have shape (25, 25), got {}".format(
                source, cm.shape))
    if cm.dtype.kind not in "iuf" or not np.all(np.isfinite(cm)):
        raise ValueError(
            "{} confusion must contain finite numeric counts".format(source))
    if np.any(cm < 0) or not np.all(cm == np.floor(cm)):
        raise ValueError(
            "{} confusion must contain non-negative integer counts".format(
                source))
    return cm.astype(np.int64, copy=False)


def load_prediction(
        path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.is_file():
        raise FileNotFoundError(
            "missing selected volume prediction {}".format(path))
    with np.load(path, allow_pickle=False) as arrays:
        if "pred" not in arrays or "label" not in arrays:
            raise ValueError("{} must contain pred and label".format(path))
        pred, label = arrays["pred"], arrays["label"]
        voxel_index = arrays["voxel_index"] if "voxel_index" in arrays else None
    for name, values in (("pred", pred), ("label", label)):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("{} {} must be a 1D integer array".format(
                path, name))
        if np.any(values < 0) or np.any(values >= N_CLASSES):
            raise ValueError("{} {} values must be in 0..24".format(
                path, name))
    if pred.shape != label.shape:
        raise ValueError("{} pred and label shapes differ: {} vs {}".format(
            path, pred.shape, label.shape))
    return pred, label, voxel_index


def confusion_from_arrays(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    return np.bincount(
        label.astype(np.int64) * N_CLASSES + pred.astype(np.int64),
        minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def spatial_components(pred: np.ndarray,
                       voxel_index: np.ndarray,
                       connectivity: int = 26) -> dict:
    """Largest predicted-rib component share without allocating a dense CT grid."""
    if voxel_index is None:
        raise ValueError(
            "--spatial requires voxel_index in every selected NPZ")
    coords = np.asarray(voxel_index)
    if coords.ndim != 2 or coords.shape != (pred.size, 3):
        raise ValueError(
            "voxel_index must have shape [N, 3] aligned with pred")
    if coords.dtype.kind not in "iu":
        raise ValueError("voxel_index must be an integer array")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    fractions, counts = [None] * N_CLASSES, [None] * N_CLASSES
    for cls in range(1, N_CLASSES):
        # Cast before subtraction: uint voxel indices otherwise wrap at zero
        # and silently break the 6/18-neighbour connectivity filter.
        points = np.unique(coords[pred == cls].astype(np.int64), axis=0)
        counts[cls] = int(len(points))
        if not len(points):
            continue
        if len(points) == 1:
            fractions[cls] = 1.0
            continue
        pairs = cKDTree(points).query_pairs(r=1,
                                            p=np.inf,
                                            output_type="ndarray")
        if connectivity == 6:
            pairs = pairs[np.abs(points[pairs[:, 0]] -
                                 points[pairs[:, 1]]).sum(1) == 1]
        elif connectivity == 18:
            pairs = pairs[np.abs(points[pairs[:, 0]] -
                                 points[pairs[:, 1]]).sum(1) <= 2]
        if len(pairs):
            edges = np.concatenate((pairs, pairs[:, ::-1]))
            graph = coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                shape=(len(points), len(points)))
            _, labels = connected_components(graph, directed=False)
            largest = np.bincount(labels).max()
        else:
            largest = 1
        fractions[cls] = float(largest / len(points))
    return {
        "connectivity": connectivity,
        "predicted_voxel_count_cw": counts,
        "largest_component_fraction_cw": fractions,
    }


def build_report(volume_log: Path,
                 output_dir: Path,
                 pred_dir: Path | None = None,
                 expected_volumes: int | None = None,
                 spatial: bool = False,
                 connectivity: int = 26,
                 purity_threshold: float = 0.9) -> dict:
    if output_dir.exists():
        raise FileExistsError(
            "output directory must be fresh: {}".format(output_dir))
    if spatial and pred_dir is None:
        raise ValueError("--spatial requires --pred-dir")
    run_time, selected = latest_run(volume_log)
    if expected_volumes is not None and len(selected) != expected_volumes:
        raise ValueError("latest run has {} volumes, expected {}".format(
            len(selected), expected_volumes))
    foreground_metrics, summarize = _eval_helpers()
    rows, per_volume = {}, []
    for vid, metrics in selected.items():
        cm_value = metrics.get("confusion")
        pred_path = pred_dir / "{}.npz".format(
            vid) if pred_dir is not None else None
        pred = label = voxels = None
        if cm_value is not None:
            cm = _valid_confusion(cm_value, "volume {}".format(vid))
            source = "jsonl_confusion"
            if spatial:
                pred, label, voxels = load_prediction(pred_path)
                if not np.array_equal(cm, confusion_from_arrays(pred, label)):
                    raise ValueError(
                        "volume {} NPZ predictions do not match the JSONL confusion"
                        .format(vid))
        else:
            if pred_path is None:
                raise ValueError(
                    "volume {} lacks confusion; --pred-dir is required".format(
                        vid))
            pred, label, voxels = load_prediction(pred_path)
            cm, source = confusion_from_arrays(pred,
                                               label), "npz_reconstructed"
        if "n_points" in metrics and int(cm.sum()) != metrics["n_points"]:
            raise ValueError(
                "volume {} confusion size differs from the recorded n_points".
                format(vid))
        row = {"confusion": cm.tolist(), **foreground_metrics(cm)}
        if spatial:
            if pred is None:
                pred, label, voxels = load_prediction(pred_path)
            row["spatial"] = spatial_components(pred, voxels, connectivity)
        rows[vid] = row
        per_volume.append({"vid": vid, "metrics": row, "source": source})
    report = {
        "source": {
            "volume_log":
            str(volume_log),
            "run_time":
            run_time,
            "pred_dir":
            str(pred_dir) if pred_dir is not None else None,
            # Deliberately enumerate only IDs selected from the latest JSONL
            # run.  Do not infer a split by globbing an experiment's result/
            # directory, which may contain stale or another-split exports.
            "npz_paths": {
                vid: str(pred_dir / "{}.npz".format(vid))
                for vid in selected
            } if pred_dir is not None else {},
            "npz_checkpoint_provenance":
            "external/unverified; NPZ metadata does not establish checkpoint provenance",
        },
        "n_volumes": len(rows),
        "volume_ids": list(rows),
        "foreground_confusion": summarize(rows, purity_threshold=purity_threshold),
        "spatial": {
            "enabled": spatial,
            "definition":
            "largest component fraction over predicted-rib voxels"
        },
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output_dir / "per_volume.jsonl").open("x") as handle:
        handle.write(json.dumps({"time": run_time}) + "\n")
        for row in per_volume:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--expected-volumes", type=int)
    parser.add_argument(
        "--purity-threshold", type=float, default=0.9,
        help="Strict upper bound for volume/rib purity order statistics, as a fraction (default: 0.9).",
    )
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--connectivity",
                        choices=(6, 18, 26),
                        type=int,
                        default=26)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_volumes is not None and args.expected_volumes < 1:
        raise ValueError("--expected-volumes must be positive")
    build_report(**vars(args))


if __name__ == "__main__":
    main()

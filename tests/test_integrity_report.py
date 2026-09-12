import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "tools/ribsegv2/integrity_report.py"
_SPEC = importlib.util.spec_from_file_location("integrity_report", _PATH)
integrity_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(integrity_report)


def _cm(rows):
    result = np.zeros((25, 25), dtype=np.int64)
    for gt, predictions in rows.items():
        for pred, count in predictions.items():
            result[gt, pred] = count
    return result


def _write_log(path, runs):
    with path.open("w") as handle:
        for timestamp, rows in runs:
            handle.write(json.dumps({"time": timestamp}) + "\n")
            for vid, metrics in rows.items():
                handle.write(
                    json.dumps({
                        "vid": vid,
                        "metrics": metrics
                    }) + "\n")


def test_latest_run_rejects_duplicate_and_selects_only_latest(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_log(path, [("old", {"old": {}}), ("new", {"one": {}})])
    time, rows = integrity_report.latest_run(path)
    assert time == "new" and set(rows) == {"one"}

    _write_log(path, [("new", {"one": {}}), ("newer", {"one": {}, "one": {}})])
    # A JSON object cannot represent duplicate dict keys, so write the second
    # line manually to exercise the JSONL duplicate guard.
    with path.open("a") as handle:
        handle.write(json.dumps({"vid": "one", "metrics": {}}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        integrity_report.latest_run(path)


def test_build_report_uses_latest_jsonl_confusion_and_writes_nulls(tmp_path):
    log = tmp_path / "records.jsonl"
    old = _cm({1: {1: 1}}).tolist()
    latest = _cm({1: {1: 2, 2: 1}, 2: {0: 1}}).tolist()
    _write_log(log, [("old", {
        "old": {
            "confusion": old
        }
    }), ("new", {
        "v": {
            "confusion": latest
        }
    })])
    output = tmp_path / "output"
    report = integrity_report.build_report(log, output, expected_volumes=1)

    assert report["source"]["run_time"] == "new"
    assert report["foreground_confusion"]["fg_scored_pair_count"] == 1
    saved = json.loads(
        (output / "per_volume.jsonl").read_text().splitlines()[-1])
    assert saved["metrics"]["fg_purity_cw"][2] is None
    with pytest.raises(FileExistsError):
        integrity_report.build_report(log, output)


def test_npz_fallback_validates_selected_arrays_and_expected_count(tmp_path):
    log = tmp_path / "records.jsonl"
    _write_log(log, [("new", {"v": {}})])
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    np.savez(pred_dir / "v.npz", pred=np.array([1, 2]), label=np.array([1, 1]))
    report = integrity_report.build_report(log,
                                           tmp_path / "output",
                                           pred_dir=pred_dir)
    assert report["foreground_confusion"]["fg_scored_pair_count"] == 1
    assert json.loads(
        (tmp_path / "output/per_volume.jsonl"
         ).read_text().splitlines()[-1])["source"] == "npz_reconstructed"

    with pytest.raises(ValueError, match="expected"):
        integrity_report.build_report(log,
                                      tmp_path / "other",
                                      pred_dir=pred_dir,
                                      expected_volumes=2)

    np.savez(pred_dir / "v.npz", pred=np.array([25]), label=np.array([1]))
    with pytest.raises(ValueError, match="0..24"):
        integrity_report.build_report(log,
                                      tmp_path / "invalid",
                                      pred_dir=pred_dir)


def test_missing_confusion_requires_npz_and_spatial_reports_disconnected_component(
        tmp_path):
    log = tmp_path / "records.jsonl"
    _write_log(log, [("new", {"v": {}})])
    with pytest.raises(ValueError, match="pred-dir"):
        integrity_report.build_report(log, tmp_path / "missing")

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    np.savez(
        pred_dir / "v.npz",
        pred=np.array([1, 1, 1, 0]),
        label=np.array([1, 1, 1, 0]),
        voxel_index=np.array([[0, 0, 0], [1, 0, 0], [9, 0, 0], [4, 4, 4]]),
    )
    integrity_report.build_report(log,
                                  tmp_path / "spatial",
                                  pred_dir=pred_dir,
                                  spatial=True)
    row = json.loads(
        (tmp_path / "spatial/per_volume.jsonl").read_text().splitlines()[-1])
    spatial = row["metrics"]["spatial"]
    assert spatial["predicted_voxel_count_cw"][1] == 3
    assert spatial["largest_component_fraction_cw"][1] == pytest.approx(2 / 3)
    assert spatial["largest_component_fraction_cw"][2] is None


def test_unsigned_voxels_connect_and_mismatched_spatial_export_is_rejected(
        tmp_path):
    points = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.uint16)
    for connectivity in (6, 18, 26):
        result = integrity_report.spatial_components(np.array([1, 1]), points,
                                                     connectivity)
        assert result['largest_component_fraction_cw'][1] == 1
    log = tmp_path / 'input.jsonl'
    _write_log(log, [('new', {'v': {'confusion': _cm({1: {1: 2}}).tolist()}})])
    pred_dir = tmp_path / 'pred'
    pred_dir.mkdir()
    np.savez(pred_dir / 'v.npz',
             pred=np.array([2, 2]),
             label=np.array([1, 1]),
             voxel_index=points)
    with pytest.raises(ValueError, match='do not match'):
        integrity_report.build_report(log,
                                      tmp_path / 'output',
                                      pred_dir=pred_dir,
                                      spatial=True)


def test_latest_incomplete_run_does_not_fall_back_to_old(tmp_path):
    log = tmp_path / 'input.jsonl'
    _write_log(log, [('old', {'v': {}}), ('incomplete', {})])
    with pytest.raises(ValueError, match='no complete'):
        integrity_report.latest_run(log)


@pytest.mark.parametrize("threshold", [None, .85])
def test_report_persists_configurable_purity_threshold(tmp_path, threshold):
    log = tmp_path / 'input.jsonl'
    _write_log(log, [('new', {'v': {'confusion': _cm({1: {1: 4, 2: 1}}).tolist()}})])
    output = tmp_path / 'threshold-report'
    options = {} if threshold is None else {"purity_threshold": threshold}
    report = integrity_report.build_report(log, output, **options)
    summary = report['foreground_confusion']
    assert summary['fg_purity_tail_threshold'] == (.9 if threshold is None else threshold)
    assert summary['fg_purity_tail_count'] == 1
    for stat in ('median', 'max', 'min'):
        assert summary['fg_purity_tail_' + stat] == .8
    assert json.loads((output / 'report.json').read_text())['foreground_confusion'] == summary

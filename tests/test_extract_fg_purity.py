import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[1] / "tools/ribsegv2/extract_fg_purity.py"
_SPEC = importlib.util.spec_from_file_location("extract_fg_purity", _PATH)
extractor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extractor)


def write_split(run, split, volumes, epoch=78, seed=20260909):
    run.mkdir(exist_ok=True)
    prefix = run / f"{split}-Ribsegv2VolumeTester"
    summary = {
        "time": split + " evaluation",
        "epoch": epoch,
        "n_volumes": len(volumes),
        "args": {
            "weight": str(run / "model/model_best.pth"),
            "seed": seed,
            "data": {
                "num_classes": 25,
                "bg_class": 0,
                "test": {"split": split, "add_trainval_incomplete": False},
            },
        },
    }
    Path(str(prefix) + ".json").write_text(json.dumps(summary))
    rows = [{"time": summary["time"]}]
    for vid, values in volumes.items():
        rows.append({"vid": vid, "metrics": {
            "fg_purity_cw": [None, *values, *([None] * (24 - len(values)))],
            # Deliberately irrelevant: the extractor must use individual pairs.
            "mean_integrity": 0.99,
        }})
    Path(str(prefix) + "-per_volume.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "run"
    write_split(path, "val", {"v1": [.2, .6, .899999, .9, 1], "v2": [.1]})
    write_split(path, "test", {"t1": [.5, .65, None, .9]})
    return path


def extract(*runs, **kwargs):
    return extractor.extract(runs, expected_val=2, expected_test=1, **kwargs)


def test_filters_unrounded_individual_pairs_before_statistics(run):
    report = extract(run)
    assert report["threshold"] == .9
    val = report["rows"][0]["splits"]["val"]
    assert val["scored_pairs"] == 6
    assert val["qualifying_pairs"] == 4
    assert val["statistics"] == {"median": .4, "max": .899999, "min": .1}
    assert [(pair["volume"], pair["rib"]) for pair in val["pairs"]] == [
        ("v1", 1), ("v1", 2), ("v1", 3), ("v2", 1)
    ]
    assert "40.00 / 57.50 | 90.00 / 65.00 | 10.00 / 50.00 | 4 / 2" in extractor.markdown(report)
    assert "fg purity <90% pairs" in extractor.markdown(report)


def test_empty_subset_and_custom_threshold(run):
    report = extract(run, threshold=.1)
    val = report["rows"][0]["splits"]["val"]
    assert val["statistics"] == dict.fromkeys(("median", "max", "min"))
    assert val["qualifying_pairs"] == 0
    assert "— / — | — / — | — / — | 0 / 0" in extractor.markdown(report)


@pytest.mark.parametrize("threshold", [70, -.01, float("nan"), float("inf")])
def test_invalid_threshold_rejected(run, threshold):
    with pytest.raises(ValueError, match="finite fraction"):
        extract(run, threshold=threshold)


@pytest.mark.parametrize("value", [70, -.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_purity_rejected(run, value):
    write_split(run, "test", {"t1": [value]})
    with pytest.raises(ValueError, match="invalid purity"):
        extract(run)


def test_latest_block_selected_and_empty_final_block_rejected(run):
    path = run / "val-Ribsegv2VolumeTester-per_volume.jsonl"
    current = path.read_text()
    old = [{"time": "old"}, {"vid": "old-volume", "metrics": {"fg_purity_cw": []}}]
    path.write_text("".join(json.dumps(row) + "\n" for row in old) + current)
    assert extract(run)["rows"][0]["splits"]["val"]["volume_ids"] == ["v1", "v2"]
    path.write_text(current + json.dumps({"time": "unfinished"}) + "\n")
    with pytest.raises(ValueError, match="empty or missing"):
        extract(run)


@pytest.mark.parametrize("problem, message", [
    ("duplicate", "duplicate volume"),
    ("truncated", "expected 2 volumes"),
    ("timestamp", "timestamp does not match"),
    ("missing_purity", "needs 25"),
    ("background", "background null"),
])
def test_inconsistent_volume_records_rejected(run, problem, message):
    path = run / "val-Ribsegv2VolumeTester-per_volume.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if problem == "duplicate":
        rows.append(rows[-1])
    elif problem == "truncated":
        rows.pop()
    elif problem == "timestamp":
        rows[0]["time"] = "another evaluation"
    elif problem == "missing_purity":
        del rows[1]["metrics"]["fg_purity_cw"]
    else:
        rows[1]["metrics"]["fg_purity_cw"][0] = .5
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        extract(run)


@pytest.mark.parametrize("keys, value, message", [
    (("epoch",), 77, "epoch mismatch"),
    (("args", "weight"), "other.pth", "checkpoint mismatch"),
    (("args", "seed"), 42, "eval_seed mismatch"),
    (("n_volumes",), 2, "expected 1 volumes"),
    (("args", "data", "test", "split"), "val", "wrong split"),
    (("args", "data", "test", "add_trainval_incomplete"), True, "not False"),
])
def test_inconsistent_summary_rejected(run, keys, value, message):
    path = run / "test-Ribsegv2VolumeTester.json"
    summary = json.loads(path.read_text())
    target = summary
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=message):
        extract(run)


def test_compared_runs_require_matched_samples_and_seed(run, tmp_path):
    other = tmp_path / "other"
    write_split(other, "val", {"v1": [.3], "v2": [.6]}, epoch=76)
    write_split(other, "test", {"t1": [.5]}, epoch=76)
    assert len(extract(run, other)["rows"]) == 2  # Different checkpoints are valid.
    write_split(other, "test", {"wrong-volume": [.5]}, epoch=76)
    with pytest.raises(ValueError, match="volume IDs differ"):
        extract(run, other)
    write_split(other, "val", {"v1": [.3], "v2": [.6]}, epoch=76, seed=42)
    write_split(other, "test", {"t1": [.5]}, epoch=76, seed=42)
    with pytest.raises(ValueError, match="evaluation seed differs"):
        extract(run, other)
    with pytest.raises(ValueError, match="duplicate experiment"):
        extract(run, run)


def test_overlapping_splits_rejected(run):
    write_split(run, "test", {"v1": [.5]})
    with pytest.raises(ValueError, match="IDs overlap"):
        extract(run)


def test_cli_json_audit_and_no_partial_output_on_failure(run, tmp_path, capsys):
    options = ["--expected-val", "2", "--expected-test", "1"]
    assert extractor.main([str(run), *options, "--format", "json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == extract(run)
    assert extractor.main([str(run), *options, "--format", "json"]) == 0
    assert capsys.readouterr().out == output
    with pytest.raises(SystemExit) as error:
        extractor.main([str(run), str(tmp_path / "missing"), *options])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "error:" in captured.err


@pytest.mark.parametrize("split, missing", [("val", "test"), ("test", "val")])
def test_single_split_needs_only_its_files_and_keeps_cell_order(run, split, missing):
    for path in run.glob(f"{missing}-Ribsegv2VolumeTester*"):
        path.unlink()
    report = extract(run, splits=[split])
    assert report["split_order"] == ["val", "test"]
    assert report["evaluated_splits"] == [split]
    assert set(report["rows"][0]["splits"]) == {split}
    rendered = extractor.markdown(report)
    assert f"Evaluated splits: {split};" in rendered
    if split == "val":
        assert "40.00 / — | 90.00 / — | 10.00 / — | 4 / —" in rendered
    else:
        assert "— / 57.50 | — / 65.00 | — / 50.00 | — / 2" in rendered
    with pytest.raises(FileNotFoundError):
        extract(run)  # Default extraction still requires both splits.


def test_val_only_cli_distinguishes_empty_subset_from_missing_test(run, capsys):
    assert extractor.main([
        str(run), "--expected-val", "2", "--splits", "val", "--threshold", "0.1",
    ]) == 0
    output = capsys.readouterr().out
    assert "— / — | — / — | — / — | 0 / —" in output


@pytest.mark.parametrize("splits", [[], ["val", "val"], ["train"], ["test", "val"]])
def test_invalid_split_selection_rejected(run, splits):
    with pytest.raises(ValueError, match="splits"):
        extract(run, splits=splits)


def test_val_only_comparison_still_checks_samples_and_seed(run, tmp_path):
    other = tmp_path / "other"
    write_split(other, "val", {"v1": [.3], "v2": [.6]}, seed=42)
    with pytest.raises(ValueError, match="evaluation seed differs"):
        extract(run, other, splits=["val"])
    write_split(other, "val", {"v1": [.3], "wrong-volume": [.6]})
    with pytest.raises(ValueError, match="volume IDs differ"):
        extract(run, other, splits=["val"])

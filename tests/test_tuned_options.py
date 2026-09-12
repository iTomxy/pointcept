import argparse
import json
from pathlib import Path

import pytest

from tools.tuned_options import build_options
from pointcept.utils.config import Config, DictAction


def _files(tmp_path, param_dicts=True, scheduler=True):
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "optimizer = dict(type='AdamW', lr=0.01, weight_decay=0.2)\n"
        + ("param_dicts = [dict(keyword='block', lr=0.001)]\n" if param_dicts else "")
        + ("scheduler = dict(type='OneCycleLR', max_lr=[0.01, 0.001])\n" if scheduler else "")
    )
    bs, lr = tmp_path / "bs.json", tmp_path / "lr.json"
    bs.write_text(json.dumps({"batch_size": 8}))
    lr.write_text(json.dumps({"batch_size": 8, "max_lr": 0.02, "lr_ratios": [1, .1]}))
    return cfg, bs, lr


def test_grouped_options_merge_with_types_and_preserve_decay(tmp_path):
    cfg, bs, lr = _files(tmp_path)
    raw_options = build_options(str(cfg), str(bs), str(lr), .75)
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", nargs="+", action=DictAction)
    opts = parser.parse_args(["--options", *[f"{k}={v}" for k, v in raw_options]]).options
    merged = Config.fromfile(str(cfg))
    merged.merge_from_dict(opts)
    assert merged.optimizer.lr == merged.scheduler.max_lr[0] == .015
    assert merged.param_dicts[0].lr == merged.scheduler.max_lr[1] == .0015
    assert merged.batch_size == 8
    assert merged.optimizer.weight_decay == .2
    assert isinstance(merged.scheduler.max_lr, list)


def test_mismatched_batch_and_ratios_fail(tmp_path):
    cfg, bs, lr = _files(tmp_path)
    lr.write_text(json.dumps({"batch_size": 4, "max_lr": .02, "lr_ratios": [1, .1]}))
    with pytest.raises(ValueError, match="batch_size"):
        build_options(str(cfg), str(bs), str(lr))
    lr.write_text(json.dumps({"batch_size": 8, "max_lr": .02, "lr_ratios": [1]}))
    with pytest.raises(ValueError, match="lr_ratios"):
        build_options(str(cfg), str(bs), str(lr))


def test_scalar_no_group_supported(tmp_path):
    cfg, bs, lr = _files(tmp_path, param_dicts=False, scheduler=False)
    with cfg.open('a') as stream:
        stream.write("scheduler = dict(type='OneCycleLR', max_lr=0.01)\n")
    lr.write_text(json.dumps({"batch_size": 8, "max_lr": .02}))
    opts = dict(build_options(str(cfg), str(bs), str(lr)))
    assert opts["optimizer.lr"] == .02
    assert float(opts["scheduler.max_lr"]) == .02
    assert not any(key.startswith("param_dicts") for key in opts)


@pytest.mark.parametrize("field,value", [
    ("max_lr", None), ("max_lr", float("nan")), ("max_lr", -1),
    ("lr_ratios", None), ("lr_ratios", [1, 0]),
    ("lr_ratios", [1, float("inf")]), ("lr_ratios", [2, .1]),
])
def test_invalid_rate_evidence_rejected(tmp_path, field, value):
    cfg, bs, lr = _files(tmp_path)
    record = json.loads(lr.read_text())
    record[field] = value
    lr.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        build_options(str(cfg), str(bs), str(lr))


def test_extra_groups_cannot_be_invented_for_ungrouped_config(tmp_path):
    cfg, bs, lr = _files(tmp_path, param_dicts=False, scheduler=False)
    with pytest.raises(ValueError, match="lr_ratios"):
        build_options(str(cfg), str(bs), str(lr))


def test_scalar_schedule_with_groups_preserves_measured_ratios(tmp_path):
    cfg, bs, lr = _files(tmp_path)
    with cfg.open('a') as stream:
        stream.write("scheduler = dict(type='OneCycleLR', max_lr=0.01)\n")
    opts = dict(build_options(str(cfg), str(bs), str(lr)))
    assert DictAction._parse_iterable(opts['scheduler.max_lr']) == [.02, .002]
    assert opts['param_dicts.0.lr'] == .002


def test_ptv3_tuning_options_pass_preflight_and_optimizer(tmp_path):
    import torch
    from pointcept.utils.optimizer import build_optimizer
    from tools.ribsegv2.contrastive_preflight import (
        apply_tuned_options, validate_inherited_recipe,
    )

    root = Path(__file__).parents[1]
    path = root / "configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py"
    canonical = Config.fromfile(str(root / "configs/ribsegv2/semseg-pt_v3m1_0_base.py"))
    bs, lr = tmp_path / "batch.json", tmp_path / "lr.json"
    bs.write_text(json.dumps({"batch_size": 6}))
    lr.write_text(json.dumps({
        "batch_size": 6, "max_lr": .003,
        "lr_ratios": [1, canonical.param_dicts[0].lr / canonical.optimizer.lr],
    }))
    overrides = [f"{key}={value}" for key, value in build_options(str(path), str(bs), str(lr))]
    overrides.append("model.contrastive.loss_weight=5")
    cfg = apply_tuned_options(Config.fromfile(str(path)), overrides)
    validate_inherited_recipe(cfg, canonical)
    model = torch.nn.ModuleDict({
        "block": torch.nn.Linear(2, 2), "projection": torch.nn.Linear(2, 2),
    })
    optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    assert [group["lr"] for group in optimizer.param_groups] == cfg.scheduler.max_lr
    assert cfg.batch_size == 6 and cfg.model.contrastive.loss_weight == 5
    assert cfg.optimizer.lr == .003

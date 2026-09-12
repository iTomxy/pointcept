import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def preflight():
    path = Path(
        __file__).parents[1] / "tools/ribsegv2/contrastive_preflight.py"
    spec = importlib.util.spec_from_file_location("contrastive_preflight_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe():
    return dict(
        model=dict(type="DefaultSegmentorV2",
                   num_classes=25,
                   backbone_out_channels=64,
                   backbone=dict(type="PT-v3m1"),
                   criteria=[dict(type="CrossEntropyLoss")]),
        data=dict(train=dict(type="Ribsegv2Dataset")),
        optimizer=dict(type="AdamW", lr=0.005102, weight_decay=5e-4),
        scheduler=dict(type="OneCycleLR", max_lr=[0.005102, 0.0005353],
                       pct_start=0.05, anneal_strategy="cos", div_factor=10.0,
                       final_div_factor=1000.0),
        param_dicts=[dict(keyword="block", lr=0.0005353)],
        batch_size=4,
        epoch=100,
        eval_epoch=100,
        enable_amp=False,
        find_unused_parameters=False,
        mix_prob=0,
        sync_bn=False,
    )


def test_recipe_validator_accepts_tuned_rates_and_weights_and_rejects_drift(preflight):
    canonical = _recipe()
    cfg = _recipe()
    cfg["model"] = dict(cfg["model"],
                        type="ContrastiveSegmentorV2",
                        contrastive=dict(type="InstanceContrastiveLoss",
                                         loss_weight=.1, temperature=.1,
                                         anchors_per_class=4,
                                         positives_per_anchor=2,
                                         negatives_per_anchor=16,
                                         candidate_chunk_size=16384,
                                         include_background_negatives=False,
                                         bg_class=0, ignore_index=-1))
    preflight.validate_inherited_recipe(cfg, canonical)
    cfg["batch_size"] = 7
    cfg["optimizer"] = dict(canonical["optimizer"], lr=.002)
    cfg["param_dicts"] = [dict(keyword="block", lr=.0002)]
    cfg["scheduler"] = dict(canonical["scheduler"], max_lr=[.002, .0002])
    cfg["model"]["contrastive"]["loss_weight"] = 5
    preflight.validate_inherited_recipe(cfg, canonical)
    cfg["scheduler"] = dict(cfg["scheduler"], max_lr=[.002, .0003])
    with pytest.raises(AssertionError, match="scheduler"):
        preflight.validate_inherited_recipe(cfg, canonical)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 0), ("batch_size", 1.5),
    ("optimizer", dict(type="AdamW", lr=-1, weight_decay=5e-4)),
])
def test_recipe_validator_rejects_invalid_tuning(preflight, field, value):
    canonical = _recipe()
    cfg = _recipe()
    cfg["model"] = dict(cfg["model"], type="ContrastiveSegmentorV2",
                         contrastive=dict(type="InstanceContrastiveLoss",
                                          loss_weight=0, temperature=.1,
                                          anchors_per_class=4,
                                          positives_per_anchor=2,
                                          negatives_per_anchor=16,
                                          candidate_chunk_size=16384,
                                          include_background_negatives=False,
                                          bg_class=0, ignore_index=-1))
    cfg[field] = value
    with pytest.raises(AssertionError):
        preflight.validate_inherited_recipe(cfg, canonical)


def test_apply_tuned_options_uses_config_parser(preflight):
    from pointcept.utils.config import Config
    canonical = _recipe()
    cfg = Config(_recipe())
    cfg.model = dict(cfg.model, type="ContrastiveSegmentorV2",
                     contrastive=dict(type="InstanceContrastiveLoss",
                                      loss_weight=.1, temperature=.1,
                                      anchors_per_class=4, positives_per_anchor=2,
                                      negatives_per_anchor=16,
                                      candidate_chunk_size=16384,
                                      include_background_negatives=False,
                                      bg_class=0, ignore_index=-1))
    preflight.apply_tuned_options(cfg, ["batch_size=7", "optimizer.lr=0.002",
                                        "param_dicts.0.lr=0.0002",
                                        "scheduler.max_lr=[0.002,0.0002]",
                                        "model.contrastive.loss_weight=10"])
    assert cfg.batch_size == 7 and cfg.optimizer.lr == .002
    assert cfg.param_dicts[0].lr == .0002
    assert cfg.scheduler.max_lr == [.002, .0002]
    assert cfg.model.contrastive.loss_weight == 10
    preflight.validate_inherited_recipe(cfg, Config(canonical))


def test_synthetic_batch_uses_unique_grid_cells_and_cumulative_offsets(
        preflight):
    torch = pytest.importorskip("torch")
    batch = preflight.make_synthetic_input(torch,
                                           5,
                                           2,
                                           .01,
                                           torch.device("cpu"),
                                           seed=7)
    assert batch["grid_coord"].shape == (10, 3)
    assert torch.unique(batch["grid_coord"], dim=0).shape[0] == 10
    assert batch["offset"].tolist() == [5, 10]
    assert batch["segment"].tolist() == list(range(10))
    degenerate = preflight.make_synthetic_input(torch,
                                                2,
                                                2,
                                                .01,
                                                torch.device("cpu"),
                                                degenerate=True)
    assert degenerate["segment"].eq(0).all()


def test_synthetic_batch_rejects_more_than_lattice(preflight):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="lattice"):
        preflight.make_synthetic_input(torch, 128**3, 2, .01,
                                       torch.device("cpu"))

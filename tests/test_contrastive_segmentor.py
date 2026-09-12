"""Real segmentor/Point CPU tests with a synthetic backbone.

GPU imports are scoped placeholders; actual PTv3 is checked by the CUDA
preflight tool. No sparse or attention operations are emulated here.
"""
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

from pointcept.models import default as _default
from pointcept.models import contrastive as _contrastive_module


@_default.MODELS.register_module(force=True)
class SyntheticBackbone(nn.Module):
    """CPU stand-in backbone: point-wise Linear(3, 64), no sparse ops."""

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(3, 64)

    def forward(self, point):
        point.feat = self.features(point.coord)
        return point


_CACHED_API = types.SimpleNamespace(
    Baseline=_default.DefaultSegmentorV2,
    Model=_contrastive_module.ContrastiveSegmentorV2,
    Point=_default.Point,
)


@pytest.fixture(scope="session")
def model_api():
    yield _CACHED_API


def model_kwargs():
    return dict(num_classes=25,
                backbone_out_channels=64,
                backbone=dict(type="SyntheticBackbone"),
                criteria=[
                    dict(type="CrossEntropyLoss", ignore_index=-1),
                    dict(type="LovaszLoss", mode="multiclass", ignore_index=-1)
                ])


def batch(background=False):
    coord = torch.tensor([
        [0., 0., 0.],
        [1., .3, 0.],
        [5., 1., .5],
        [6., 0., 1.],
        [0., 4., 0.],
        [2., 4., 1.],
        [4., 5., 1.],
        [6., 6., 2.],
    ])
    return dict(coord=coord,
                feat=coord[:, :1],
                offset=torch.tensor([4, 8]),
                segment=torch.zeros(8, dtype=torch.long)
                if background else torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]))


def make_model(api, **kwargs):
    torch.manual_seed(12)
    return api.Model(**model_kwargs(), **kwargs)


def test_training_wrapper_and_gradients(model_api):
    model = make_model(model_api)
    data = batch()
    captured = {}
    handle = model.contrastive.register_forward_pre_hook(
        lambda module, inputs: captured.update(embedding=inputs[0],
                                               mined=inputs[1]))
    out = model(data)
    handle.remove()
    assert all(torch.is_tensor(v) and v.ndim == 0 for v in out.values())
    assert out["loss_instance_contrastive"] > 0
    assert out["contrastive_anchors"] == 8
    assert out["contrastive_clouds"] == 2
    torch.testing.assert_close(
        out["loss"], out["loss_seg"] + out["loss_instance_contrastive"])
    assert not out["loss_seg"].requires_grad
    assert not out["loss_instance_contrastive"].requires_grad
    full = model.projection(model.backbone.features(data["coord"]))
    torch.testing.assert_close(captured["embedding"],
                               full[captured["mined"].selected_indices])
    out["loss"].backward()
    for branch in (model.backbone, model.seg_head, model.projection):
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in branch.parameters())
        assert sum(p.grad.abs().sum().item() for p in branch.parameters()) > 0


def test_auxiliary_gradient_alone_reaches_decoder(model_api):
    model = make_model(model_api)
    captured = {}
    handle = model.contrastive.register_forward_hook(
        lambda module, inputs, output: captured.update(loss=output))
    model(batch())
    handle.remove()
    captured["loss"].backward()
    assert model.backbone.features.weight.grad.abs().sum() > 0
    assert model.projection[0].weight.grad.abs().sum() > 0
    assert model.seg_head.weight.grad is None


@pytest.mark.parametrize("background,weight", [(True, .1), (False, 0.)])
def test_empty_pairs_and_zero_weight_keep_projection_graph(
        model_api, background, weight):
    model = make_model(model_api, contrastive=dict(loss_weight=weight))
    out = model(batch(background))
    assert out["loss_instance_contrastive"] == 0
    out["loss"].backward()
    for parameter in model.projection.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_eval_matches_baseline_and_never_projects_or_mines(model_api):
    model = make_model(model_api).eval()
    baseline = model_api.Baseline(**model_kwargs()).eval()
    baseline.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("projection.")
        },
        strict=True)
    with patch.object(model.projection, "forward", side_effect=AssertionError("projection in eval")), \
            patch.object(model.contrastive, "mine", side_effect=AssertionError("mining in eval")):
        for include_label in (True, False):
            data = batch()
            if not include_label:
                data.pop("segment")
            with torch.no_grad():
                expected, actual = baseline(data), model(data)
            assert actual.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(actual[key],
                                           expected[key],
                                           rtol=0,
                                           atol=0)
        with torch.no_grad():
            result = model(batch(), return_point=True)
        assert isinstance(result["point"], model_api.Point)
        torch.testing.assert_close(result["point"].coord, batch()["coord"])
        torch.testing.assert_close(result["point"].offset, batch()["offset"])


def test_state_round_trip_and_projection_structure(model_api):
    model = make_model(model_api)
    other = make_model(model_api)
    other.load_state_dict(model.state_dict(), strict=True)
    assert len(model.projection) == 4
    assert isinstance(model.projection[1], nn.LayerNorm)
    assert isinstance(model.projection[2], nn.GELU)
    assert isinstance(model.projection[-1], nn.Linear)
    assert model.projection[-1].bias is not None
    model.eval()
    other.eval()
    torch.testing.assert_close(
        model(batch())["seg_logits"],
        other(batch())["seg_logits"])


def test_explicit_point_return_and_bad_inputs(model_api):
    model = make_model(model_api)
    out = model(batch(), return_point=True)
    assert isinstance(out["point"], model_api.Point)
    torch.testing.assert_close(out["point"].feat,
                               model.backbone.features(batch()["coord"]))
    data = batch()
    data.pop("offset")
    with pytest.raises(KeyError, match="offset"):
        model(data)
    data = batch()
    data["coord"] = data["coord"][:0]
    with pytest.raises(ValueError, match="nonempty"):
        model(data)
    with pytest.raises(ValueError, match="num_classes"):
        make_model(model_api, contrastive=dict(num_classes=26))


def test_resolved_config_retains_canonical_recipe():
    from pointcept.utils.config import Config
    base = Config.fromfile(
        str(ROOT / "configs/ribsegv2/semseg-pt_v3m1_0_base.py"))
    method = Config.fromfile(
        str(ROOT / "configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py"))
    assert method.model.type == "ContrastiveSegmentorV2"
    for key in ("backbone", "criteria", "backbone_out_channels",
                "num_classes"):
        assert method.model[key] == base.model[key]
    for key in ("data", "optimizer", "param_dicts", "scheduler", "epoch",
                "eval_epoch", "batch_size", "enable_amp", "sync_bn",
                "mix_prob", "hooks", "test"):
        assert method[key] == base[key]


def test_optimizer_uses_ordinary_lr_for_projection(model_api):
    from pointcept.utils.config import Config
    from pointcept.utils.optimizer import build_optimizer
    cfg = Config.fromfile(
        str(ROOT / "configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py"))
    model = make_model(model_api)
    optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    ordinary = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert all(id(p) in ordinary for p in model.projection.parameters())
    assert not any("block" in name for name, _ in model.named_parameters()
                   if name.startswith("projection."))
    assert [group["lr"]
            for group in optimizer.param_groups] == list(cfg.scheduler.max_lr)


def _ddp_step(rank, model_class, init_file):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo",
                            init_method="file://" + init_file,
                            rank=rank,
                            world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(5)
        model = DistributedDataParallel(model_class(**model_kwargs()),
                                        find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch(background=(rank == 0)))
            assert out["contrastive_anchors"].item() == (0 if rank == 0 else 8)
            out["loss"].backward()
            for parameter in model.module.projection.parameters():
                assert parameter.grad is not None and torch.isfinite(
                    parameter.grad).all()
            # The populated rank contributes a nonzero projection gradient on
            # both ranks. A second iteration detects unfinished DDP reduction.
            assert model.module.projection[0].weight.grad.abs().sum() > 0
            optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="CPU DDP test uses spawn")
def test_two_rank_ddp_with_one_degenerate_rank(model_api, tmp_path,
                                               monkeypatch):
    import socket
    import torch.multiprocessing as mp

    # Sandboxed CI may prohibit even local sockets. Leave model/reducer errors
    # as test failures; skip only an explicit networking permission denial.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip(
            "CPU DDP requires localhost sockets, denied by this sandbox")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    # Spawn, not fork: forking after earlier tests have run torch ops can
    # hang on thread-pool locks. Spawned children reimport this module, which
    # registers SyntheticBackbone at module level, so the build works there.
    mp.start_processes(_ddp_step,
                       args=(model_api.Model, str(tmp_path / "ddp-init")),
                       nprocs=2,
                       join=True,
                       start_method="spawn")

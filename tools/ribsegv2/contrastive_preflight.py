#!/usr/bin/env python3
"""CUDA preflight for the RibSegV2 PTv3 contrastive recipe.

Run this on a machine with the real Pointcept CUDA dependencies installed:

    torchrun --standalone --nproc_per_node=2 \\
      tools/ribsegv2/contrastive_preflight.py --points-per-cloud 4096

It deliberately creates no checkpoints, datasets, or exports.  The synthetic
point count is a runtime smoke check only; it is not a real-volume memory
guarantee (use ``--points-per-cloud 200000`` separately to exercise the recipe
upper bound on an appropriate GPU).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py"
CANONICAL_CONFIG = ROOT / "configs/ribsegv2/semseg-pt_v3m1_0_base.py"


def apply_tuned_options(cfg, raw_options):
    """Merge CLI KEY=VALUE overrides using Pointcept's standard value parser."""
    from pointcept.utils.config import DictAction

    options = {}
    for raw in raw_options or ():
        if "=" not in raw:
            raise ValueError("--options entries must use KEY=VALUE")
        key, value = raw.split("=", 1)
        if not key:
            raise ValueError("--options key must not be empty")
        if not value:
            raise ValueError("--options value must not be empty")
        options[key] = DictAction._parse_iterable(value)
    cfg.merge_from_dict(options)
    return cfg


def _get(mapping, name):
    return mapping[name] if isinstance(mapping, dict) else getattr(
        mapping, name)


def validate_inherited_recipe(cfg, canonical):
    """Raise AssertionError if the contrastive config drifted from the recipe."""
    model, base_model = _get(cfg, "model"), _get(canonical, "model")
    assert _get(model, "type") == "ContrastiveSegmentorV2"
    for key in ("num_classes", "backbone_out_channels", "backbone",
                "criteria"):
        assert _get(model,
                    key) == _get(base_model,
                                 key), "model.%s differs from canonical" % key
    assert _get(cfg, "data") == _get(canonical, "data"), "data differs from canonical"
    # Batch size and learning rates are deliberately selected per run.  Keep
    # every other optimizer/scheduler/group option pinned to the recipe.
    batch_size = _get(cfg, "batch_size")
    assert isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0
    optimizer, base_optimizer = _get(cfg, "optimizer"), _get(canonical, "optimizer")
    lr = _get(optimizer, "lr")
    assert isinstance(lr, (int, float)) and not isinstance(lr, bool) and math.isfinite(lr) and lr > 0
    assert {k: v for k, v in optimizer.items() if k != "lr"} == {
        k: v for k, v in base_optimizer.items() if k != "lr"
    }, "optimizer differs from canonical"
    groups, base_groups = _get(cfg, "param_dicts"), _get(canonical, "param_dicts")
    assert len(groups) == len(base_groups)
    block_lrs = []
    normalized_groups = []
    normalized_base_groups = []
    for group, base_group in zip(groups, base_groups):
        is_block = _get(group, "keyword") == "block"
        if is_block:
            group_lr = _get(group, "lr")
            assert isinstance(group_lr, (int, float)) and not isinstance(group_lr, bool)
            assert math.isfinite(group_lr) and group_lr > 0
            block_lrs.append(group_lr)
        normalized_groups.append({k: v for k, v in group.items() if not (is_block and k == "lr")})
        normalized_base_groups.append({k: v for k, v in base_group.items() if not (_get(base_group, "keyword") == "block" and k == "lr")})
    assert len(block_lrs) == 1, "expected one block parameter group"
    assert normalized_groups == normalized_base_groups, "param_dicts differs from canonical"
    scheduler, base_scheduler = _get(cfg, "scheduler"), _get(canonical, "scheduler")
    assert _get(scheduler, "max_lr") == [lr, block_lrs[0]], "scheduler max_lr must match optimizer/group lr"
    assert {k: v for k, v in scheduler.items() if k != "max_lr"} == {
        k: v for k, v in base_scheduler.items() if k != "max_lr"
    }, "scheduler differs from canonical"
    for key in ("epoch", "eval_epoch", "enable_amp",
                "find_unused_parameters", "mix_prob", "sync_bn"):
        assert _get(cfg, key) == _get(canonical,
                                      key), "%s differs from canonical" % key
    assert _get(cfg, "enable_amp") is False
    assert _get(cfg, "find_unused_parameters") is False
    assert _get(cfg, "mix_prob") == 0
    contrastive = _get(model, "contrastive")
    assert _get(contrastive, "type") == "InstanceContrastiveLoss"
    for key, expected in (("temperature", 0.1), ("anchors_per_class", 4),
                          ("positives_per_anchor", 2),
                          ("negatives_per_anchor", 16),
                          ("candidate_chunk_size", 16384),
                          ("include_background_negatives", False),
                          ("bg_class", 0), ("ignore_index", -1)):
        assert _get(contrastive, key) == expected, "contrastive.%s differs from recipe" % key
    loss_weight = _get(contrastive, "loss_weight")
    assert isinstance(loss_weight, (int, float)) and not isinstance(loss_weight, bool)
    assert math.isfinite(loss_weight) and loss_weight >= 0


def make_synthetic_input(torch,
                         points_per_cloud,
                         clouds,
                         grid_size,
                         device,
                         degenerate=False,
                         seed=0):
    """Create unique 128^3 grid cells, preserving Pointcept's offset contract."""
    if points_per_cloud <= 0 or clouds <= 0:
        raise ValueError("points_per_cloud and clouds must be positive")
    total = points_per_cloud * clouds
    lattice = 128**3
    if total > lattice:
        raise ValueError("synthetic points exceed the unique 128^3 lattice")
    generator = torch.Generator(device=device).manual_seed(seed)
    flat = torch.randperm(lattice, device=device, generator=generator)[:total]
    grid = torch.stack((flat // (128 * 128), (flat // 128) % 128, flat % 128),
                       dim=1).int()
    segment = torch.arange(total, device=device, dtype=torch.long) % 25
    if degenerate:
        segment.zero_()
    return {
        "coord":
        grid.float() * grid_size,
        "grid_coord":
        grid,
        "feat":
        torch.rand((total, 1), device=device, generator=generator),
        "segment":
        segment,
        "offset":
        torch.arange(1, clouds + 1, device=device, dtype=torch.int32) *
        points_per_cloud,
    }


def _finite_grad(torch, module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    present = [g for g in gradients if g is not None]
    finite = bool(gradients) and len(present) == len(gradients) and all(
        torch.isfinite(g).all().item() for g in present)
    return finite, any(g.abs().sum().item() > 0 for g in present)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-CUDA contrastive PTv3 preflight checks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--points-per-cloud", type=int, default=4096)
    parser.add_argument("--clouds-per-rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--degenerate-rank",
        type=int,
        default=None,
        help="Make this rank all-background to check the contrastive zero path."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--options", nargs="+", default=None,
                        metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 for the DDP preflight")
    # Keep imports after argument parsing: --help and helper-only tests work on CPU.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "contrastive_preflight requires CUDA and Pointcept's CUDA extensions; run it on the GPU runner"
        )
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    from pointcept.utils.optimizer import build_optimizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(
        "cuda", local_rank if world > 1 else torch.cuda.current_device())
    try:
        if args.degenerate_rank is not None and not 0 <= args.degenerate_rank < world:
            raise ValueError(
                "--degenerate-rank must name a participating rank")
        cfg, canonical = Config.fromfile(str(args.config)), Config.fromfile(
            str(CANONICAL_CONFIG))
        apply_tuned_options(cfg, args.options)
        validate_inherited_recipe(cfg, canonical)
        torch.manual_seed(args.seed)
        model = build_model(cfg.model).to(device)
        optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
        if world > 1:
            model = DDP(model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=False)
        torch.cuda.reset_peak_memory_stats(device)
        base = model.module if world > 1 else model
        last = None
        for step in range(args.steps):
            batch = make_synthetic_input(
                torch,
                args.points_per_cloud,
                args.clouds_per_rank,
                cfg.grid_size,
                device,
                degenerate=(rank == args.degenerate_rank),
                seed=args.seed + rank + step)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            began = time.perf_counter()
            output = model(batch)
            assert all(
                torch.is_tensor(v) and v.ndim == 0 and torch.isfinite(v)
                for v in output.values())
            value = output["loss"]
            assert value.ndim == 0 and torch.isfinite(
                value), "training loss must be finite scalar"
            value.backward()
            projection_finite, projection_nonzero = _finite_grad(
                torch, base.projection)
            backbone_finite, backbone_nonzero = _finite_grad(
                torch, base.backbone)
            assert projection_finite and backbone_finite, "missing or non-finite model gradients"
            if not (world == 1 and rank == args.degenerate_rank):
                assert backbone_nonzero, "expected nonzero backbone gradients"
                if _get(_get(cfg, "model"), "contrastive").get("loss_weight", 0) > 0:
                    assert projection_nonzero, "expected nonzero projection gradients"
            optimizer.step()
            torch.cuda.synchronize(device)
            last = (output, projection_nonzero, backbone_nonzero,
                    time.perf_counter() - began, batch)

        output, projection_nonzero, backbone_nonzero, elapsed, batch = last
        projection_calls = [0]
        hook = base.projection.register_forward_hook(
            lambda *_: projection_calls.__setitem__(0, projection_calls[0] + 1
                                                    ))
        model.eval()
        from unittest.mock import patch
        test_batch = {
            key: value
            for key, value in batch.items() if key != "segment"
        }
        with torch.no_grad(), patch.object(
                base.contrastive,
                "mine",
                side_effect=AssertionError("mining in eval")):
            eval_out = model(test_batch, return_point=True)
        hook.remove()
        point = eval_out["point"]
        assert projection_calls[0] == 0, "evaluation must not project or mine"
        assert eval_out["seg_logits"].shape == (batch["segment"].numel(),
                                                cfg.model.num_classes)
        assert point.feat.shape == (batch["segment"].numel(),
                                    cfg.model.backbone_out_channels)
        assert torch.equal(point.coord, batch["coord"]) and torch.equal(
            point.offset, batch["offset"])
        report = dict(
            device=str(device),
            config=str(args.config),
            torch=torch.__version__,
            rank=rank,
            world_size=world,
            seed=args.seed,
            points=args.points_per_cloud,
            clouds=args.clouds_per_rank,
            batch_size=cfg.batch_size,
            contrastive_loss_weight=_get(_get(cfg, "model"), "contrastive").get("loss_weight"),
            loss=float(output["loss"].detach()),
            anchors=int(output["contrastive_anchors"]),
            loss_instance_contrastive=float(
                output["loss_instance_contrastive"]),
            optimizer_lrs=[group["lr"] for group in optimizer.param_groups],
            selected=int(output["contrastive_selected_points"]),
            foreground=int(output["contrastive_foreground_points"]),
            projection_grad_nonzero=projection_nonzero,
            backbone_grad_nonzero=backbone_nonzero,
            step_seconds=elapsed,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device))
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

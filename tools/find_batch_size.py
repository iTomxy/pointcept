"""Find the largest per-GPU batch size that fits in GPU memory.

Measures a real forward+backward of the configured model, rather than guessing
from parameter counts: for point-cloud models the activations dominate, and how
they scale depends on the attention/pooling settings.

Sizing is done against the CONFIGURED upper bound, not a typical batch. When the
pipeline caps a sample at `max_points` (see the `LimitPoint` transform), the
worst batch training can ever assemble is exactly `bs * max_points` points, so
that is what gets measured. Sizing on an average batch leaves training one
unlucky shuffle away from an out-of-memory crash hours in.

Usage:
    python tools/find_batch_size.py --config configs/x/y.py \
        --num-gpus 2 --mem-frac 0.9 --out bs.json
"""

import os
import sys

# runnable as `python tools/<name>.py` from anywhere; scripts/*.sh instead export
# PYTHONPATH, so guard against inserting the repo root twice
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse, json
import torch

from pointcept.utils.config import Config, DictAction
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to the config file")
    p.add_argument("--num-gpus", type=int, default=1, help="GPUs the real run will use; scales the reported total")
    p.add_argument("--max-points", type=int, default=0,
                   help="points per sample to size against; 0 = cfg.max_points, else measured from the data")
    p.add_argument("--max-batch-size", type=int, default=16, help="stop searching above this per-GPU batch size")
    p.add_argument("--mem-frac", type=float, default=0.9,
                   help="largest fraction of GPU memory the worst-case step may reserve")
    p.add_argument("--steps", type=int, default=4, help="optimiser steps per trial; >1 catches optimiser-state growth")
    p.add_argument("--scan-volumes", type=int, default=64,
                   help="samples to inspect when max_points is not configured")
    p.add_argument("--out", default="", help="where to write the JSON result")
    p.add_argument("--options", nargs="+", action=DictAction, help="override config entries, e.g. max_points=200000")
    return p.parse_args(argv)


def worst_case_sample(dataset, n_points, scan):
    """a real sample truncated to exactly `n_points`, plus what was achievable"""
    best, best_n = None, -1
    for i in range(min(len(dataset), scan)):
        s = dataset[i]
        n = s["coord"].shape[0]
        if n > best_n:
            best, best_n = s, n
        if n >= n_points:
            break
    n = min(n_points, best_n)
    sample = {
        k: (v[:n] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == best_n else v)
        for k, v in best.items()
    }
    sample["offset"] = torch.tensor([n])
    return sample, n, best_n


def measure(cfg, sample, bs, steps):
    """peak allocated/reserved GiB for `steps` fwd+bwd on a batch of `bs` copies"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg.model).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = collate_fn([sample] * bs)
    n_points = batch["coord"].shape[0]
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    try:
        for _ in range(steps):
            loss = model(batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        alloc = torch.cuda.max_memory_allocated() / 2 ** 30
        reserved = torch.cuda.max_memory_reserved() / 2 ** 30
        oom = False
    except torch.cuda.OutOfMemoryError:
        alloc = reserved = float("nan")
        oom = True
    del model, optimizer, batch
    torch.cuda.empty_cache()
    return n_points, alloc, reserved, oom


def main(argv=None):
    args = parse_args(argv)
    assert torch.cuda.is_available(), "find_batch_size needs a GPU"
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    dataset = build_dataset(cfg.data.train)
    n_points = args.max_points or cfg.get("max_points", 0)
    source = "--max-points" if args.max_points else ("cfg.max_points" if n_points else "measured")
    if not n_points:
        n_points = max(dataset[i]["coord"].shape[0] for i in range(min(len(dataset), args.scan_volumes)))
    sample, per_sample, largest = worst_case_sample(dataset, n_points, args.scan_volumes)

    total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    budget = args.mem_frac * total_mem
    print("device        : %s (%.1f GiB, budget %.1f GiB at mem_frac=%.2f)" % (
        torch.cuda.get_device_name(0), total_mem, budget, args.mem_frac), flush=True)
    if per_sample >= n_points:
        print("points/sample : %d (requested %d from %s)" % (per_sample, n_points, source), flush=True)
    else:
        # Note: overriding a top-level config variable with --options does NOT
        # reach the transform list, because the config file already substituted
        # its value into the pipeline dict. The achievable size is authoritative.
        print("points/sample : %d  [requested %d from %s, but no sample reaches it -- "
              "the pipeline caps samples lower, so THIS is the real bound]"
              % (per_sample, n_points, source), flush=True)
    print("\n%-4s %-11s %-9s %-9s %-8s" % ("bs", "batch pts", "alloc", "reserved", "of budget"), flush=True)

    trials, chosen = [], 0
    for bs in range(1, args.max_batch_size + 1):
        pts, alloc, reserved, oom = measure(cfg, sample, bs, args.steps)
        fits = (not oom) and reserved <= budget
        trials.append(dict(batch_size=bs, n_points=pts, oom=oom,
                           alloc_gib=None if oom else round(alloc, 3),
                           reserved_gib=None if oom else round(reserved, 3), fits=fits))
        print("%-4d %-11d %-9s %-9s %-8s %s" % (
            bs, pts, "OOM" if oom else "%.2f" % alloc, "OOM" if oom else "%.2f" % reserved,
            "-" if oom else "%.0f%%" % (100 * reserved / budget), "" if fits else "<- over budget"), flush=True)
        if not fits:
            break
        chosen = bs

    assert chosen >= 1, (
        "even batch size 1 does not fit in %.1f GiB. Reduce max_points, shrink the "
        "model, or enable AMP." % budget)
    result = dict(
        config=args.config, gpu=torch.cuda.get_device_name(0), total_mem_gib=round(total_mem, 2),
        mem_frac=args.mem_frac, points_per_sample=per_sample, points_bound_source=source,
        steps=args.steps, batch_size_per_gpu=chosen, num_gpus=args.num_gpus,
        batch_size=chosen * args.num_gpus, trials=trials,
    )
    print("\nbatch_size_per_gpu=%d  x %d GPUs  ->  batch_size=%d" % (
        chosen, args.num_gpus, result["batch_size"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("written to %s" % args.out, flush=True)
    return result


if __name__ == "__main__":
    main()

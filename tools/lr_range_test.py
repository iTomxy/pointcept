"""Learning-rate range test (Smith, arXiv:1506.01186).

Sweeps the learning rate exponentially over a few hundred steps and records the
smoothed training loss. The useful learning rate is where the loss is still
descending fastest; past that the curve flattens and then diverges.

Two details that make the answer transferable to the real run:

- The effective batch matches `cfg.batch_size` via gradient accumulation, so the
  gradient noise being probed is the noise training will actually see. Testing
  at a different batch size measures a different optimisation problem.
- Configs with `param_dicts` (e.g. PTv3's separate "block" group) keep their
  relative learning rates; the sweep scales the whole schedule rather than
  collapsing the groups onto one value.

Usage:
    python tools/lr_range_test.py --config configs/x/y.py \
        --micro-batch-size 3 --out lr.json
"""

import os
import sys

# runnable as `python tools/<name>.py` from anywhere; scripts/*.sh instead export
# PYTHONPATH, so guard against inserting the repo root twice
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse, json, math
import numpy as np
import torch

from pointcept.utils.config import Config, DictAction
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.optimizer import build_optimizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="samples per forward pass; accumulate to reach cfg.batch_size")
    p.add_argument("--batch-size", type=int, default=0, help="effective batch to emulate; 0 = cfg.batch_size")
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-max", type=float, default=1)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--diverge-factor", type=float, default=4.0,
                   help="stop once the smoothed loss exceeds this multiple of its best")
    p.add_argument("--ema", type=float, default=0.8, help="smoothing for the loss curve")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--safety", type=float, default=3.0,
                   help="max_lr = turn-up knee / safety")
    p.add_argument("--from-json", default="",
                   help="re-derive the suggestion from a saved curve instead of sweeping again")
    p.add_argument("--out", default="")
    p.add_argument("--options", nargs="+", action=DictAction)
    return p.parse_args(argv)


def suggest(records, safety=3.0, turn_up=0.05):
    """derive a one-cycle max_lr from the smoothed curve

    Anchored on the turn-up knee -- the first learning rate past the minimum
    where the smoothed loss climbs back above (1 + turn_up) x its best. That is
    the stability boundary, which is what a peak learning rate has to respect,
    and one-cycle then anneals down from `knee / safety`.

    The textbook "steepest descent" point is reported but not used: it is a
    derivative of the raw loss, so it is dominated by wherever the loss happens
    to be largest, and on a curve that falls several units it lands an order of
    magnitude below the range that actually trains best. Both variants are kept
    in the output so the choice stays auditable.
    """
    lr = np.array([r["lr"] for r in records], dtype=float)
    ema = np.array([r["ema"] for r in records], dtype=float)
    ok = np.isfinite(ema)
    lr, ema = lr[ok], ema[ok]
    if lr.size < 5:
        return dict(lr_steepest=None, lr_steepest_rel=None, lr_min_loss=None, lr_knee=None, max_lr=None)
    i_min = int(np.argmin(ema))
    lo = max(2, int(0.1 * lr.size))
    hi = max(lo + 1, i_min + 1)
    i_steep = lo + int(np.argmin(np.gradient(ema, np.log10(lr))[lo:hi]))
    i_steep_rel = lo + int(np.argmin(np.gradient(np.log(ema), np.log10(lr))[lo:hi]))
    after = np.nonzero(ema[i_min:] > (1.0 + turn_up) * ema[i_min])[0]
    knee = float(lr[i_min + after[0]]) if after.size else float(lr[-1])
    return dict(
        lr_steepest=float(lr[i_steep]),
        lr_steepest_rel=float(lr[i_steep_rel]),
        lr_min_loss=float(lr[i_min]),
        lr_knee=knee,
        safety=safety,
        max_lr=knee / safety,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.from_json:
        saved = json.load(open(args.from_json))
        out = dict(saved, **suggest(saved["curve"], args.safety))
        for k in ("lr_steepest", "lr_steepest_rel", "lr_min_loss", "lr_knee", "max_lr"):
            print("%-16s: %s" % (k, "n/a" if out[k] is None else "%.3e" % out[k]))
        json.dump(out, open(args.out or args.from_json, "w"), indent=1)
        print("written to %s" % (args.out or args.from_json))
        return out
    assert torch.cuda.is_available(), "lr_range_test needs a GPU"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    target_bs = args.batch_size or cfg.batch_size
    micro = min(args.micro_batch_size, target_bs)
    accum = max(1, round(target_bs / micro))
    print("effective batch %d = micro %d x accum %d" % (micro * accum, micro, accum), flush=True)

    dataset = build_dataset(cfg.data.train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=micro, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, drop_last=True, persistent_workers=args.num_workers > 0)

    model = build_model(cfg.model).cuda().train()
    optimizer = build_optimizer(cfg.optimizer, model, cfg.get("param_dicts", None))
    base = [g["lr"] for g in optimizer.param_groups]
    ratios = [b / base[0] for b in base]
    print("param groups: %d, lr ratios %s" % (len(base), [round(r, 4) for r in ratios]), flush=True)

    gamma = (args.lr_max / args.lr_min) ** (1.0 / max(1, args.steps - 1))
    records, ema, best = [], None, math.inf
    it = iter(loader)
    for step in range(args.steps):
        lr = args.lr_min * gamma ** step
        for g, r in zip(optimizer.param_groups, ratios):
            g["lr"] = lr * r
        optimizer.zero_grad(set_to_none=True)
        total, n_points = 0.0, 0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss = model(batch)["loss"] / accum
            loss.backward()
            total += float(loss.item())
            n_points += batch["coord"].shape[0]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        optimizer.step()
        ema = total if ema is None else args.ema * ema + (1 - args.ema) * total
        best = min(best, ema)
        records.append(dict(step=step, lr=lr, loss=total, ema=ema, grad_norm=grad_norm, n_points=n_points))
        if step % 10 == 0 or step == args.steps - 1:
            print("step %3d  lr %.3e  loss %.4f  ema %.4f  |g| %.2e" % (step, lr, total, ema, grad_norm), flush=True)
        if not math.isfinite(total) or ema > args.diverge_factor * best:
            print("diverged at lr %.3e (ema %.4f vs best %.4f); stopping" % (lr, ema, best), flush=True)
            break

    out = dict(config=args.config, batch_size=micro * accum, micro_batch_size=micro,
               accum=accum, lr_ratios=ratios, curve=records, **suggest(records, args.safety))
    print("\n--- LR range test ---")
    for k in ("lr_steepest", "lr_steepest_rel", "lr_min_loss", "lr_knee", "max_lr"):
        print("%-16s: %s" % (k, "n/a" if out[k] is None else "%.3e" % out[k]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        print("written to %s" % args.out, flush=True)
    return out


if __name__ == "__main__":
    main()

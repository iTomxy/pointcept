"""Turn tuning results into `--options` overrides for train.sh / test.sh.

Reads the JSON written by `find_batch_size` and `lr_range_test` and prints the
`-o KEY=VALUE` arguments that apply them, so a driver script can do:

    OPTS=$(python tools/tuned_options.py --bs-json bs.json --lr-json lr.json --config cfg.py)
    bash scripts/train.sh ... $OPTS

Keeping this out of the config file means the tuned numbers travel with the run
(they land in the saved config.py) without the checked-in config drifting to
whichever GPU happened to be free that day.

`scheduler.max_lr` is emitted as a list when the config uses per-group learning
rates, preserving the ratios between groups. Note that for OneCycleLR the
scheduler's max_lr governs training; `optimizer.lr` and any `param_dicts` lr
only set the pre-schedule initial value.
"""

import os
import sys

# runnable as `python tools/<name>.py` from anywhere; scripts/*.sh instead export
# PYTHONPATH, so guard against inserting the repo root twice
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse, json, math
import numbers


def _positive_number(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config being tuned; read for scheduler/param_dicts shape")
    p.add_argument("--bs-json", default="", help="output of find_batch_size")
    p.add_argument("--lr-json", default="", help="output of lr_range_test")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="multiply the suggested max_lr, e.g. 0.5 to stay clear of the divergence knee")
    p.add_argument("--prefix", default="-o", help="flag each override is passed with")
    return p.parse_args(argv)


def build_options(config, bs_json="", lr_json="", lr_scale=1.0):
    from pointcept.utils.config import Config
    cfg = Config.fromfile(config)
    options = []

    if bs_json:
        bs = json.load(open(bs_json))
        batch_size = bs["batch_size"]
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        options.append(("batch_size", batch_size))

    if lr_json:
        lr = json.load(open(lr_json))
        max_lr = _positive_number(lr["max_lr"], "suggested max_lr")
        lr_scale = _positive_number(lr_scale, "lr_scale")
        if bs_json and lr.get("batch_size") != batch_size:
            raise ValueError("batch_size and lr batch_size disagree")
        max_lr = _positive_number(max_lr * lr_scale, "scaled learning rate")
        params = cfg.get("param_dicts", None) or []
        ratios = lr.get("lr_ratios")
        if ratios is None and not params:
            ratios = [1.0]
        if not isinstance(ratios, (list, tuple)) or len(ratios) != len(params) + 1:
            raise ValueError("lr_ratios must contain the default group and every param_dict")
        ratios = [_positive_number(r, "LR ratio") for r in ratios]
        if ratios[0] != 1.0:
            raise ValueError("the default group's LR ratio must be 1")
        rates = [_positive_number(max_lr * r, "group learning rate") for r in ratios]
        rounded = [float("%.4g" % r) for r in rates]
        options.append(("optimizer.lr", rounded[0]))
        for i, rate in enumerate(rounded[1:]):
            options.append(("param_dicts.%d.lr" % i, rate))
        sched = cfg.get("scheduler", None)
        if sched is not None and "max_lr" in sched:
            if isinstance(sched["max_lr"], (list, tuple)):
                if len(sched["max_lr"]) != len(rates):
                    raise ValueError(
                        "config has %d max_lr entries but the range test saw %d param groups"
                        % (len(sched["max_lr"]), len(rates)))
            if isinstance(sched["max_lr"], (list, tuple)) or params:
                value = "[%s]" % ",".join("%.4g" % r for r in rounded)
            else:
                value = "%.4g" % rounded[0]
            options.append(("scheduler.max_lr", value))
    return options


def main(argv=None):
    args = parse_args(argv)
    options = build_options(args.config, args.bs_json, args.lr_json, args.lr_scale)
    print(" ".join("%s %s=%s" % (args.prefix, k, v) for k, v in options))


if __name__ == "__main__":
    main()

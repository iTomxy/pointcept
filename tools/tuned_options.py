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

import argparse, json


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
        options.append(("batch_size", bs["batch_size"]))

    if lr_json:
        lr = json.load(open(lr_json))
        max_lr = lr["max_lr"]
        assert max_lr, "lr_range_test produced no suggestion; inspect its curve"
        max_lr *= lr_scale
        options.append(("optimizer.lr", float("%.4g" % max_lr)))
        sched = cfg.get("scheduler", None)
        if sched is not None and "max_lr" in sched:
            ratios = lr.get("lr_ratios") or [1.0]
            if isinstance(sched["max_lr"], (list, tuple)):
                assert len(sched["max_lr"]) == len(ratios), (
                    "config has %d max_lr entries but the range test saw %d param groups"
                    % (len(sched["max_lr"]), len(ratios)))
                value = "[%s]" % ",".join("%.4g" % (max_lr * r) for r in ratios)
            else:
                value = "%.4g" % max_lr
            options.append(("scheduler.max_lr", value))
    return options


def main(argv=None):
    args = parse_args(argv)
    options = build_options(args.config, args.bs_json, args.lr_json, args.lr_scale)
    print(" ".join("%s %s=%s" % (args.prefix, k, v) for k, v in options))


if __name__ == "__main__":
    main()

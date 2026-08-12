# Shared settings for every sequential-tuning step.
#
# Two deliberate differences from the full baseline recipe, so a sweep costs
# hours instead of days (see plans/20260802-tune_ptv3.md):
#
# 1. 50 epochs instead of 100. On the baseline run val mIoU was 0.8413 at epoch
#    50 against 0.8465 at the best epoch (74) -- 0.5 pt lower, but the ranking
#    signal is intact, and a run drops from ~2.5h to ~1.25h. The winning recipe
#    gets re-run at 100 epochs at the end.
# 2. The held-out split is val, not test. Scoring ~20 candidates on the
#    157-volume test split would tune against it and leave the final number
#    meaningless, so test is untouched until the winning recipe.
#
# Steps are compared on `Ribsegv2VolumeTester` output over val, NOT on the
# `Currently Best mIoU` in train.log. The training hook micro-averages over
# grid-subsampled points across batches; the tester scatters back to full
# resolution and averages per volume, and reports the foreground-only dice/IoU
# that the final result is quoted in. Those are different numbers, and only the
# latter matches how the model will be reported.
#
# The learning rate is the one measured for this model/GPU by
# tools/lr_range_test.py (knee 1.716e-2 / 3), pinned here rather than passed as
# an override so every tuning run differs only in the variable under test.

_base_ = ["../semseg-pt_v3m1_0_base.py"]

epoch = 50
eval_epoch = epoch

# Evaluate on the validation split. `add_trainval_incomplete` MUST be off here:
# it unions in every incomplete volume from train and test alike, which would
# put training volumes into the set the tuning decisions are made on.
data = dict(
    test=dict(
        split="val",
        add_trainval_incomplete=False,
    ),
)

test = dict(save_pred=False, save_cm=False)

optimizer = dict(type="AdamW", lr=0.005719, weight_decay=5e-4)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005719, 0.0005719],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

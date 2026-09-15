# Method 1: rib integrity with contrastive supervision

Improve label agreement along each rib without sacrificing rib-index correctness
or foreground/background accuracy. The implementation is complete; the next
agent will execute, monitor and record the **λ = 1, 5, 10** sweep below.
Baseline and λ = 0.1 results are already available.
The executing agent should follow [Steps](#steps) and update its checklist.

Based on the sibling plan [~/codes/pointnext-lightning/plans/20260909-xxli_integrity_method_1.md](../../pointnext-lightning/plans/20260909-xxli_integrity_method_1.md)
and the [canonical PTv3 recipe](../configs/ribsegv2/semseg-pt_v3m1_0_base.py).

## Performance

Each metric cell is **val / test** (73 / 157 volumes). IoU, recall and precision
are foreground class means within each volume, then means across volumes,
reported as percentages. `λ` is `model.contrastive.loss_weight`; baseline is 0.
`Selected epoch` is the checkpoint epoch. Pending rows contain no measurements.

**Purity median/max/min and the low-purity count use individual (volume, rib)
pairs with `fg_purity_cw < 0.90`, filtering unrounded values before aggregation.**
This selects ribs with more than 10% of recognized foreground points outside
the dominant predicted label. Null values are excluded. Purity is displayed in
percent; pair counts are integers. `—` means unmeasured or an empty subset;
a measured empty subset has count **0**. Read the count alongside the conditional
statistics: the qualifying pairs may differ between models.

The two **fg frag** columns cover **all scored pairs**, independently of the
90% filter. A fragment is a predicted label receiving at least **5%** of a GT
rib's recognized foreground points. `mean` averages that label count; `>1` is
the percentage of pairs with at least two such labels. An 80% / 20% label split
has purity 80% and count 2. These measure label splitting, not spatial connected
components. All-background predictions are unscored; a rib assigned entirely
to one wrong label still has purity 100% and count 1. Keep overlap and coverage
in the assessment. Very diffuse predictions can have zero labels above 5%.

| Arm / evidence | λ | Selected epoch | IoU | Recall | Precision | fg purity median | fg purity max | fg purity min | fg purity <90% pairs | fg frag mean | fg frag >1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Baseline](../exp/ribsegv2/ptv3-integrity-baseline-eval/) | 0 | 78 | 87.76 / 87.68 | 92.78 / 92.23 | 94.09 / 94.29 | 77.97 / 72.12 | 85.59 / 89.33 | 50.24 / 46.06 | 20 / 41 | 1.01 / 1.01 | 1.29 / 1.35 |
| [Method 1, seed 1](../exp/ribsegv2/ptv3-integrity-m1-s1/) | 0.1 | 76 | 87.88 / 87.15 | 92.86 / 91.46 | 94.14 / 94.34 | 74.30 / 78.24 | 88.77 / 89.63 | 50.36 / 38.63 | 14 / 61 | 1.01 / 1.02 | 0.88 / 2.16 |
| [Method 1, seed 1](../exp/ribsegv2/ptv3-integrity-m1-w1-s1/) | 1 | 99 | 85.10 / — | 90.79 / — | 91.56 / — | 78.72 / — | 89.90 / — | 50.00 / — | 55 / — | 1.05 / — | 4.86 / — |
| [Method 1, seed 1](../exp/ribsegv2/ptv3-integrity-m1-w5-s1/) | 5 | 96 | 79.21 / — | 86.66 / — | 87.20 / — | 76.09 / — | 89.81 / — | 50.37 / — | 231 / — | 1.20 / — | 19.28 / — |
| [Method 1, seed 1](../exp/ribsegv2/ptv3-integrity-m1-w10-s1/) | 10 | 100 | 64.30 / — | 76.05 / — | 76.30 / — | 73.39 / — | 90.00 / — | 38.20 / — | 784 / — | 1.67 / — | 53.93 / — |

Completed rows were measured on 2026-09-11/12/13/15 with evaluation seed **20260909**.
Both used total batch 4, base LR 0.005102 and block LR 0.0005353; these describe
those runs, not requirements for new runs. Baseline evaluates
`exp/ribsegv2/final/model/model_best.pth`; method uses its own best checkpoint.
Full provenance is in the linked evaluation summaries and saved configs.

**Current result:** λ = 0.1 improved validation low-purity counts (20 → 14),
but test worsened (41 → 61), with IoU −0.53 percentage points and fragmentation
1.35% → 2.16%. It is not adopted. The λ = 1/5/10 sweep finished 2026-09-15:
all three are val-negative with degradation growing in λ (val IoU 85.10 →
79.21 → 64.30; low-purity pairs 55 → 231 → 784). **Baseline retained; no test
evaluation run.** No matched-seed repeats have been run.

## Method and experiment controls

```text
L_total = L_CE + L_Lovasz + λ * L_contrastive
s(a,b) = dot(normalize(z_a), normalize(z_b)) / 0.1
L_a = mean over positives p of:
      logsumexp([s(a,p)] + [s(a,n) for n in negatives]) - s(a,p)
```

The [method config](../configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py)
inherits the canonical model and data pipeline. During training, final decoder
features feed a **64 → 128 → 64** projection with LayerNorm/GELU. Within each
volume, sample up to 4 anchors per rib; use up to 2 farthest same-rib positives
and 16 nearest different-rib negatives. Background supplies no pairs. Average
losses over positives, anchors, ribs and clouds. Mining is streamed and only
selected features are projected. Inference uses the existing classifier and
full-resolution inverse scatter, with no projection or mining.

Keep the following controls for this λ sweep:

- Train seed **1**, 100 epochs, `eval_epoch=epoch`; evaluate with seed **20260909**.
- Preserve canonical augmentation, 3 mm grid, radius 259.16, 200,000-point
  training cap, windowed HU, fp32/no flash, attention windows and loss/mining
  settings. Keep `mix_prob=0` and `add_trainval_incomplete=False`.
- **Select batch size for each run's allocated GPUs, then tune LR at that batch.**
  Keep AdamW weight decay 5e-4 and the OneCycle schedule shape. Scale the block
  LR with the tuned base LR using the reference ratio `0.0005353 / 0.005102`;
  update both optimizer group LRs and both scheduler maxima together.
- Use comparable tuning budgets and full-resolution **validation** to select λ
  and LR. Keep held-out test out of tuning. Since batch/LR may differ, this
  compares independently tuned recipes rather than isolating λ alone.

The initial 0.1 was not calibrated. In its logged epochs 51–100, mean weighted
contrastive loss was 0.00142 versus segmentation loss 0.18190. Larger weights
probe whether stronger supervision helps; scalar loss ratios do not establish
relative backbone-gradient influence.

## Steps

Follow this checklist in order. Check completed actions and append a short
result/date with an artifact link; record failures beside unchecked actions.
Update each performance row as work progresses. Leave test cells blank until
the validation decision in item 5.

1. [x] [Verify the environment and pass the checks](#environment-and-checks).
   Done 2026-09-12 on venus4 (2× A5500, idle): focused 8-file suite **94 passed**.
   Deviation: `tests/test_env.py --require-gpu` fails in the available old image
   (`~/pointcept.sif`, torch 2.5.0+cu124 — no `SharedArray`; the script pins the
   newer cu128 image, absent here). All workload gates below ran in this image.
2. [x] Complete **λ = 1**, run `ptv3-integrity-m1-w1-s1` — val-negative
   (IoU 85.10, 55 low-purity pairs vs baseline 20); test cells blank.

   - [x] [Select batch size and tune LR](#batch-and-lr-tuning). 2026-09-12:
     batch probe → 2/GPU (total 4); LR test min-loss 0.0489, accepted max_lr
     **0.02444** / block **0.002564** (flat valley, ~10× below explosion).
     Artifacts: `exp/ribsegv2/ptv3-integrity-m1-w1-s1/tuning/`.
   - [x] [Pass preflight and smoke checks](#preflight-and-smoke). 2026-09-12:
     degenerate-rank DDP passed; smoke fit exit 0 (aux finite/nonzero all 212
     iters, mean 0.78); smoke uncapped val eval passed. Exception: 200k×2
     synthetic preflight OOMs at ~23.5 GB (genuine capacity; real-data smoke is
     the operative memory gate). Details in `exp/ribsegv2/ptv3-integrity-m1-w1-s1/run.md`.
   - [x] [Train 100 epochs and evaluate validation](#training-and-validation).
     Fit exit 0 on venus4 2×A5500 (hook-best epoch 99); val eval seed 20260909.
   - [x] [Record validation metrics and artifacts](#recording-and-comparison).
     Val row filled 2026-09-12; extractor-verified against the two reference arms.

3. [x] Complete **λ = 5**, run `ptv3-integrity-m1-w5-s1` — val-negative
   (IoU 79.21, 231 low-purity pairs vs baseline 20); test cells blank.

   - [x] [Select batch size and tune LR](#batch-and-lr-tuning). Batch 4 (2/GPU);
     accepted max_lr **0.0183** / block **0.00192**. Artifacts: `tuning/`.
   - [x] [Pass preflight and smoke checks](#preflight-and-smoke). New-run 200k×2
     synthetic skipped (λ-independent OOM at ~23.5 GB, recorded under λ=1);
     degenerate-rank DDP passed; smoke fit exit 0 (aux nonzero on all 212 iters,
     mean 3.26); smoke uncapped val eval passed.
   - [x] [Train 100 epochs and evaluate validation](#training-and-validation).
     Fit exit 0 on venus4 2×A5500 (hook-best epoch 96); val eval seed 20260909.
   - [x] [Record validation metrics and artifacts](#recording-and-comparison).
     Val row filled 2026-09-13; extractor-verified against the reference arms.

4. [x] Complete **λ = 10**, run `ptv3-integrity-m1-w10-s1` (on venus4) —
   val-negative (IoU 64.30, 784 low-purity pairs vs baseline 20); test cells blank.

    - [x] [Select batch size and tune LR](#batch-and-lr-tuning). Re-probed on
      venus4 after the mars11 tuning (L4) was discarded: batch 4 (2/GPU); LR
      test min-loss 0.0387, knee 0.0549, accepted max_lr **0.0183** / block
      **0.00192** (same stable valley as the λ=5 tune on this node).
    - [x] [Pass preflight and smoke checks](#preflight-and-smoke). 2026-09-13:
      degenerate-rank DDP passed; smoke fit exit 0 (aux finite/nonzero all 212
      iters, mean 6.85); smoke uncapped val eval passed. Earlier mars11 attempt
      OOM'd in the smoke fit (L4 exposes 22.05 GiB vs A5500 23.56; batch 4 does
      not fit real data there), so the arm moved to venus4 with fresh tuning.
      Details in `exp/ribsegv2/ptv3-integrity-m1-w10-s1/run.md`.
    - [x] [Train 100 epochs and evaluate validation](#training-and-validation).
      Fit exit 0 on venus4 2×A5500 (hook-best epoch 100, train mIoU 0.6548);
      val eval seed 20260909.
    - [x] [Record validation metrics and artifacts](#recording-and-comparison).
      Val row filled 2026-09-15; extractor-verified against the reference arms.

5. [x] [Compare validation results](#recording-and-comparison). 2026-09-15:
   **retain the baseline; no candidate selected.** λ = 1 (IoU 85.10, 55 pairs),
   λ = 5 (79.21, 231 pairs) and λ = 10 (64.30, 784 pairs, frag>1 53.93%) are
   all val-negative against baseline (87.76, 20 pairs) on overlap, purity and
   fragmentation; λ = 0.1 was val-marginal but test-worse. Degradation grows
   monotonically with λ. No promising candidate, so no matched-seed
   confirmation runs.
6. [x] ~~[Evaluate the selected candidate on test](#final-test-evaluation)~~ —
   skipped: baseline retained after a negative sweep, per the plan's
   "retain the baseline and record the negative sweep" clause. Test cells stay `—`.

### Environment and checks

Use the allocated GPUs in the [Pointcept container](env.md), with only those
devices visible. Run these Bash blocks from the repository root in the same
shell session. Recheck GPU availability before each new arm.

```bash
set -euo pipefail
python tests/test_env.py --require-gpu --skip-visualization
python -m pytest -q -rs tests/test_instance_contrastive_loss.py \
  tests/test_contrastive_segmentor.py tests/test_contrastive_preflight.py \
  tests/test_tuned_options.py tests/test_eval_cm_integrity.py \
  tests/test_ribsegv2_integrity_reporting.py tests/test_integrity_report.py \
  tests/test_extract_fg_purity.py
```

### Batch and LR tuning

Change only `INTEGRITY_LAMBDA` below to 1, 5 or 10; the run name follows it.
Use a fresh directory for a new attempt. When resuming an unfinished run,
reuse its own tuning files only after confirming the same config/allocation;
otherwise use a new suffix and update the row link.

```bash
INTEGRITY_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
test "$INTEGRITY_GPUS" -ge 1 || exit 1
INTEGRITY_CONFIG=configs/ribsegv2/+contrastive/semseg-pt_v3m1_0_base.py
INTEGRITY_LAMBDA=1
INTEGRITY_RUN=ptv3-integrity-m1-w${INTEGRITY_LAMBDA}-s1
INTEGRITY_RUN_DIR=exp/ribsegv2/$INTEGRITY_RUN
INTEGRITY_TUNE_DIR=$INTEGRITY_RUN_DIR/tuning
test ! -e "$INTEGRITY_RUN_DIR" || exit 1
mkdir -p "$INTEGRITY_TUNE_DIR"
nvidia-smi > "$INTEGRITY_TUNE_DIR/gpus.txt"
git rev-parse HEAD > "$INTEGRITY_TUNE_DIR/revision.txt"
git status --short > "$INTEGRITY_TUNE_DIR/worktree-status.txt"

python tools/find_batch_size.py \
  --config "$INTEGRITY_CONFIG" --num-gpus "$INTEGRITY_GPUS" --mem-frac 0.9 \
  --out "$INTEGRITY_TUNE_DIR/batch.json" \
  --options model.contrastive.loss_weight="$INTEGRITY_LAMBDA" \
  > "$INTEGRITY_TUNE_DIR/batch.log" 2>&1

INTEGRITY_BATCH=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' \
  "$INTEGRITY_TUNE_DIR/batch.json")
INTEGRITY_MICRO_BATCH=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["batch_size_per_gpu"])' \
  "$INTEGRITY_TUNE_DIR/batch.json")

python tools/lr_range_test.py \
  --config "$INTEGRITY_CONFIG" --batch-size "$INTEGRITY_BATCH" \
  --micro-batch-size "$INTEGRITY_MICRO_BATCH" --seed 1 \
  --out "$INTEGRITY_TUNE_DIR/lr.json" \
  --options model.contrastive.loss_weight="$INTEGRITY_LAMBDA" \
  > "$INTEGRITY_TUNE_DIR/lr.log" 2>&1

python tools/tuned_options.py --config "$INTEGRITY_CONFIG" \
  --bs-json "$INTEGRITY_TUNE_DIR/batch.json" \
  --lr-json "$INTEGRITY_TUNE_DIR/lr.json" --prefix '' \
  > "$INTEGRITY_TUNE_DIR/options.txt"
read -r -a INTEGRITY_TUNED_OPTIONS < "$INTEGRITY_TUNE_DIR/options.txt"
```

Inspect the LR curve before accepting its suggestion. The batch probe measures
one device; on heterogeneous GPUs, probe the most constrained device first.
The LR search uses one GPU with accumulation to emulate the total batch.
Confirm both decisions in [preflight and smoke](#preflight-and-smoke), including the largest
capped batch and uncapped volume evaluation. If allocation changes, reconsider
batch size and retune LR.

### Preflight and smoke

```bash
torchrun --standalone --nproc_per_node="$INTEGRITY_GPUS" \
  tools/ribsegv2/contrastive_preflight.py --config "$INTEGRITY_CONFIG" \
  --points-per-cloud 200000 --clouds-per-rank "$INTEGRITY_MICRO_BATCH" \
  --options "${INTEGRITY_TUNED_OPTIONS[@]}" \
  model.contrastive.loss_weight="$INTEGRITY_LAMBDA" \
  2>&1 | tee "$INTEGRITY_TUNE_DIR/preflight.log"

if [ "$INTEGRITY_GPUS" -ge 2 ]; then
  torchrun --standalone --nproc_per_node="$INTEGRITY_GPUS" \
    tools/ribsegv2/contrastive_preflight.py --points-per-cloud 4096 \
    --clouds-per-rank 1 --degenerate-rank 0 \
    --options "${INTEGRITY_TUNED_OPTIONS[@]}" \
    model.contrastive.loss_weight="$INTEGRITY_LAMBDA" \
    2>&1 | tee "$INTEGRITY_TUNE_DIR/preflight-ddp.log"
fi

sh scripts/train.sh -g "$INTEGRITY_GPUS" -d ribsegv2 \
  -c +contrastive/semseg-pt_v3m1_0_base -n "$INTEGRITY_RUN-smoke" \
  -o "${INTEGRITY_TUNED_OPTIONS[*]} model.contrastive.loss_weight=$INTEGRITY_LAMBDA seed=1 epoch=2 eval_epoch=2"

sh scripts/test.sh -g "$INTEGRITY_GPUS" -d ribsegv2 \
  -n "$INTEGRITY_RUN-smoke" -w model_best \
  -o "data.test.split=val seed=20260909 test.save_cm=True"
```

Proceed only after finite losses/gradients, active eligible pairs, correct LR
groups, and adequate memory are confirmed. Log weighted contrastive and
segmentation losses, pair counts, peak VRAM and step time. The smoke's val run
checks uncapped evaluation memory; its metrics do not belong in the table.
If a gate fails, record command/config/error and diagnose it before the full fit.

### Training and validation

Start a fresh 100-epoch fit using this arm's selected settings; do not resume
from its smoke checkpoint.

```bash
sh scripts/train.sh -g "$INTEGRITY_GPUS" -d ribsegv2 \
  -c +contrastive/semseg-pt_v3m1_0_base -n "$INTEGRITY_RUN" \
  -o "${INTEGRITY_TUNED_OPTIONS[*]} model.contrastive.loss_weight=$INTEGRITY_LAMBDA seed=1"

sh scripts/test.sh -g "$INTEGRITY_GPUS" -d ribsegv2 \
  -n "$INTEGRITY_RUN" -w model_best \
  -o "data.test.split=val seed=20260909 test.save_cm=True"
```

Monitor each fit to completion and inspect losses, LR traces, pair counts and
memory. Keep the hook-selected `model_best.pth`; its grid-level mIoU is a
checkpoint-selection metric, not the table's volume-wise foreground IoU.
`scripts/train.sh` saves the resolved config and code snapshot; testing uses
the current checkout with that saved config. Record both revisions and any
uncommitted changes in `$INTEGRITY_RUN_DIR/run.md`, alongside GPU allocation,
chosen batch/LRs, commands, runtime and checkpoint epoch.

### Recording and comparison

Use [extracting fg purity results](#extracting-fg-purity-results) below to fill
the val purity/count cells without requiring test output. For the other cells,
read the matching `val-Ribsegv2VolumeTester.json`:

| Table field | JSON source |
| --- | --- |
| Selected epoch | `epoch` |
| IoU / Recall / Precision | `metrics.iou_fg` / `metrics.recall_fg` / `metrics.precision_fg`, each ×100 |
| fg frag mean | `metrics.fg_fragment_count_5pct_mean` |
| fg frag >1 | `metrics.fg_fragment_gt1_fraction` ×100 |

Round displayed values to two decimals, retain raw reports, and leave test
cells `—`. Confirm 73 volumes, evaluation seed 20260909 and the intended λ in
the saved config. Also inspect coverage, absent/unrecognized-pair counts,
`fg_recall_cw` and the worst volume/rib pairs; purity alone can reward missed
foreground or coherent wrong-index predictions. Spatial diagnosis is optional
via [integrity_report.py](../tools/ribsegv2/integrity_report.py).

After all three arms have validation results, rank lower low-purity counts
alongside IoU, recall, coverage and fragmentation. The prior overlap-metric
noise floor was about 0.40 percentage points; it is not an uncertainty estimate
for purity counts. Record the choice and its rationale here. If a candidate
looks promising, confirm it and the baseline with matched training seeds,
using fresh names/rows and the same per-run tuning policy.

### Final test evaluation

Only evaluate a selected candidate after recording the validation decision;
do not evaluate every trial on test to choose λ. If none improves the overall
validation trade-off, retain the baseline and record the negative sweep.
Set `INTEGRITY_RUN` and `INTEGRITY_GPUS` to the selected run and current
allocated evaluation GPUs, then run:

```bash
sh scripts/test.sh -g "$INTEGRITY_GPUS" -d ribsegv2 \
  -n "$INTEGRITY_RUN" -w model_best \
  -o "data.test.split=test seed=20260909 test.save_cm=True"
```

Verify 157 volumes, rerun extraction for both splits, and fill that row's test
cells. The extractor requires the same checkpoint/epoch and evaluation seed
for val and test. A new baseline fit must use its own saved config/checkpoint;
the completed baseline row remains historical evidence.

## extracting fg purity results

[extract_fg_purity.py](../tools/ribsegv2/extract_fg_purity.py) needs only Python's
standard library. During the sweep, compare the current arm against the two
completed references using **validation only**:

```bash
python tools/ribsegv2/extract_fg_purity.py \
  exp/ribsegv2/ptv3-integrity-baseline-eval \
  exp/ribsegv2/ptv3-integrity-m1-s1 \
  "exp/ribsegv2/$INTEGRITY_RUN" --threshold 0.90 --splits val
```

Copy its median/max/min and count cells directly. Output always uses
**val / test**; unrequested splits print `—`, never a fabricated count of zero.
After step 6, omit `--splits val` to extract both splits. Add `--format json`
to retain an audit record of the unrounded statistics and every qualifying
`(volume, rib, purity)` pair; JSON purity values are fractions.

The script validates the latest JSONL block against the summary timestamp and
volume count, rejects duplicate/invalid records, and requires matching volume
IDs and evaluation seeds across compared arms. It emits no partial table on
failure. Keep the 0.90 threshold and 73/157 volume-count defaults throughout
this sweep. A qualifying value may round to `90.00` when displayed.

## Implementation and verification

Core code: [segmentor](../pointcept/models/contrastive.py),
[loss/mining](../pointcept/models/losses/contrastive.py),
[metrics](../pointcept/utils/eval_cm.py),
[tester](../pointcept/engines/test.py),
[tuning overrides](../tools/tuned_options.py), and
[CUDA preflight](../tools/ribsegv2/contrastive_preflight.py).

Local verification on 2026-09-12: **32 tuning/preflight/loss tests** and
**51 reporting/extraction tests** passed, including validation-only extraction.
The Bash blocks and historical table values were checked. Broader segmentor/CUDA
checks require the runner container; this host lacks `torch_scatter`.
Execute step 1 in that environment before the sweep.

## Method 1b — deferred

Method 1b would use embeddings for inference-time assignment, such as nearest
rib-class prototypes, and is a separate experiment. PTv3 already processes
whole volumes. Within-volume contrastive training does not explicitly align
class embeddings across patients; prototype identities must be established
without test labels. Embedding export/assignment is not implemented and would
not by itself guarantee spatial connectivity. Finish the λ sweep first.

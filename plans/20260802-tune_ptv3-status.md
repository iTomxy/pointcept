# Tuning status — handoff (as of 10 Aug 2026)

Companion to [20260802-tune_ptv3.md](20260802-tune_ptv3.md), which now carries a
generated `# Results` table — one row per run, with config link, log path, the
batch/lr that actually trained, and the val metrics. Regenerate it with
`python3 tmp-ai-results_table.py` after every step; the runners call it
automatically.

**Everything is finished.** The tuned recipe is trained at 100 epochs and the
single test-split pass is done: `exp/ribsegv2/final/`. All state is on shared
storage (`exp/`, configs, tuning JSONs). `/tmp` scratch is node-local and does
NOT survive a node switch — nothing important lives there.

**The recipe now lives in
[configs/ribsegv2/semseg-pt_v3m1_0_base.py](../configs/ribsegv2/semseg-pt_v3m1_0_base.py)**,
flattened and standalone (no `_base_`; the generic runtime block is inlined).
Verified to resolve byte-identically to `exp/ribsegv2/final/config.py` —
train/val/test transforms, model, optimizer, scheduler, param_dicts, hooks — so
running it reproduces the result below. `configs/ribsegv2/seq-tune/` is kept as
the record of how each setting was chosen, and every tuning config there
overrides its own pipeline, so none of them changed behaviour when the base did.

Two intentional differences from the saved `final/config.py`: `weight=None` (that
file has the checkpoint path `scripts/test.sh` wrote into it) and `save_cm=True`
(the tuning override had turned it off, so the final test pass did not write a
confusion-matrix `.npy`/`.png` — the `fg_confusion` summary and per-volume JSONL
are in the result JSON regardless).

## RESULT: test dice_fg 0.8819 -> 0.9264, +4.44 pt

157 held-out volumes, read once, after every decision was made on val.

| metric | baseline | tuned | delta |
| --- | --- | --- | --- |
| dice_fg | 0.8819 | **0.9264** | **+4.44** |
| iou_fg | 0.8289 | 0.8796 | +5.07 |
| precision_fg | 0.9010 | 0.9414 | +4.04 |
| recall_fg | 0.8770 | 0.9214 | +4.44 |
| label_acc | 0.9097 | 0.9611 | +5.15 |
| dice (incl. bg) | 0.8860 | 0.9287 | +4.27 |
| iou (incl. bg) | 0.8345 | 0.8833 | +4.88 |

Mislabelled rib points: **1,895,622 -> 206,294, an 89% reduction.** Still 95%
off-by-one and 0% cross-side, so the residual error is the same kind, just far
less of it.

Two sanity checks passed:
- **val 0.9306 vs test 0.9264** — a 0.42 pt gap, about the noise floor. ~35 runs
  of selection on val did not overfit it.
- **100 epochs bought +0.05 pt on val over 50**, so the 50-epoch shortcut used
  for every tuning run cost nothing. The original justification (0.5 pt, from
  the baseline) was conservative — augmentation appears to do the regularising
  the longer schedule was compensating for.

**Read the lr section below before trusting any delta recorded before 10 Aug.**

## Standing best recipe

**`augc-all5`** — val dice_fg **0.9336** (+2.96 pt over the un-augmented 0.9041)

    grid 3mm | enc_patch_size (128,128,256,512,1024) | enable_flash=False (fp32)
    features whu1 (single 200-1500 HU window, 1 ch) | batch 4 | 50 epochs
    augmentation: RandomTruncateRibPoint + CTIntensityVariation(shift)
                  + RandomRotate(z) + RandomRotate(y) + RandomScale, in sequence

## Decision rule

Judged on `Ribsegv2VolumeTester` over **val** (73 volumes, full-res, per-volume,
foreground-only), not the train.log hook. **Noise floor 0.40 pt**, measured as
the range over 4 seeds of one identical config. Treat < 0.4 pt as a tie.

## Steps 1-3: DONE (re-tested at matched lr — see the lr section for what moved)

**Step 1 — input features: no change.** All four HU representations within
0.33 pt (< noise). Position as a feature gave no upside. Keep `whu1`.
Likely because preprocessing already thresholds at HU > 200, so the
bone/background decision is made before the model sees anything.

**Step 2 — augmentation singles** (delta vs no-aug 0.9041):

| aug | delta | | aug | delta |
| --- | --- | --- | --- | --- |
| scale | +1.91 | | hushift | +0.59 |
| roty | +1.49 | | hujitter | -0.03 (tie) |
| truncrib | +1.26 | | droprib | **-0.58 (hurts)** |
| rotz | +0.60 | | | |

`droprib` deletes whole rib pairs mid-cage — an anatomy absent from the data —
while `truncrib` cuts along z like a real partial field of view. `jitter` was
excluded by design: PTv3 keys off `grid_coord`, so sub-cell displacement is
invisible to attention and only inflates the point count (5-54%).

`rotz`'s +0.60 is understated — it drew an lr over the cliff (see below).

**Step 2b — combinations**, all at matched lr after the re-runs:
`all5` **0.9336** > `scale` 0.9269 > `all5alt` 0.9258 > `top3` 0.9230.
**`all5` wins**, by 0.67 pt over the runner-up. Stacking the top-3 singles
bought nothing over `scale` alone, while adding the two weakest positives did.
Sequential rotations beat `RandomApply` choose-one by **0.79 pt** — the 2.7 pt
originally recorded was mostly lr artefact.

**Step 3 — grid x attention x precision**, all at matched lr after the re-runs:

| grid | windows | precision | val dice_fg | vs `all5` |
| --- | --- | --- | --- | --- |
| 3mm | reduced | **fp32** | **0.9336** | — |
| 4mm | full | bf16 | 0.9314 | −0.23 (tie) |
| 3mm | full | bf16 | 0.9290 | −0.46 |
| 4mm | reduced | fp32 | 0.9280 | −0.56 |
| 3mm | reduced | bf16 | 0.9238 | −0.98 |

Each factor in isolation: **bf16 −0.98, full windows +0.52, 4mm −0.56.** So:
- **keep 3mm** (one-variable fp32/reduced comparison: coarsening costs 0.56 pt)
- **keep fp32** (`enable_flash=False`); corroborates `enable_amp=False`
- **`enc_patch_size` is NOT irrelevant** — larger windows are worth +0.52 pt at
  matched precision. The original −0.02 "tie" was two opposing errors
  cancelling. It changes nothing in practice: fp32 forces reduced windows (no
  flash ⇒ the attention matrix is materialised ⇒ memory linear in patch size ⇒
  OOM at 8×), and +0.52 never covers the −0.98 that flash costs to get there.
- the factors are still not additive — stacking all three predicts −1.02 and
  measures −0.23 — but the earlier claim that each *flips sign* depending on the
  other was itself an lr artefact.
- 2.5mm excluded on measurement: 96% of a 22GB card AND stage4_max 1042 > 1024,
  which loses the single global attention window at the deepest stage.

## The lr range test is not stable — read this before trusting a delta

Found 10 Aug while setting up step 4. `tools/lr_range_test.py` was treated as
producing a per-arm optimum, so every run from step 2 onward drew its own lr.
It does not. Two configs that resolve **byte-identical** (`augc-all5` and the
first `opt-ref` attempt) got **0.0051 and 0.0115 — 2.25×**. Their loss curves
agree to three decimals up to lr 0.012, then one hits a transient spike
(grad-norm 3–5, loss 1.64) and never recovers its minimum while the other keeps
improving to 0.027. `lr_min_loss` therefore lands on whichever side of the spike
that single noisy pass happened to fall. Both curves agree on where training
actually blows up (~0.08); the heuristic just isn't reading that.

Across steps 2–3 the drawn lr spanned **4×** (0.0045–0.0183) between arms that
were supposed to differ only in the factor under test. Steps 1 and 1b are
unaffected — they pin lr at 0.005719 for every run.

Fix: `FIXED_LR` in `tmp-ai-tune_hp.sh`, the exact counterpart of `FIXED_BS`, and
**both are now mandatory for any comparison sweep**.

### What it costs: −2.22 pt

Measured, not estimated. `opt-ref` and `opt-lr2x` are the same recipe at
0.005102 and 0.0115; nothing else differs:

| run | lr | val dice_fg | iou_fg | label_acc |
| --- | --- | --- | --- | --- |
| `opt-ref` | 0.005102 | 0.9300 | 0.8823 | 0.9665 |
| `opt-lr2x` | 0.0115 | 0.9078 | 0.8579 | 0.9421 |

**−2.22 pt from the lr draw alone** — larger than every augmentation effect in
step 2, and 5.5x the noise floor.

`opt-ref` also reproduced `augc-all5` (0.9300 vs 0.9336, −0.36 pt, inside the
floor), so the A5500 and the L4 agree and no node correction is needed.

### Which conclusions this breaks

Every confounded arm drew a **higher** lr than the reference it lost to, so
every one was penalised and its true score is ≥ what was recorded. Nothing was
unfairly flattered — only losers need re-running.

| arm | lr vs ref | Δ recorded | status |
| --- | --- | --- | --- |
| `augc-all5alt` | 2.68x | −2.67 | **confounded** — the gap ≈ the lr penalty |
| `aug-rotz` | 2.39x | +0.60 | confounded, but rotz is already in `all5` |
| `grid-g3p1024` | 2.12x | −1.00 | **confounded** — may actually be a win |
| `grid-g4p1024` | 2.01x | −0.26 | **confounded** |
| `augc-top3` | 1.89x | −0.98 | **confounded** — likely a tie with `all5` |
| `grid-g4p128nf` | 1.79x | −0.60 | mild |
| `aug-truncrib` | 1.42x | +1.25 | mild; already in `all5` |
| everything else | ≤1.30x | | clean |

**Step 2b is now resolved by re-runs.** At matched lr: `all5` 0.9336 > `scale`
0.9269 > `all5alt` 0.9258 > `top3` 0.9230. `all5` is still the winner, by 0.67
pt over the runner-up. But *"sequential rotations beat `RandomApply` by 2.7 pt"*
was wrong — **the real margin is 0.79 pt**, and ~70% of the original claim was
lr artefact. *"Stacking the top-3 singles bought nothing"* survives unchanged.

**Step 3 is now resolved too.** *"Keep 3mm"* and *"keep fp32"* are restored on
the one-variable fp32/reduced comparison; *"`enc_patch_size` is irrelevant"* is
not — larger windows are worth +0.52 pt. Still standing untouched: all of step 1
(lr pinned throughout) and the step-2 singles except `rotz`/`truncrib`.

**Net effect of the whole audit on the recipe: none.** `all5` at 3mm / reduced
windows / fp32 was the standing recipe before and remains it after. What changed
is the *reasons* — three recorded conclusions were wrong, and the margins on two
others shrank to roughly twice the noise floor.

### The penalty is a threshold, not a slope

Three calibration points now:

| lr | cost vs the same recipe at 0.005102 |
| --- | --- |
| 0.009653 (`top3`) | −0.09 pt — harmless |
| 0.0115 (`lr2x`) | −2.22 pt |
| 0.01368 (`all5alt`) | −1.88 pt |

So the damage is a **cliff between 0.0097 and 0.0115**, not a gradient in the
ratio — which matches the range-test curve, whose loss spike sits at lr
0.013–0.019. Re-reading the audit against absolute lr rather than ratio:

| arm | lr | verdict |
| --- | --- | --- |
| `augc-all5alt` | 0.01368 | above the cliff — **re-run, recovered +1.88 pt** |
| `aug-rotz` | 0.01368 | above; already in `all5`, so it changes no recipe — but see below |
| `grid-g3p1024` | 0.01084 | above — re-test justified (running) |
| `grid-g4p1024` | 0.01024 | at the edge — re-test justified |
| `grid-g4p128nf` | 0.009117 | below, and below `top3`'s harmless 0.009653 |
| `augc-top3` | 0.009653 | below — **re-run, moved −0.09; conclusion stands** |
| `aug-truncrib` | 0.008118 | below — its +1.25 is real |

`g4p128nf` is kept in the queue anyway: two data points do not locate a cliff
precisely, and it is the 4mm-fp32 cell the grid decision turns on. 2h is cheap
against a wrong recipe propagating into everything downstream. (`g4p1024` at
0.01024 then came back +0.03 — harmless — narrowing the cliff onset to between
0.01024 and 0.01084.)

**Optional, low priority: re-run `aug-rotz` at the pinned lr.** It drew 0.01368,
well above the cliff, so its recorded +0.60 understates it — possibly by ~2 pt.
This changes no recipe (`rotz` is already in `all5`), but it does undercut one
of the two "surprises" recorded for step 2b: *"adding the two weakest positives
bought what stacking the top-3 did not"* rests on `rotz` being weak. If `rotz`
is actually a strong single, that surprise dissolves and `all5` is simply "the
strong ones". Worth 2h after step 4 if the node is still free; it is a
narrative correction, not a decision.

## DONE: lr-pinned re-runs (5 arms)

`tmp-ai-lrpin.sh`, tmux session `lrpin`, log `tmp-ai-lrpin.out`. Results go to
new dirs `<prefix>-<mode>-lrpin`; the originals stay as the record of what the
unpinned tool produced.

All five landed. **The recipe is unchanged: `all5` at 3mm, reduced windows,
fp32.** Nothing beat it, so `tmp-ai-after_lrpin.sh` returned GO and step 4
started at 05:18.

| candidate (matched lr) | val dice_fg | vs `all5` |
| --- | --- | --- |
| `augc-all5` — 3mm, reduced, fp32 | **0.9336** | — |
| `grid-g4p1024-lrpin` — 4mm, full, bf16 | 0.9314 | −0.23 (tie) |
| `grid-g3p1024-lrpin` — 3mm, full, bf16 | 0.9290 | −0.46 |
| `grid-g4p128nf-lrpin` — 4mm, reduced, fp32 | 0.9280 | −0.56 |
| `augc-scale` | 0.9269 | −0.67 |
| `augc-all5alt-lrpin` | 0.9258 | −0.79 |
| `grid-g3p128` — 3mm, reduced, bf16 | 0.9238 | −0.98 |
| `augc-top3-lrpin` | 0.9230 | −1.07 |

Step 3 resolved, each factor now clean in isolation: **bf16 costs 0.98 pt,
full windows gain 0.52 pt, 4mm costs 0.56 pt.** *"Keep 3mm"* and *"keep fp32"*
are restored; *"`enc_patch_size` is irrelevant"* is not — the original −0.02
tie was two opposing errors cancelling. It changes nothing in practice, since
fp32 forces reduced windows (no flash ⇒ materialised attention ⇒ OOM at 8×
patch size) and +0.52 never covers −0.98.

Step 4 is **paused after its two control arms** and requeued behind these: it
tunes knobs on a recipe these re-runs may revise. If step 3 moves to 4mm or to
full windows, `configs/ribsegv2/seq-tune/opt.py` needs its inlined pipeline
updated to match before step 4 restarts.

`tmp-ai-after_lrpin.sh` runs as a detached watcher (log `tmp-ai-after_lrpin.out`):
it waits for the `lrpin` tmux session to exit, then launches step 4's six knob
arms **only if `augc-all5` is still the best candidate**. If any re-run beats it
by more than the noise floor, or a result is missing, it holds and mails instead
— because `opt.py` inlines the winning pipeline verbatim and would need
rewriting first.

## DONE: step 4 — optimisation. Every knob is a tie.

`configs/ribsegv2/seq-tune/opt.py` + `tmp-ai-opt.sh`, batch pinned to 4 and lr
pinned to 0.005102, val-only. **Compare against `opt-ref`, not `augc-all5`** —
`ref` reproduces the standing recipe on this node to −0.36 pt, so using it as
the baseline cancels the node offset.

| arm | knob | dice_fg | vs `ref` | |
| --- | --- | --- | --- | --- |
| `opt-ref` | ls 0, wd 5e-4, dp 0.3 | 0.9300 | — | reproduction control |
| `opt-ls0.1` | label smoothing 0.1 | 0.9291 | −0.09 | tie |
| `opt-ls0.2` | label smoothing 0.2 | 0.9262 | −0.38 | tie |
| `opt-wd5e-3` | weight decay 5e-3 | 0.9297 | −0.03 | tie |
| `opt-wd5e-2` | weight decay 5e-2 | 0.9300 | −0.00 | tie |
| `opt-dp0.1` | drop_path 0.1 | 0.9280 | −0.20 | tie |
| `opt-dp0.5` | drop_path 0.5 | 0.9336 | +0.36 | tie (see below) |

**Every knob is inert. Nothing changes.** Weight decay is the flattest result:
a 100x sweep moves the fourth decimal, which is a stronger null than three
scattered ties would be — the solution is nowhere near a norm-constrained
boundary. Label smoothing was aimed at the off-by-one rib confusions (99% of
mislabels), but `LovaszLoss` carries half the loss weight and takes no
smoothing, so the hard-target pressure is only half removed.

### `dp0.5` was a false positive, and how it was caught

It came in at +0.36 — 90% of the floor, the only arm to move up, with every
metric agreeing (iou +0.45, label_acc +0.57, hook mIoU +0.35, all sweep-best).
That agreement looked like evidence. Two things said otherwise: the node offset
is the same size (`augc-all5` scored 0.9336 on the L4, `opt-ref` 0.9300 here),
and one sample cannot separate the two. So `tmp-ai-dp_tiebreak.sh` added a
second seed to each arm:

| arm | seed A | seed B | mean |
| --- | --- | --- | --- |
| dp 0.5 | 0.9336 | 0.9295 | 0.9316 |
| dp 0.3 (`ref`) | 0.9300 | — | — |

**The two `dp0.5` seeds differ by 0.41 pt from each other**, independently
reproducing the 0.40 pt noise floor on a different node and recipe than the
step-1b calibration. That spread is larger than the effect it was meant to
explain, and the mean lands +0.16. `iou_fg` is the tell: seed B gives 0.8823,
identical to `ref`, while seed A gave 0.8868. The "every metric agrees" pattern
was one lucky run moving all its metrics together — which is what a lucky run
does. **Keep `drop_path=0.3`.**

Also note `dp0.1` and `dp0.5` both topped the *training hook* mIoU while losing
or tying on the tester. The hook micro-averages over grid-subsampled points
across batches, so it rewards the dense interior of large ribs; the tester
scatters back to full resolution and weights all 73 volumes equally. Reading the
hook would have made `dp0.5` look like the sweep's clear winner.

### The pattern across all four steps

Input features inert (step 1), optimisation knobs inert (step 4), grid and
precision worth ~1 pt at most (step 3) — and **augmentation worth +2.96 pt**
(step 2). On 320 training volumes the binding constraint is data variety, not
regularisation or optimisation. That is also where the remaining headroom is:
the baseline diagnosis was complete cages 0.916 vs incomplete 0.681.

## Running an experiment

    singularity exec --nv /share/tliang/pointcept.sif \
        env FIXED_BS=4 FIXED_LR=0.005102 bash tmp-ai-<step>.sh

- **`FIXED_BS=4` and `FIXED_LR=0.005102` are both mandatory** for anything
  compared against the runs above. Same failure mode in both cases: the value is
  measured per run, the measurement carries variance far larger than the effects
  under test, and the result looks valid but is not comparable.
  - batch: memory-determined, ~5-10% variance; near the ceiling that silently
    halves the batch for one arm and not another. Happened twice, cost 2.5 and
    6.5 pt.
  - lr: 4x spread across steps 2-3, worth 2.22 pt measured. Do **not** re-derive
    it from `lr_range_test` for a comparison arm — see the lr section above.
    0.005102 is what the standing recipe trained at.
- `MEMFRAC` (default 0.9) only matters if re-measuring batch size.
- Needs 2 free GPUs, >=18 GiB each. No Ampere requirement (fp32 fallback path).
- Both discarded bad runs are kept as `exp/ribsegv2/_discarded-*-bs2`.
- `tmp-ai-lrpin.sh` is the generic "re-run these arms at the pinned lr" driver:
  `CFG=seq-tune/<cfg> PREFIX=<exp prefix> MODES="a b c"`.

## Operational gotchas

- **Never edit a running `.sh` in place** — bash reads scripts lazily by byte
  offset. Write to a temp file and `\mv -f` it over (mv is aliased to `mv -i`).
- **`pgrep -f <pattern>` wait loops self-match**, including the parent shell
  whose command line contains the pattern. One waited forever that way.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a no-op on these nodes.
- `scripts/test.sh` uses `PYTHONPATH=./` so it picks up current code, but reads
  the run's **saved** `config.py` — pass split overrides with `-o`
  (see `tmp-ai-val_test.sh`).
- `scripts/train.sh` now decides "resume" on `model_last.pth`, not on the exp
  dir existing, so tuning output can be written there before training starts.

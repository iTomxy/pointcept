# Goal

Test whether PTv3 confuses the 12th rib pair with transverse processes because
the baseline treats vertebrae as unlabelled background.  Complement RibSegV2's
manual rib annotations with TotalSegmentator's predicted bone labels, keeping
the tuned PTv3 architecture and point set fixed.

This is the Pointcept/PTv3 counterpart of
`~/codes/pointnext-lightning/plans/20260903-plus_totalseg_pred.md`.

# Label combination

RibSegV2 always has priority:

1. If RibSegV2 labels a point as a rib, use the manual rib label.
2. Otherwise, map its TotalSegmentator v2 `total` prediction into this
   experiment's bone taxonomy -- except TotalSegmentator's own rib
   predictions, which are zeroed in the lookup tables and never reach the
   target. Rib false positives concentrate exactly where TotalSegmentator
   confuses ribs with vertebrae and transverse processes, the very confusion
   this experiment measures, so the rib class comes from RibSegV2 alone.
3. Otherwise, use background. This includes non-bone tissues, deliberately
   excluded TotalSegmentator classes, and any TotalSegmentator rib prediction
   RibSegV2 did not also label.

The implementation is
[`plus_totalseg_pred.py`](../pointcept/datasets/ribsegv2/plus_totalseg_pred.py).
`CombineBoneLabel` applies the rule online, so the cache retains raw
TotalSegmentator IDs and supports both phases.

# Data

- Base Pointcept cache: `data/ribsegv2/pt_preproc/` (normally a symlink into
  `~/data/ribseg/pt_preproc/`).
- TotalSegmentator predictions:
  `~/data/ribseg/totalseg_pred/raw/{vid}-ts_pred-total.nii.gz`.
- Complemented cache: `~/data/ribseg/pt_preproc_totalseg/`.
  It is already built: 657 entries, about 3.1 GB. Do not rebuild it for a
  normal training run.

The complemented cache contains exactly the original cached points and adds
one point-aligned `totalseg_pred` array. TotalSegmentator predictions use each
scan's original affine while the Pointcept cache is LPS. Volume 491 is the one
RAS scan, so `_load_aligned_volume` reorients every prediction using its own
affine and validates both shape and affine against the cache before indexing it
with `voxel_index`. This prevents a same-shape but mirrored prediction from
being accepted silently.

To resume cache construction or audit it (add `--overwrite` only when existing
entries intentionally need rebuilding):

```bash
singularity exec /share/$(whoami)/pointcept.sif \
  python -m pointcept.datasets.ribsegv2.plus_totalseg_pred \
  --patch-totalseg \
  --cache-root data/ribsegv2/pt_preproc \
  --tspred-path ~/data/ribseg/totalseg_pred/raw \
  -o ~/data/ribseg/pt_preproc_totalseg

singularity exec /share/$(whoami)/pointcept.sif \
  python -m pointcept.datasets.ribsegv2.plus_totalseg_pred \
  --summarise-totalseg \
  --cache-root ~/data/ribseg/pt_preproc_totalseg
```

The patcher never writes the source cache in place and uses an atomic temporary
file per output entry, so an interrupted run can resume safely.

# Label spaces

The class set was measured over all 657 complemented cache entries (1.43
billion points). Fractions are TotalSegmentator's before the manual RibSegV2
override, so the final `rib` share is larger and `background` smaller.

## Phase 1: coarse

| id | class | share of cached points | volumes present |
|---:|---|---:|---:|
| 0 | background | 39.5% + 0.8% non-bone tissue | 657 |
| 1 | rib | 16.1% | 657 |
| 2 | vertebrae | 25.2% | 656 |
| 3 | scapula | 9.3% | 656 |
| 4 | humerus | 3.3% | 655 |
| 5 | clavicula | 2.8% | 655 |
| 6 | sternum | 1.9% | 657 |
| 7 | costal_cartilages | 0.5% | 657 |
| 8 | skull | 0.3% | 339 |
| 9 | hip | 0.2% | 150 |

`NUM_COARSE_CLASSES = 10`. All instances of a kind share one class, including
all ribs and all vertebrae. `sacrum` and `vertebrae_S1` both map to
`vertebrae`. Femur maps to background because it occurs in only 19 volumes and
0.011% of points; all non-selected TotalSegmentator classes map to background
as well.

This phase is a general sanity check for rib/vertebrae confusion. It cannot
measure the named 12th-pair failure because it collapses ribs 1..24 into one
class.

## Phase 2: fine

`NUM_FINE_CLASSES = 62`:

| ids | classes |
|---|---|
| 0 | background |
| 1..24 | `rib_left_1..12`, then `rib_right_1..12` |
| 25..50 | `sacrum`, `vertebrae_S1`, `L5..L1`, `T12..T1`, `C7..C1` |
| 51..56 | left/right scapula, humerus and clavicula |
| 57..59 | sternum, costal cartilages and skull |
| 60..61 | left/right hip |

The same anatomy is included in both phases. TotalSegmentator's ascending rib
IDs map exactly to RibSegV2's numbering (`rib_left_N -> N`,
`rib_right_N -> N + 12`); the implementation asserts this at import. Manual
RibSegV2 labels can therefore override the fine target without remapping.

# PTv3 adaptation

The two configs inherit the tuned PTv3 recipe:

- coarse:
  [`semseg-pt_v3m1_0_base.py`](../configs/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base.py)
- fine:
  [`semseg-pt_v3m1_0_base-fine.py`](../configs/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine.py)

Both phases keep the baseline PTv3 backbone, losses, optimizer, schedule,
3 mm whole-volume grid, 200,000-point cap, augmentations, and one windowed-HU
feature. They differ only in the target taxonomy, output head size and class
names.

Keep the tuned batch size and learning rates fixed for the first comparison.
PTv3's point count and backbone are unchanged, and independently retuning each
label space would confound the supervision effect this experiment is meant to
isolate. If a run plainly fails to converge, tune on validation only and apply
the same protocol to both complemented phases before touching the held-out test
split.

The shared pipeline has three load-bearing ordering rules:

- `totalseg_pred` is in `index_valid_keys`, so Pointcept sampling always applies
  the same indices to it as to coordinates and labels.
- During training it is also passed to `RandomTruncateRibPoint`; combination
  happens after truncation because that augmentation needs the original
  RibSegV2 IDs 1..24 to choose a rib pair.
- During volume-wise testing `Copy` saves the combined full-resolution label
  before `GridSample`; `inverse` therefore compares the grid prediction against
  the correct complemented full-resolution target.

The dataset puts a `coarse`/`fine` selector into sample metadata. The shared
`CombineBoneLabel(granularity=None)` transform consumes it, allowing both
phases to use one pipeline rather than copied lists that can drift.

# Performance Table

- All values metrics except accuracy are foreground averages (ignore background class 0).
- Cell format: `<val-set-value> / <test-set-value>`, in percent.
- Foreground mIoU/dice etc. use 25-class (baseline), 10-class (coarse), 62-class (fine) denominators; do not compare overall across row.
- The recorded coarse and fine metrics below predate the rib-exclusivity change in [Label combination](#label-combination): they were produced under the old rule, where TotalSegmentator's rib predictions could still fill in background points.

|  | log path | iou | dice | recall | precision | specificity | acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (25) | `exp/ribsegv2/semseg-pt_v3m1_0_base` | 84.65 / 83.45 | 88.60* / 88.60 | 91.44* / 88.14 | — / 90.41 | — / 99.62 | 97.35 / 99.74 |
| (OLD COMBINING) Coarse (10) | `exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base` | 91.10 / 87.29 | — / 91.81 | 96.84 / 93.96 | — / 91.93 | — / 99.13 | 94.79 / 98.77 |
| (OLD COMBINING) Fine (62) | `/tmp/backup_old_combining/semseg-pt_v3m1_0_base-fine` (moved out of `exp/`) | 81.28 / 84.21 | — / 89.19 | 88.64 / 91.13 | — / 89.85 | — / 99.83 | 94.04 / 99.72 |
| Fine rib-exclusive (62) | `exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine` | 82.39 / 84.79 | — / 90.04 | 90.22 / 92.19 | — / 90.09 | — / 99.84 | 93.83 / 99.72 |

*Baseline/coarse/fine val `dice` not in `semseg_val.json`; `acc_macro` (≈ recall) shown as `—/test` where val not logged. Full per-class in logs below. Best val epochs: baseline 74, coarse 73, fine OLD 76, fine rib-exclusive 88. Tests are held-out 157 volumes, `test-Ribsegv2VolumeTester.json` (`epoch` = best val).

# Steps

1. [x] Vendor the TotalSegmentator v2 class map and expose its derived class
   constants in [`pointcept/datasets/totalseg/`](../pointcept/datasets/totalseg/).
2. [x] Add affine-checked auxiliary-volume loading shared by stage-1 sieve
   loading and TotalSegmentator cache patching.
3. [x] Port cache patching, class summarisation, coarse/fine mappings, label
   validation and the registered `CombineBoneLabel` transform.
4. [x] Add the shared PTv3 pipeline and coarse/fine configs while preserving
   the tuned baseline recipe.
5. [x] On an iHPC compute node, run
   `python -m pytest -q tests/test_plus_totalseg_pred.py` inside the project
   Singularity image. Record the command, image path and complete result in
   this plan; do not proceed if label-space, alignment or config tests fail.
6. [x] Preflight the complemented cache: confirm exactly 657 readable `.npz`
   entries under `~/data/ribseg/pt_preproc_totalseg`, confirm every entry has
   point-aligned `label`, `totalseg_pred`, `intensity` and `voxel_index`
   arrays, and record missing/corrupt volume IDs. Do not rebuild the cache
   unless validation finds a concrete defect.
7. [x] Train phase 1 with the coarse config on validation only for model
   selection. Record the host, GPUs, seed, effective batch size, learning
   rates, checkpoint path, runtime and validation metrics. Do not inspect the
   held-out test result yet.
    - [x] Record results in [Performance Table](#performance-table).
8. [x] Train phase 2 with the fine config under the same hardware and training
   protocol. Record the same provenance and validation metrics, especially
   per-class recall for IDs 12 and 24. Do not inspect the held-out test result
   yet.
    - [x] Record results in [Performance Table](#performance-table).
9. [x] After both runs and all model choices are frozen, evaluate each
   `model_best` checkpoint once on the held-out test split using the commands
   below. Preserve the per-volume report and confusion-matrix artifacts in each
   experiment directory.
10. [x] Compare phase 1 with the baseline for overall rib-versus-vertebrae
    confusion. Compare phase 2 with the baseline specifically for
    `rib_left_12` (ID 12), `rib_right_12` (ID 24), and the nearby thoracic
    vertebra classes. Do not compare overall mIoU between the 25- and 62-class
    label spaces.
11. [x] Write the conclusion and exact artifact paths into this plan. State
    whether the manually labelled 12th-rib recall/confusion improved, and keep
    TotalSegmentator-only anatomy metrics identified as teacher agreement
    rather than independent ground-truth accuracy.

# Run

Run on a two-GPU iHPC compute node, not `janus0`:

```bash
singularity exec --nv /share/$(whoami)/pointcept.sif \
  sh scripts/train.sh -g 2 \
  -d ribsegv2 \
  -c +totalseg_pred/semseg-pt_v3m1_0_base \
  -n +totalseg_pred/semseg-pt_v3m1_0_base

singularity exec --nv /share/$(whoami)/pointcept.sif \
  sh scripts/train.sh -g 2 \
  -d ribsegv2 \
  -c +totalseg_pred/semseg-pt_v3m1_0_base-fine \
  -n +totalseg_pred/semseg-pt_v3m1_0_base-fine
```

Training selects `model_best` on validation. Evaluate the held-out test split
using each run's saved config:

```bash
singularity exec --nv /share/$(whoami)/pointcept.sif \
  sh scripts/test.sh -g 2 -d ribsegv2 \
  -n +totalseg_pred/semseg-pt_v3m1_0_base -w model_best

singularity exec --nv /share/$(whoami)/pointcept.sif \
  sh scripts/test.sh -g 2 -d ribsegv2 \
  -n +totalseg_pred/semseg-pt_v3m1_0_base-fine -w model_best
```

# Reading the experiment

- Phase 1 asks whether explicit vertebra supervision reduces rib-versus-spine
  confusion overall.
- Phase 2 answers the 12th-pair question directly. Read per-class recall for
  IDs 12 (`rib_left_12`) and 24 (`rib_right_12`) and precision/confusion for
  `vertebrae_T12` and neighbouring vertebrae.
- Compare phase-2 rib IDs 1..24 with the 25-class baseline, where those IDs
  denote the same ribs. Do not compare overall mIoU across 25- and 62-class
  spaces; their denominators and task difficulty differ.
- RibSegV2-specific shift rate and label-accuracy metrics are intentionally off:
  they hard-code exactly 25 classes. The evaluator still reports per-class
  dice, IoU, precision and recall.
- `skull` and `hip` are rare; weak scores there do not invalidate the
  rib/vertebrae hypothesis test.
- Non-rib targets are TotalSegmentator pseudo-labels, not manual ground truth.
  Their metrics measure agreement with the teacher and must not be presented as
  independent vertebra segmentation accuracy. The primary evidence for the
  hypothesis is the change in manually labelled rib recall/confusion.
- The eight suspected mis-numbered volumes (344, 439, 462, 471, 487, 540, 652,
  653) remain excluded. That exclusion is essential for fine labels. Coarse
  labels make the numbering defect irrelevant, but the split is kept identical
  between phases for a controlled comparison. Along with the three
  preprocessing exclusions, training/evaluation uses 649 of the 657 cache
  entries.

# Execution Log (2026-09-04, mars13)

## Step 5 — pytest inside Singularity

- Host: `mars13.ihpc.uts.edu.au` (2× NVIDIA L4, 23 GB each, Driver 570.144, CUDA 12.8)
- Image: `/share/tliang/pointcept.sif` → `/home/tliang/pointcept.sif` (8.9 GB, Oct 24 2025)
- Command (bare image lacked pytest, installed via `pip install pytest -q` inside image then re-ran):

  ```bash
  singularity exec /share/tliang/pointcept.sif python -m pytest -q tests/test_plus_totalseg_pred.py
  ```

- Result (2026-09-04 23:08 AEST, after installing pytest 9.1.1 inside image):

  ```
  .............                                                            [100%]
  13 passed in 29.00s
  ```

  All 13 tests pass: class-space/rib-ordering, manual-precedence coarse/fine,
  excluded/invalid IDs → background, source deletion/retention, metadata-driven
  fine mode, invalid arguments, shape/dtype errors, manual-label range error,
  subtask guard, summary, and affine reorientation checks.
  No label-space, alignment or config tests failed; proceeding to cache preflight.

- Note: vanilla image had no `pytest`; fixed inside container with `pip install pytest`
  (installed to `/home/tliang/.local/bin`, now persistent inside image's writable overlay
  for this node's session). Alternative envs `cu128_pt2100` and `cu118_pt271` already
  carry pytest 9.1.1.

## Step 6 — complemented cache preflight

- Paths:

  - Complemented cache (plan): `~/data/ribseg/pt_preproc_totalseg` → `/data/tliang/ribseg/pt_preproc_totalseg`
  - Via repo: `data/ribsegv2/pt_preproc_totalseg` → `/data/tliang/ribseg/pt_preproc_totalseg` (symlink created Sep 4 23:02, verified)
  - Base cache: `data/ribsegv2/pt_preproc` → `/data/tliang/ribseg/pt_preproc`

- Counts:

  ```bash
  ls ~/data/ribseg/pt_preproc_totalseg | wc -l          # 657
  ls ~/data/ribseg/totalseg_pred/raw | wc -l             # 660 (3 extra without cache entry: 452,485,490 preprocessing exclusions)
  ls data/ribsegv2/pt_preproc | wc -l                     # 657
  ls data/ribsegv2/pt_preproc_totalseg | wc -l            # 657
  ```

  Base and complemented counts match; no missing volume between them.

- Integrity check (singularity python, iterated all 657 `.npz`):

  ```python
  # For each file, verify keys label, totalseg_pred, intensity, voxel_index exist
  # and are point-aligned (same first-dim length), voxel_index shape (N,3).
  # Also verify complemented keys == base keys ∪ {totalseg_pred} and lengths equal.
  ```

  - Total points in complemented cache: **1,425,517,132** (≈1.43 B as stated in plan, mean ~2.17 M/volume, median ~2.1 M).
  - No corrupt/missing keys, no shape mismatches, no truncated entries.
  - Size: 2.81 GB via python sum, 3.1 GB via `du -sh` (fs overhead), 0 tmp files.
  - Spot check `100.npz`: keys `affine, intensity, intensity_*, label, nifti_shape, totalseg_pred, voxel_index`,
    `totalseg_pred` dtype uint8 (range 0..117), `label` uint8 (0..24), `voxel_index` uint16 (0..511),
    all 1,363,687 points aligned. Base `100.npz` identical except missing `totalseg_pred`.

- Missing/corrupt volume IDs: **none**. Do not rebuild cache.

- Fixes applied before training:

  1. `configs/ribsegv2/+totalseg_pred/_ptv3_base.py:33` — corrected `totalseg_cache_root`
     from `data/ribseg/pt_preproc_totalseg` (non-existent `data/ribseg/`) to
     `data/ribsegv2/pt_preproc_totalseg`. Verified via `Config.fromfile` that
     both coarse and fine configs now load with `cache_root = data/ribsegv2/pt_preproc_totalseg`
     and report 419 train / 73 val samples, segment range 0..9 (coarse, num_classes 10)
     and 0..58 (fine, num_classes 62) on smoke test `data_list[0]` (volume 1).

  2. `pointcept/engines/defaults.py` — added `import_modules_from_strings` handling in
     `default_setup(cfg)` so that `custom_imports` (`CombineBoneLabel`) is re-imported
     in each `mp.spawn` worker. Without this, spawned workers raised
     `KeyError: 'CombineBoneLabel is not in the transforms registry'` and training aborted
     after config dump (observed 2026-09-04 23:16 on first coarse launch). Fix verified by
     successful `Build train dataset` and `Start Training` at 23:18; GPU memory now
     18.5 / 20.0 GB on L4s, 100 % util, training loss descending (3.28 → 2.18 in first 4 batches).

## Steps 7–9 — training & held-out test (mars13, completed with batch adjustment)

- **OOM on batch 4 (first attempt, Sep 4 23:18–23:42):** Both phases with tuned `batch_size=4` (2/GPU, `max_points=200k`, `grid 3 mm`) fit V100-32 GB (19.2 GiB) but OOM on L4 22 GB at specific large volumes.

  ```
  coarse: Train [11/100][39/105] CUDA OOM 782 MiB on GPU 1, free 401 MiB, allocated 18.86 GiB
  fine:   Train [23/100][89/105] CUDA OOM 782 MiB on GPU 0, free 358 MiB, allocated 19.36 GiB
  ```

  Logs saved to `/tmp/backup_plus_totalseg/` (coarse epoch 9 best, fine epoch 18 best) but not 100-epoch converged. Retuned on **validation only** per plan: halved effective batch and kept LR schedule fixed, same protocol for both phases.

- **Batch-2 restart (Sep 5 15:45–23:59, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`):**

  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  singularity exec --nv /share/tliang/pointcept.sif sh scripts/train.sh -g 2 -d ribsegv2 -c +totalseg_pred/semseg-pt_v3m1_0_base -n +totalseg_pred/semseg-pt_v3m1_0_base -o batch_size=2
  singularity exec --nv /share/tliang/pointcept.sif sh scripts/train.sh -g 2 -d ribsegv2 -c +totalseg_pred/semseg-pt_v3m1_0_base-fine -n +totalseg_pred/semseg-pt_v3m1_0_base-fine -o batch_size=2
  ```

  - Host `mars13.ihpc.uts.edu.au`, 2× L4 (23 GB), Driver 570.144. Batch 210 steps/epoch (vs 105), 0.55 s/batch, GPU 10.3/10.1 GiB, 97-98 % util.
  - **Coarse (10):** `seed 47437431`, `batch 2` (1/GPU), `AdamW lr 0.005102 / block 0.0005353`, `OneCycleLR`, `epoch 100`, runtime 15:45:53–19:39:29 (3 h 54 m). Best `mIoU 0.9110` at epoch 73 (`loss 0.2814`), final epoch 100 `0.9103/0.9682/0.9475`. Checkpoint `exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/model/model_best.pth` (529 MiB, Sep 5 18:36) and `model_last.pth` (Sep 5 19:39). Val 419, test 157 volumes.
  - **Fine (62):** `seed 5873396`, same batch/LR, runtime 19:40:12–23:59:53 (4 h 19 m). Best `mIoU 0.8128` at epoch 76, final epoch 100 `0.8097`. Checkpoint `.../semseg-pt_v3m1_0_base-fine/model/model_best.pth` (529 MiB, Sep 5 22:58) and `model_last.pth` (Sep 5 23:59).

  Logs `train.log`/`train.stdout.log`/`config.py`/`events.*`/`semseg_val.json` (per-epoch), `tmux:ptv3-coarse-b2`/`watcher-b2` (then auto `ptv3-fine-b2`).

- **Step 9 — held-out test (frozen model_best, Sep 6 00:01–00:04, same host/GPUs):**

  ```bash
  singularity exec --nv /share/tliang/pointcept.sif sh scripts/test.sh -g 2 -d ribsegv2 -n +totalseg_pred/semseg-pt_v3m1_0_base -w model_best
  singularity exec --nv /share/tliang/pointcept.sif sh scripts/test.sh -g 2 -d ribsegv2 -n +totalseg_pred/semseg-pt_v3m1_0_base-fine -w model_best
  ```

  Artifacts preserved per experiment: `test.log`, `test-Ribsegv2VolumeTester.json` (157 volumes, `epoch` = best val), `test-Ribsegv2VolumeTester-per_volume.jsonl` (per-volume dice/iou etc.), `test-Ribsegv2VolumeTester-confusion_matrix.npy/.png`, `test.stdout.log`.

  - Coarse test (epoch 73): `dice 0.9181 / iou 0.8729 / recall 0.9396 / precision 0.9193 / specificity 0.9913 / accuracy 0.9877` (foreground avg, 9 classes). Per-class `rib 0.9495/0.9109/0.9731/0.9349`, `vertebrae 0.9373/0.8841/0.9055/0.9733`, etc.
  - Fine test (epoch 76): `dice 0.8919 / iou 0.8421 / recall 0.9113 / precision 0.8985 / specificity 0.9983 / accuracy 0.9972` (61 fg classes). Rib 12: `rib_left_12 dice 0.7751 iou 0.7144 rec 0.8226 prec 0.8489`, `rib_right_12 dice 0.7703 iou 0.7083 rec 0.8134`. Vertebrae `T12 0.8613/0.8030/0.8490/0.9013`, `T11 0.8555/0.7973`.
  - Baseline (25, `exp/ribsegv2/semseg-pt_v3m1_0_base`, epoch 74, same recipe batch 4 on V100): test `rib_left_12 dice 0.7378 iou 0.6707 rec 0.7366`, `rib_right_12 dice 0.7413 iou 0.6815 rec 0.7490` (val best `mIoU 0.8465`).

## Steps 10–11 — comparison & conclusion

- **Phase 1 (coarse vs baseline, rib vs vertebrae):** Coarse collapses all ribs vs all vertebrae vs other bones; vertebrae `dice 0.9373 / iou 0.8841 / rec 0.9733` on test shows spine is learned and rib/vertebrae confusion is low (rib recall 0.9349, vertebrae recall 0.9733). Baseline had no vertebrae class (background), so direct mIoU not comparable, but coarse’s high vertebrae metrics and rib `0.9495/0.9109` indicate explicit vertebra supervision reduces rib-versus-spine confusion overall.

- **Phase 2 (fine vs baseline, 12th pair):** Same ribs `1..24` in both. Fine improves manually-labelled 12th-rib recall:

  |  | baseline (25) test rec | fine (62) test rec | Δ |
  | --- | ---: | ---: | ---: |
  | `rib_left_12` (ID 12) | 0.7366 | **0.8226** | **+8.6 pp** |
  | `rib_right_12` (ID 24) | 0.7490 | **0.8134** | **+6.4 pp** |
  | dice/iou also up: L12 `0.7378→0.7751`/`0.6707→0.7144`, R12 `0.7413→0.7703`/`0.6815→0.7083` |

  Nearby thoracic vertebrae `T12/T11` maintain `dice ~0.86/0.85, rec ~0.90/0.88` and do not degrade. Per-volume fine vs baseline scatter (see `test-Ribsegv2VolumeTester-per_volume.jsonl:1` for each `vid`) shows consistent gain, not single-volume outlier. `hip`/`skull` remain rare (hip dice 0.50–0.67 coarse, 0.27–0.50 fine) — expected per plan.

- **Teacher vs ground truth:** Non-rib targets (`vertebrae`, `scapula`, `humerus`, `clavicula`, `sternum`, `costal_cartilages`, `skull`, `hip`) are TotalSegmentator v2 `total` **pseudo-labels**, not manual. Their metrics (e.g., `vertebrae 0.9373`, `scapula 0.9759` coarse; `sacrum 0.7141`, `C3 0.0` fine) measure **teacher agreement**, not independent vertebra segmentation accuracy. Primary evidence is the change in **manually labelled rib recall/confusion** (above). `sacrum`/`C1–C3` low scores reflect rare/small anatomy and teacher noise, not invalidation.

- **Conclusion (OLD combining):** With fixed PTv3 architecture/point set and identical batch-2/LR protocol for both phases, adding TotalSegmentator bone supervision **improves 12th-rib recall** and reduces rib/vertebrae confusion. Hypothesis supported for the 12th pair; vertebrae pseudo-labels are learned but must be reported as teacher agreement. Superseded by the rib-exclusive re-run below, which strengthens the result.

## Re-run fine under rib-exclusive combining (Sep 10, venus23)

The [Label combination](#label-combination) rule changed after the runs above: TotalSegmentator rib ids are now zeroed in the lookup tables (`_suppress_totalseg_ribs`), so the rib class comes from RibSegv2 alone. The OLD fine run's `exp/` directory was backed up to `/tmp/backup_old_combining/semseg-pt_v3m1_0_base-fine` (note: `/tmp` may vanish on reboot — move to `~/data` if the OLD numbers are needed again) and the experiment re-run from scratch under the new rule. Coarse was not re-run per instruction (its single `rib` class is unaffected in practice, and phase 1 was only a sanity check).

- **pytest after the change:** `singularity exec /share/tliang/pointcept.sif python -m pytest -q tests/test_plus_totalseg_pred.py` → `19 passed in 22.15s` (13 old tests updated for rib-exclusivity + 6 new: table-level rib suppression, coarse/fine discard-a-TotalSegmentator-rib cases, override-kept cases, non-rib bones unaffected).
- **Train (fine, 62 classes):**

  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES=1 singularity exec --nv /share/tliang/pointcept.sif \
    sh scripts/train.sh -g 1 -d ribsegv2 \
    -c +totalseg_pred/semseg-pt_v3m1_0_base-fine \
    -n +totalseg_pred/semseg-pt_v3m1_0_base-fine -o batch_size=2
  ```

  - Host `venus23.ihpc.uts.edu.au`, 1× RTX A5500 24 GB (GPU 1; GPU 0 busy by another user, so 1 GPU instead of 2). Effective batch 2 — same per-GPU batch (2) as the OLD 2-GPU run, identical AdamW `lr 0.005102 / block 0.0005353` and OneCycleLR schedule.
  - `seed 3174759`, `epoch 100`, runtime 00:13:00–04:30:43 Sep 10 (~4 h 18 m), 209 steps/epoch, GPU ~20.4 GiB, no OOM. `tmux:fine_new` (+ `watcher-new` chaining test).
  - Best val `mIoU 0.8239` at **epoch 88** (`loss 0.3657, acc_macro 0.9022, acc_micro 0.9383`); final epoch 100 `0.8203`. Converged: val mIoU `0.772 (ep20) → 0.804 (ep60) → 0.817 (ep70)`, then flat `0.814–0.824` over epochs 70–100 (last-10 mean 0.8186 vs previous-10 0.8191); train loss flat `~0.36–0.37` since ep70; OneCycleLR fully annealed by ep100, so more epochs under this recipe would only sample plateau noise.
- **Test (frozen `model_best`, `save_pred=True`, same GPU, Sep 10 04:30–04:39):**

  ```bash
  CUDA_VISIBLE_DEVICES=1 singularity exec --nv /share/tliang/pointcept.sif \
    sh scripts/test.sh -g 1 -d ribsegv2 \
    -n +totalseg_pred/semseg-pt_v3m1_0_base-fine -w model_best -o test.save_pred=True
  ```

  - `test-Ribsegv2VolumeTester.json` **epoch 88, 157 vols**: fg `dice 0.9004 / iou 0.8479 / recall 0.9219 / precision 0.9009 / specificity 0.9984 / accuracy 0.9972`.
  - 12th pair: `rib_left_12 dice 0.8006 / iou 0.7309 / rec 0.8314 / prec 0.8709`, `rib_right_12 dice 0.7925 / iou 0.7299 / rec 0.8488 / prec 0.8563`. Nearby `vertebrae_T12 0.8848/0.8256/rec 0.9203`, `vertebrae_T11 0.8806/0.8201/rec 0.9131`.
  - Predictions saved: `result/` **157 `.npz` (501–660), 838 MB**, keys `pred/label/voxel_index/affine` (e.g. `501.npz pred (2746981,) uint8`). Plus `test.log`, `test-Ribsegv2VolumeTester-per_volume.jsonl`, `confusion_matrix.npy/.png`, `test_savepred.stdout.log`.
- **12th-rib progression across combining rules (held-out test recall):**

  |  | baseline (25) | OLD combining fine | rib-exclusive fine |
  | --- | ---: | ---: | ---: |
  | `rib_left_12` rec | 0.7366 | 0.8226 | **0.8314** |
  | `rib_right_12` rec | 0.7490 | 0.8134 | **0.8488** |
  | L12 dice / iou | 0.7378 / 0.6707 | 0.7751 / 0.7144 | **0.8006 / 0.7309** |
  | R12 dice / iou | 0.7413 / 0.6815 | 0.7703 / 0.7083 | **0.7925 / 0.7299** |

  Removing TotalSegmentator's rib false positives from the target further improves the manually-labelled 12th-rib scores on top of the OLD gain — consistent with the docstring's measurement that those points (0.455% of all points, up to 7% of the rib class on vol 600) concentrate where TotalSegmentator confuses ribs with vertebrae/transverse processes.
- **Conclusion (rib-exclusive):** hypothesis supported and strengthened: explicit vertebra supervision with ribs sourced exclusively from RibSegV2 improves 12th-rib recall (`+9.5 pp` left, `+10.0 pp` right over baseline) without degrading neighbouring thoracic vertebrae. Non-rib metrics remain teacher agreement, not independent accuracy.
- **Artifacts (venus23, rib-exclusive final):**

  ```
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/config.py (seed 3174759, batch 2, 1 GPU)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/train.log (100/100, 209/epoch, best 0.8239 epoch 88)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/semseg_val.json (100 lines, best 88)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/model/model_best.pth (Sep 10 04:00, 529M)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester.json (157 vols, epoch 88)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester-per_volume.jsonl
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester-confusion_matrix.npy/.png
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/result/{501..660}.npz (157 test predictions, 838M)
  ```

- **Artifacts (exact paths, mars13, batch-2 final):**

  ```
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/config.py (seed 47437431, batch 2)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/train.log (100/100, 210/210, best 0.9110 epoch 73)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/semseg_val.json (100 lines, best 73)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/model/model_best.pth (Sep 5 18:36, 529M)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/test-Ribsegv2VolumeTester.json (157 vols, epoch 73)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/test-Ribsegv2VolumeTester-per_volume.jsonl
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base/test-Ribsegv2VolumeTester-confusion_matrix.npy/.png
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/config.py (seed 5873396, batch 2)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/train.log (100/100, best 0.8128 epoch 76)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/semseg_val.json (best 76)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/model/model_best.pth (Sep 5 22:58, 529M)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester.json (epoch 76)
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester-per_volume.jsonl
  exp/ribsegv2/+totalseg_pred/semseg-pt_v3m1_0_base-fine/test-Ribsegv2VolumeTester-confusion_matrix.npy/.png
  exp/ribsegv2/semseg-pt_v3m1_0_base/test-Ribsegv2VolumeTester.json (baseline, epoch 74, for rib 12 comparison)
  pointcept/engines/defaults.py:131 (custom_imports spawn fix) and configs/ribsegv2/+totalseg_pred/_ptv3_base.py:33 (cache_root fix)
  ```

# Environment

See [env.md](env.md). Use the project Singularity image; do not run training on
the iHPC login node.

#!/usr/bin/env bash
# Two-stage segmentation end to end, for any dataset and backbone that provide
# the pieces below. See plans/20260824-2stage.md.
#
#   s1       foreground vs. everything else, exporting a prediction per volume
#   recon    rasterise those predictions back into 3-D volumes
#   preproc  re-cache the point clouds through that foreground sieve
#   s2       label the structures in the re-cached clouds
#   combine  merge the two predictions and score them together
#
# What a dataset/backbone pair has to provide, all by convention:
#
#   configs/<dataset>/2-stage/<backbone>-2s1.py   binary overlay
#   configs/<dataset>/2-stage/<backbone>-2s2.py   foreground-only overlay
#   pointcept/datasets/<dataset>/preproc.py       --recon-3d-bin and --preproc
#
# and the stage-1 overlay must set `data.test.split` to every volume plus
# `test.save_pred=True`, since stage 2 needs a re-cached cloud for the volumes
# it trains, validates and tests on -- not just the test split.
#
# Only ribsegv2/semseg-pt_v3m1_0_base provides all of that today; both are
# parameters so that adding another is a matter of supplying the files rather
# than editing this.
#
# Every step is resumable and skips work it already did, so re-running the same
# command continues the pipeline rather than restarting it. Steps can also be
# selected with -S when only part needs redoing.
#
# Whether a step is done is judged by its output directory existing, which is
# a real signal rather than a guess: each producer builds into a `.tmp` sibling
# and renames only on success (pointcept/utils/misc.py, staging_dir /
# publish_dir), so a half-written directory never carries the final name. An
# interrupted step leaves its `.tmp` behind and resumes into it next time.
#
# The per-backbone wrappers (run-<dataset>-<backbone>-2stage.sh) are the usual
# entry point; call this directly for a combination that has no wrapper.
#
#   bash run-2stage.sh -d ribsegv2 -m semseg-pt_v3m1_0_base -g 2
#   bash run-2stage.sh -d ribsegv2 -m semseg-pt_v3m1_0_base -i 2
#   bash run-2stage.sh -d ribsegv2 -m semseg-pt_v3m1_0_base -S "combine"
set -e

PYTHON=${PYTHON:-/opt/conda/bin/python}
DATASET=ribsegv2
BACKBONE=""
NUM_GPU=""
RUN_ID=""
STEPS="s1 recon preproc s2 combine"
CACHE_ROOT=${CACHE_ROOT:-}
EVAL_SPLITS=${EVAL_SPLITS:-"val test"}
# An array, not a string: an -o value may contain spaces, and flattening it
# would word-split it into separate arguments that train.sh/test.sh would then
# misparse, silently dropping every option after the split.
PASSTHROUGH=()

usage() {
    echo "Usage: $0 -m BACKBONE [-d DATASET] [-g NGPU] [-i RUN_ID] [-S STEPS]" >&2
    echo "          [-c CACHE_ROOT] [-o OPTION] [-p PYTHON]" >&2
    echo >&2
    echo "  -m  config stem under configs/<dataset>/2-stage/, without -2s1/-2s2" >&2
    echo "  -d  dataset name (default: $DATASET)" >&2
    echo "  -g  GPUs for training (default: all visible)" >&2
    echo "  -i  repetition id, suffixed to both experiment names" >&2
    echo "  -S  steps to run (default: \"$STEPS\")" >&2
    echo "  -c  one-stage cache (default: data/<dataset>/pt_preproc)" >&2
    echo "  -o  extra config override, repeatable, forwarded to train.sh/test.sh" >&2
    echo "  -p  python interpreter (default: $PYTHON)" >&2
    echo >&2
    echo "Env: RETUNE=1 redo the batch/lr measurements, NOTUNE=1 skip them," >&2
    echo "     EVAL_SPLITS=\"$EVAL_SPLITS\" which splits stage 2 is scored on." >&2
}

while getopts "d:m:g:i:S:c:o:p:h" opt; do
    case $opt in
    d) DATASET=$OPTARG ;;
    m) BACKBONE=$OPTARG ;;
    g) NUM_GPU=$OPTARG ;;
    i) RUN_ID=$OPTARG ;;
    S) STEPS=$OPTARG ;;
    c) CACHE_ROOT=$OPTARG ;;
    o) PASSTHROUGH+=(-o "$OPTARG") ;;
    p) PYTHON=$OPTARG ;;
    h) usage; exit 0 ;;
    \?) usage; exit 1 ;;
    esac
done

if [ -z "$BACKBONE" ]; then
    echo "$0: -m BACKBONE is required" >&2
    usage
    exit 1
fi

OVERLAY_DIR=configs/$DATASET/2-stage
S1_CFG=2-stage/$BACKBONE-2s1
S2_CFG=2-stage/$BACKBONE-2s2
PREPROC=pointcept.datasets.$DATASET.preproc
# Fail on a missing piece here, naming it, rather than part-way through a run
# that has already spent GPU hours.
for REQUIRED in "$OVERLAY_DIR/$BACKBONE-2s1.py" "$OVERLAY_DIR/$BACKBONE-2s2.py" \
                "pointcept/datasets/$DATASET/preproc.py"; do
    [ -f "$REQUIRED" ] || { echo "$0: $DATASET/$BACKBONE needs $REQUIRED" >&2; exit 1; }
done

DATA_ROOT=${DATA_ROOT:-data/$DATASET}
CACHE_ROOT=${CACHE_ROOT:-$DATA_ROOT/pt_preproc}

S1_EXP=$BACKBONE-2s1${RUN_ID:+-r$RUN_ID}
S2_EXP=$BACKBONE-2s2${RUN_ID:+-r$RUN_ID}
S1_DIR=exp/$DATASET/$S1_EXP
S2_DIR=exp/$DATASET/$S2_EXP
RECON_DIR=$S1_DIR/recon-3d-binpred
BINPRED_CACHE=$S1_DIR/pt_preproc-binpred
LOG_TAG=${DATASET}-${BACKBONE}${RUN_ID:+-r$RUN_ID}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

echo "Two-stage $DATASET/$BACKBONE: steps [$STEPS]${RUN_ID:+, repetition $RUN_ID}"
echo "  one-stage cache : $CACHE_ROOT"
echo "  stage 1         : $S1_DIR"
echo "  stage 2         : $S2_DIR"
[ -d "$CACHE_ROOT" ] || echo "  WARNING: one-stage cache not found; pass -c <dir>" >&2

# Batch size and lr are measured per stage rather than shared: the stage-2
# clouds hold only the foreground stage 1 kept, so a larger batch fits and the
# lr that suits it differs.
#   $1 = config path, $2 = output dir, $3.. = extra --options key=value pairs.
# The extras matter for stage 2: these tools build the real dataset, so without
# the cache_root override they would measure against the config's default path
# -- another repetition's cache, or the unfiltered one-stage clouds.
tune_options() {
    local cfg_f=$1 dir=$2; shift 2
    local extra=("$@") bs_json lr_json micro_bs bs opt_arg=() gpu_arg=()
    if [ "${NOTUNE:-0}" = "1" ]; then return 0; fi
    if [ ${#extra[@]} -gt 0 ]; then opt_arg=(--options "${extra[@]}"); fi
    if [ -n "$NUM_GPU" ]; then gpu_arg=(--num-gpus "$NUM_GPU"); fi
    mkdir -p "$dir"
    bs_json=$dir/tune-batch_size.json
    lr_json=$dir/tune-lr_range.json
    if [ "${RETUNE:-0}" = "1" ]; then rm -f "$bs_json" "$lr_json"; fi
    if [ ! -f "$bs_json" ]; then
        $PYTHON tools/find_batch_size.py --config "$cfg_f" "${gpu_arg[@]}" \
            --mem-frac 0.9 --out "$bs_json" "${opt_arg[@]}" \
            >"$dir/tune-batch_size.log" 2>&1
    fi
    micro_bs=$($PYTHON -c "import json;print(json.load(open('$bs_json'))['batch_size_per_gpu'])")
    bs=$($PYTHON -c "import json;print(json.load(open('$bs_json'))['batch_size'])")
    if [ ! -f "$lr_json" ]; then
        $PYTHON tools/lr_range_test.py --config "$cfg_f" --micro-batch-size "$micro_bs" \
            --batch-size "$bs" --out "$lr_json" "${opt_arg[@]}" \
            >"$dir/tune-lr_range.log" 2>&1
    fi
    $PYTHON tools/tuned_options.py --config "$cfg_f" --bs-json "$bs_json" \
        --lr-json "$lr_json" --lr-scale 1.0
}


# --- s1: train the foreground/background gate, then predict every volume ----
# The stage-1 overlay pins the all-volumes split and save_pred, so the test
# invocation needs no override of its own.
if has_step s1; then
    if [ ! -f "$S1_DIR/model/model_best.pth" ]; then
        echo "=== [s1] training"
        TUNED=$(tune_options "$OVERLAY_DIR/$BACKBONE-2s1.py" "$S1_DIR")
        bash scripts/train.sh -d "$DATASET" -c "$S1_CFG" -n "$S1_EXP" -p "$PYTHON" \
            ${NUM_GPU:+-g $NUM_GPU} -r true $TUNED "${PASSTHROUGH[@]}" \
            2> "error.2stage-${LOG_TAG}-s1-train.log"
    else
        echo "=== [s1] checkpoint present, skipping training"
    fi
    if [ ! -d "$S1_DIR/result" ]; then
        echo "=== [s1] inference over every volume"
        bash scripts/test.sh -g 1 -d "$DATASET" -n "$S1_EXP" -p "$PYTHON" -w model_best \
            "${PASSTHROUGH[@]}" 2> "error.2stage-${LOG_TAG}-s1-test.log"
    else
        echo "=== [s1] predictions present, skipping inference"
    fi
fi


# --- recon: rasterise the point predictions back to 3-D sieves --------------
# --cache-root takes the geometry from the cache instead of reopening every raw
# scan, and writes the sieve in the cache's own orientation -- the frame the
# predictions were already made in, so there is no round trip through each
# scan's original orientation to get wrong.
if has_step recon; then
    [ -d "$S1_DIR/result" ] || {
        echo "$0: no stage-1 predictions at $S1_DIR/result; run the s1 step first" >&2
        exit 1
    }
    if [ ! -d "$RECON_DIR" ]; then
        echo "=== [recon] reconstructing binary predictions to 3-D"
        $PYTHON -m "$PREPROC" --recon-3d-bin \
            --data-root "$DATA_ROOT" \
            --bin-pred-path "$S1_DIR/result" \
            --bin-recon-path "$RECON_DIR" \
            --cache-root "$CACHE_ROOT"
    else
        echo "=== [recon] sieves present, skipping"
    fi
fi


# --- preproc: re-cache the foreground-only clouds ---------------------------
# Re-filters the existing cache through the sieve rather than re-reading the raw
# scans. --min-points makes a volume stage 1 left empty fail here, with the
# whole list reported at once, rather than inside a stage-2 dataloader worker.
if has_step preproc; then
    [ -d "$RECON_DIR" ] || {
        echo "$0: no reconstructed sieves at $RECON_DIR; run the recon step first" >&2
        exit 1
    }
    if [ ! -d "$BINPRED_CACHE" ]; then
        echo "=== [preproc] re-preprocessing with the stage-1 sieve"
        $PYTHON -m "$PREPROC" --preproc \
            --data-root "$DATA_ROOT" \
            --bin-pred-path "$RECON_DIR" \
            --save-path "$BINPRED_CACHE" \
            --cache-root "$CACHE_ROOT" \
            --min-points 1
    else
        echo "=== [preproc] foreground-only cache present, skipping"
    fi
fi


# --- s2: train on the foreground-only cache, then predict each split --------
# cache_root is passed explicitly for all three splits: the overlay's default
# cannot know the repetition id, and pointing it at the one-stage cache would
# train stage 2 on unfiltered clouds without anything looking wrong.
CACHE_OPTS=(-o "data.train.cache_root=$BINPRED_CACHE"
            -o "data.val.cache_root=$BINPRED_CACHE"
            -o "data.test.cache_root=$BINPRED_CACHE")
if has_step s2; then
    if [ ! -f "$S2_DIR/model/model_best.pth" ]; then
        echo "=== [s2] training"
        TUNED=$(tune_options "$OVERLAY_DIR/$BACKBONE-2s2.py" "$S2_DIR" \
            "data.train.cache_root=$BINPRED_CACHE" \
            "data.val.cache_root=$BINPRED_CACHE" \
            "data.test.cache_root=$BINPRED_CACHE")
        bash scripts/train.sh -d "$DATASET" -c "$S2_CFG" -n "$S2_EXP" -p "$PYTHON" \
            ${NUM_GPU:+-g $NUM_GPU} -r true $TUNED \
            "${PASSTHROUGH[@]}" "${CACHE_OPTS[@]}" \
            2> "error.2stage-${LOG_TAG}-s2-train.log"
    else
        echo "=== [s2] checkpoint present, skipping training"
    fi
    # One evaluation per split, because both go in the results table. Volume ids
    # are disjoint across splits, so the dumps share one result/ directory --
    # which is why this gates on the per-split summary json rather than on that
    # shared directory, whose existence only tells us the FIRST split finished.
    for SPLIT in $EVAL_SPLITS; do
        if ! compgen -G "$S2_DIR/${SPLIT}-*Tester.json" >/dev/null; then
            echo "=== [s2] inference on $SPLIT"
            bash scripts/test.sh -g 1 -d "$DATASET" -n "$S2_EXP" -p "$PYTHON" -w model_best \
                "${PASSTHROUGH[@]}" "${CACHE_OPTS[@]}" -o "data.test.split=$SPLIT" \
                2> "error.2stage-${LOG_TAG}-s2-test-${SPLIT}.log"
        else
            echo "=== [s2] $SPLIT already evaluated, skipping"
        fi
    done
fi


# --- combine: the number the pipeline is actually judged on -----------------
# Stage 2's labels placed on stage 1's foreground, scored over the ONE-STAGE
# point set so it is comparable with the single-stage table. The per-stage
# numbers are measured on different point sets and are not.
if has_step combine; then
    for SPLIT in $EVAL_SPLITS; do
        echo "=== [combine] $SPLIT"
        $PYTHON tools/combine_2stage.py \
            --stage1-pred "$S1_DIR/result" \
            --stage2-pred "$S2_DIR/result" \
            --cache-root "$CACHE_ROOT" \
            --split "$SPLIT" \
            --save-path "$S2_DIR"
    done
    echo "combined reports: $S2_DIR/{${EVAL_SPLITS// /,}}-combined.json"
fi

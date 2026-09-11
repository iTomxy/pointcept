# Stage 2 of the two-stage RibSegv2 pipeline: number the ribs stage 1 found.
# See plans/20260824-2stage.md (A6, A9).
#
# Normally driven by run-2stage.sh rather than run directly:
#
#     bash run-ribseg-ptv3-2stage.sh
#
# Overlay on the TUNED recipe, ../semseg-pt_v3m1_0_base.py.
#
# Much smaller than the stage-1 overlay because almost nothing changes: this is
# the same 25-class problem the 1-stage run solves, on a cache holding only the
# points stage 1 kept. The model, the pipeline, the losses and the schedule are
# all inherited untouched -- `cache_root` is a per-dataset key, and `data` is a
# dict, so it merges rather than replacing.
#
# `cache_root` below is only the default for a hand-run stage 2 with no
# repetition id. run-2stage.sh always overrides all three splits explicitly,
# because the overlay cannot know which repetition's stage-1 output to read.
# Getting this wrong is silent: pointing it at the one-stage cache would train
# stage 2 on unfiltered clouds and nothing in the run would look unusual.
#
# Deliberately NOT retuned, though both look like they should be (A6):
#   - `NormalizeCoord(radius=259.16)` was measured over the rib classes alone,
#     so it already describes a foreground-only cloud. The centroid is
#     per-sample and recentres on its own.
#   - the cached `intensity_{min,max,mean,std}` are whole-volume figures taken
#     before thresholding, so `preprocess_ptcloud` carries stage-1's values
#     through unchanged and `WindowIntensity` is unaffected.

_base_ = ["../semseg-pt_v3m1_0_base.py"]

# Written by:
#   python -m pointcept.datasets.ribsegv2.preproc --preproc \
#       --bin-pred-path <s1_log>/recon-3d-binpred \
#       --save-path <s1_log>/pt_preproc-binpred \
#       --cache-root data/ribsegv2/pt_preproc
# Expect 649 entries; anything less means stage 1 found no foreground in some
# volume (`--min-points` reports which) and stage 2 would silently train on less
# data than intended.
cache_root = "exp/ribsegv2/semseg-pt_v3m1_0_base-2s1/pt_preproc-binpred"

data = dict(
    train=dict(cache_root=cache_root),
    val=dict(cache_root=cache_root),
    test=dict(
        cache_root=cache_root,
        # Scored on the foreground-only cloud here; the number that goes in the
        # Performance Table is the COMBINED one, from tools/ribsegv2/combine_2stage.py,
        # which scores over the full stage-1 point set instead.
        split="test",
    ),
)

test = dict(
    # tools/ribsegv2/combine_2stage.py merges these with stage 1's.
    save_pred=True,
)

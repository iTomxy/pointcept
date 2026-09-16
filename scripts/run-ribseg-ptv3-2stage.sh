#!/usr/bin/env bash
# Two-stage PTv3 on RibSegv2 -- the tuned recipe, both stages.
#
#   bash run-ribseg-ptv3-2stage.sh              # repetition 1
#   bash run-ribseg-ptv3-2stage.sh -i 2         # further repetitions
#   bash run-ribseg-ptv3-2stage.sh -S "combine" # partial re-run
#
# Every flag of run-2stage.sh is forwarded, so -g/-c/-o/-p work here too; see
# that script for the step list and the environment overrides.
set -e
trap '. $HOME/mail.sh "cmd error" "[`date`] `whoami`@`hostname`:`realpath ${BASH_SOURCE[0]}`, line $LINENO"' ERR TERM HUP

# PTv3 runs with enable_flash=False (it costs 0.98 pt at this grid size), and
# the non-flash attention path allocates in large contiguous blocks -- without
# this, fragmentation alone can push a batch that fits into an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Expect 649 volumes in the stage-2 cache: 660 minus the 11 in IGNORE_VOLUMES.
# Nothing asserts that: the producers publish a directory under its final name
# only once they finish, so a cache that is there is complete. This is just the
# figure to sanity-check a full run against.
bash run-2stage.sh -d ribsegv2 -m semseg-pt_v3m1_0_base "$@"

. $HOME/mail.sh "finish ${BASH_SOURCE[0]}" "Two-stage PTv3 pipeline completed."

#!/bin/bash
set -e
trap '. $HOME/mail.sh "cmd error" "[`date`] `whoami`@`hostname`:`realpath ${BASH_SOURCE[0]}`, line $LINENO"' ERR TERM HUP # INT

# The non-flash attention path allocates in large contiguous blocks; without
# this, fragmentation alone can push a batch that fits into an OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dset=cbai_hip
cfg=semseg-pt_v3m1_0_base
exp=${cfg}
cfg_f=configs/$dset/${cfg}.py
py=${PY:-/opt/conda/bin/python}
ngpu=${NGPU:-4}

# Tune batch size and lr (with range test)
tune_dir=exp/$dset/$exp
mkdir -p "$tune_dir"
bs_json=$tune_dir/tune-batch_size.json
lr_json=$tune_dir/tune-lr_range.json
if [ "${RETUNE:-0}" = "1" ]; then
    rm -f "$bs_json" "$lr_json"
fi

# 1) Largest per-GPU batch that fits, measured on a real fwd+bwd against the
#    worst batch the pipeline can build (batch_size x max_points points).
if [ ! -f "$bs_json" ]; then
    $py tools/find_batch_size.py \
        --config $cfg_f --num-gpus $ngpu --mem-frac 0.9 --out "$bs_json" \
        2>&1 | tee "$tune_dir/tune-batch_size.log"
fi
micro_bs=$($py -c "import json;print(json.load(open('$bs_json'))['batch_size_per_gpu'])")
bs=$($py -c "import json;print(json.load(open('$bs_json'))['batch_size'])")

# 2) LR range test at that batch size, accumulating to the full effective batch.
if [ ! -f "$lr_json" ]; then
    $py tools/lr_range_test.py \
        --config $cfg_f --micro-batch-size $micro_bs --batch-size $bs \
        --out "$lr_json" 2>&1 | tee "$tune_dir/tune-lr_range.log"
fi

# 3) Turn both into --options overrides. They land in the run's saved config.py,
#    so the checked-in config stays free of machine-specific numbers.
tuned=$($py tools/tuned_options.py \
    --config $cfg_f --bs-json "$bs_json" --lr-json "$lr_json" --lr-scale 1.0)
echo "tuned overrides: $tuned"


bash scripts/train.sh -g $ngpu -d $dset -c $cfg -n $exp -p $py -r true $tuned \
    2> error.${0%.*}-train.log


bash scripts/test.sh -g 1 -d $dset -n $exp -p $py \
    -w model_best -o test.save_pred=True \
    2> error.${0%.*}-test.log


bash scripts/test.sh -g 1 -d $dset -n $exp -p $py \
    -w model_best -o test.save_pred=False -o data.test.split=train \
    2> error.${0%.*}-test_trainset.log


. $HOME/mail.sh "cbai_hip done" "[`date`] `whoami`@`hostname`:`realpath ${BASH_SOURCE[0]}`, line $LINENO"

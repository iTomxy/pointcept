#!/bin/bash
set -e
# trap '. $HOME/mail.sh "cmd error" "[`date`] `whoami`@`hostname`:`realpath $0`, $LINENO"' ERR TERM HUP # INT
# . $HOME/ld_lib_path.sh
# export CUDA_HOME=/usr/local/cuda-11.8

dset=ribsegv2
cfg=semseg-pt_v3m1_0_base
exp=${cfg}
cfg_f=configs/$dset/${cfg}.py


bash scripts/train.sh -g 2 -d $dset -c $cfg -n $exp -p /opt/conda/bin/python -r true \
    2> error.${0%.*}-train.log


bash scripts/test.sh -g 1 -d $dset -n $exp -p /opt/conda/bin/python \
    -w model_best -o test.save_pred=True -o data.test.add_trainval_incomplete=False \
    2> error.${0%.*}-test.log


bash scripts/test.sh -g 1 -d $dset -n $exp -p /opt/conda/bin/python \
    -w model_best -o test.save_pred=False -o data.test.add_trainval_incomplete=False -o data.test.split=train \
    2> error.${0%.*}-test_trainset.log


# . $HOME/mail.sh "cmd done" "[`date`] `whoami`@`hostname`:`realpath $0`, $LINENO"

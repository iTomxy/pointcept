#!/bin/bash
set -e
# trap '. $HOME/mail.sh "cmd error" "[`date`] `whoami`@`hostname`:`realpath $0`, $LINENO"' ERR TERM HUP # INT
# . $HOME/ld_lib_path.sh
# export CUDA_HOME=/usr/local/cuda-11.8

dset=ribsegv2
# cfg=insseg-pointgroup-v1m1-0-spunet-base
# cfg=insseg-pointgroup-cl-fg
# cfg=semseg-pointgroup-fg
# cfg=semseg-dgcnn
cfg=clreg-dgcnn
exp=${cfg}
cfg_f=configs/$dset/${cfg}.py


. scripts/train.sh -g 2 -d $dset -c $cfg -n $exp -p /opt/conda/bin/python \
    2> error.${0%.*}.log


. scripts/test.sh -g 2 -d $dset -n $exp -p /opt/conda/bin/python \
    -w model_best \
    2> error.${0%.*}.log


# . $HOME/mail.sh "cmd done" "[`date`] `whoami`@`hostname`:`realpath $0`, $LINENO"

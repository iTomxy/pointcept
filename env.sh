#!/bin/bash
set -e
trap 'echo [`date`] $LINENO > error.${0%.*}.log' ERR # TERM HUP INT

CONDA_P=${1-"$HOME/miniconda3"}
CONDA_BIN=$CONDA_P/bin
ENV=pointcept
if [ ! -d $CONDA_P/envs/$ENV ]; then
    $CONDA_BIN/conda create --name $ENV python=3.11 -y
fi
ENV_BIN=$CONDA_P/envs/$ENV/bin
export PATH=$ENV_BIN:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

$ENV_BIN/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$ENV_BIN/pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
$ENV_BIN/pip install torch_geometric torchelastic

$ENV_BIN/pip install spconv-cu128
https://github.com/traveller59/spconv?tab=readme-ov-file#linux
$ENV_BIN/pip uninstall -y cumm spconv
if [ ! -d cumm ]; then git clone https://github.com/FindDefinition/cumm; fi
if [ ! -d spconv ]; then git clone https://github.com/traveller59/spconv; fi
cd cumm
git checkout tags/v0.7.13 # spconv-2.3.8 requires this (see its setup.py)
$ENV_BIN/pip install -e .
$ENV_BIN/python -c 'import cumm'
cd ../spconv
$ENV_BIN/pip install -e .
cd ..
$ENV_BIN/python -c 'import spconv'

$ENV_BIN/pip install nibabel itk simpleitk medpy
$ENV_BIN/pip install tensorboard opencv-python plyfile camtools open3d wandb yapf
conda install -n $ENV -y scikit-learn scipy matplotlib pyyaml scikit-image jupyter pandas tqdm imageio

cd libs/pointops
$ENV_BIN/python setup.py install
cd ../pointgroup_ops
$ENV_BIN/python setup.py install
cd ../..

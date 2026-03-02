#!/bin/bash

echo soft-link Ribsegv2 data here

src=`realpath $HOME/data/ribseg`
dst=ribsegv2


echo link labels
dst_lab_p=$dst/label
if [ ! -d $dst_lab_p ]; then
    mkdir -p $dst_lab_p
fi
src_lab_p=$src/ribseg_v2/seg
for f in `ls $src_lab_p`; do
    ln -s $src_lab_p/$f $dst_lab_p/$f
done


echo link centre lines
dst_lab_p=$dst/centreline
if [ ! -d $dst_lab_p ]; then
    mkdir -p $dst_lab_p
fi
src_lab_p=$src/ribseg_v2/cl
for f in `ls $src_lab_p`; do
    ln -s $src_lab_p/$f $dst_lab_p/$f
done


echo link images
dst_img_p=$dst/image
if [ ! -d $dst_img_p ]; then
    mkdir -p $dst_img_p
fi
src_img_p=$src/ribfrac
for subp in \
    ribfrac-train-images-1/Part1 \
    ribfrac-train-images-2/Part2 \
    ribfrac-val-images \
    ribfrac-test-images;
do
    echo $subp
    for f in `ls $src_img_p/$subp`; do
        ln -s $src_img_p/$subp/$f $dst_img_p/$f
    done
done


echo preprocessed image \& label
dname=pt_preproc
ln -s $src/$dname $dst/$dname


echo link binary fg-bg prediction
bp_p=$HOME/codes/tmp.ptcloud/log/ribsegv2/semseg-dgcnn-bin-rndapply/recon-3d-binpred
# bp_p=../exp/ribsegv2/semseg-pt_v3m1_0_base-bin/recon-3d-binpred
ln -s `realpath $bp_p` $dst/binpred


cd $dst
# ln -s `realpath $HOME/codes/ribsegv2/segmentation/data/ribsegv2/ribsegv2-fg-stat.json`
ln -s `realpath $HOME/codes/ribsegv2/segmentation/data/ribsegv2/ribsegv2-statistics.json`

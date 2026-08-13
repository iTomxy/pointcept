#!/bin/bash
# (12 Aug 2026) A hip dataset in Curvebeam AI's server.

root=/straxdata/point_cloud_dataset/dataset
dst=cbai_hip
if [ ! -d $dst ]; then mkdir -p $dst; fi
for d in imagesTr imagesTs labelsTr labelsTs; do
    ln -s $root/$d $dst/$d
done

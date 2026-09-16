#!/bin/bash
# launch an interative singularity environment

IMAGE=/share/`whoami`/pointcept.sif

singularity exec --nv $IMAGE bash

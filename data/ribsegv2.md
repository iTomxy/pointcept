Preparing the Ribseg v2 dataset for pointcloud-based rib segmentation.

1. Download data (*.nii.gz) following [Ribsegv2](https://github.com/HINTLab/RibSeg/tree/ribsegv2):
    - images (ct scans): [MICCAI 2020 RibFrac Challenge:
Rib Fracture Detection and Classification](https://ribfrac.grand-challenge.org/), join the challenge first.
    - labels: [ribseg_v2.zip](https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view) (Google Drive)

2. Data folder structure:
    ```
    <DATA_ROOT>/
    |- image/
    |  |- RibFrac<VOLUME_ID>-image.nii.gz
    `- label/
       |- RibFrac<VOLUME_ID>-rib-seg.nii.gz
    ```

3. Preprocess npz files for fast reading with [pointcept/datasets/ribsegv2/preproc.py](pointcept/datasets/ribsegv2/preproc.py):
    - `DATA_ROOT`: where you place the extracted data, see above.
    - `CACHE_ROOT`: where to put the preprocessed npz files.
    ```shell
    # one-stage
    python -m pointcept.datasets.ribsegv2.preproc --preproc --hu-thres 200 \
        --data-root $DATA_ROOT \
        --save-path $CACHE_ROOT
    ```

For 2-stage pipeline,
after predicting the whole dataset (train/val/test) with the trained stage-1 model,
reconstructing volumetric binary prediction and re-preprocess the npz for stage-2 is needed:
- already included in [scripts/run-2stage.sh](../scripts/run-2stage.sh).
- `BIN_PRED_PATH`: path to the stage-1 predictions (also *.npz, point-wise format).
- `BIN_RECON_PATH`: path to placed the reconstructed volumetric stage-1 binary prediction.
- also pass `CACHE_ROOT` (preprocessed npz for one-stage model) to reduce raw data loading time.
- `CACHE_ROOT_2`: where to save the re-preprocessed npz files for stage-2.
```shell
# reconstruct volumetric binary prediction (as sieving mask for stage-2)
python -m pointcept.datasets.ribsegv2.preproc --recon-3d-bin \
    --data-root $DATA_ROOT \
    --bin-pred-path $BIN_PRED_PATH \
    --bin-recon-path $BIN_RECON_PATH \
    --cache-root $CACHE_ROOT

# re-preprocess npz for stage-2
python -m pointcept.datasets.ribsegv2.preproc --preproc \
    --data-root $DATA_ROOT \
    --bin-pred-path $RECON_DIR \
    --save-path $CACHE_ROOT_2 \
    --cache-root $CACHE_ROOT \
    --min-points 1
```

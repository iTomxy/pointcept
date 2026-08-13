A hip dataset hosted by Curvebeam AI.

Raw data:
```
/straxdata/point_cloud_dataset/dataset/
|- imagesTr/        # training ct scans
|  |- <VOLUME_ID>_0000.nii.gz
|- imagesTs/        # test ct scans
|  |- <VOLUME_ID>_0000.nii.gz
|- labelsTr/        # training segmentation annotation volumes
|  |- <VOLUME_ID>.nii.gz
`- labelsTs/        # test segmentation annotation volumes
   |- <VOLUME_ID>.nii.gz
```

- preprocessed data: `<THIS_PROJECT>/data/cbai_hip/pt_preproc/<VOLUME_ID>.npz`
- class set: `{0: bg, 1: left femur, 2: right femur, 3: left hip, 4: right hip}`

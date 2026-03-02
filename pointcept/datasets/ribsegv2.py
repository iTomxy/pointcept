import os, json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, reorient_points, determine_reorient
# from pointcept.utils.cache import shared_dict

"""
instance segmentation of ribs.
Adapted from ./scannet.py/ScanNetDataset
"""

IGNORE_VOLUMES = (452, 485, 490)
IGNORE_VOLUMES += (
    # (17 Dec 2025, iTom) potential wrong label
    439, 462, 471, 487, 540,
    # (17 Dec 2025, iTom) too hard, seemingly unable to solve
    652,
    # (22 Jan 2026, iTom) too noisy, even binary seg cannot work well, ignore for now
    501, 507, 570, 589, 630, 653,
)
SPLITS = {
    "train": [x for x in range(1, 421) if x not in IGNORE_VOLUMES],
    "val": [x for x in range(421, 501) if x not in IGNORE_VOLUMES],
    "test": [x for x in range(501, 661) if x not in IGNORE_VOLUMES],
}
SPLITS["all"] = SPLITS["train"] + SPLITS["val"] + SPLITS["test"]
# MIN_HU = 200
INCOMPLETE_VOLS = set()
with open("data/ribsegv2/ribsegv2-statistics.json", "r") as f:
    for line in f:
        d = json.loads(line)
        if "vid" in d and len(d["class_set"]) < 24 + 1:
            INCOMPLETE_VOLS.add(int(d["vid"]))


@DATASETS.register_module()
class Ribsegv2Dataset(DefaultDataset):
    """
    Use `strength` to store HU value to avoid being ignored in GridSample. See
        ./transform.py/index_operator/"index_valid_keys"
    for details.
    """
    def get_data_list(self):
        return SPLITS[self.split]

    def get_data(self, idx):
        volume_id = self.data_list[idx % len(self.data_list)]
        return {
            "name": str(volume_id),
            "index_valid_keys": ["coord", "strength", "segment", "voxel_index"], # don't use tuple
            "intensity": os.path.join(self.data_root, "pt_preproc", "{}-image.nii.gz".format(volume_id)),
            "segment": os.path.join(self.data_root, "pt_preproc", "{}-label.nii.gz".format(volume_id)),
            "sieve_mask": os.path.join(self.data_root, "binpred", "{}.nii.gz".format(volume_id)),
        }

#
# Volume-wise Dataset
#

class Ribsegv2Volume(Dataset):
    def __init__(self,
        volume_id,
        preproc_transform=None, # config dict, for volume reading & preprocessing
        transform=None, # config dict, remaining augmentations
        npoints=15000,
        data_root='data/ribsegv2',
        drop_last_thres=0, # drop the last batch if it has less #points than this threshold
    ):
        super(Ribsegv2Volume, self).__init__()
        self.volume_id = volume_id
        self.transform = Compose(transform)
        self.npoints = npoints
        self.data_dict = Compose(preproc_transform)(dict(
            intensity=os.path.join(data_root, "pt_preproc", "{}-image.nii.gz".format(volume_id)),
            segment=os.path.join(data_root, "pt_preproc", "{}-label.nii.gz".format(volume_id)),
            sieve_mask=os.path.join(data_root, "binpred", "{}.nii.gz".format(volume_id)),
            index_valid_keys=["coord", "strength", "segment", "voxel_index"], # don't use tuple
        ))
        coord = self.data_dict["coord"]
        shuffle_indices = np.random.permutation(coord.shape[0])
        if shuffle_indices.shape[0] % npoints != 0:
            n_left = shuffle_indices.shape[0] % npoints
            if n_left >= drop_last_thres or shuffle_indices.shape[0] < npoints:
                n_pad = npoints - n_left
                pad_indices = np.random.choice(coord.shape[0], n_pad, replace=True)
                shuffle_indices = np.concatenate((shuffle_indices, pad_indices), axis=0)

        assert shuffle_indices.shape[0] > 0, "Empty volume {}: coord.shape = {}, len(shuffle_indices) = {}, args: {}".format(
            volume_id, coord.shape, shuffle_indices.shape, json.dumps(dict(
                volume_id=volume_id,
                preproc_transform=preproc_transform,
                transform=transform,
                npoints=npoints,
                data_root=data_root,
                drop_last_thres=drop_last_thres
            ))
        )

        self.shuffle_indices = shuffle_indices
        self._len = max(1, len(self.shuffle_indices) // self.npoints)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        batch_idx = self.shuffle_indices[index * self.npoints: (index + 1) * self.npoints]
        return self.transform({"sample_idx": batch_idx, **self.data_dict})

#
# Loader of Volume-wise Dataset (for testint)
#

@DATASETS.register_module()
class Ribsegv2VolumeLoader:
    DATASET_CLASSES = {c.__name__: c for c in [
        Ribsegv2Volume,
    ]}

    def __init__(self, split, dataset_cls, **kwargs):
        """kwargs: same as `Ribsegv2Volume` except for `volume_id`"""
        if kwargs.pop("test_all_incomplete", False):
            print("Test with all incomplete volumes from train, val and test set.")
            id_list = list(INCOMPLETE_VOLS)
        else:
            id_list = SPLITS[split]

        if kwargs.pop("add_trainval_incomplete", False):
            print("Test with test volumes (complete & incomplete) + incomplete volumes from train & val.")
            id_list = list(set(id_list).union(INCOMPLETE_VOLS))

        self.id_list = [x for x in id_list if x not in IGNORE_VOLUMES]
        self.kwargs = kwargs
        self.dataset_cls = self.DATASET_CLASSES[dataset_cls]

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, volume_id):
        """can beyond `self.id_list`"""
        return self.dataset_cls(volume_id, **self.kwargs)

    def __iter__(self):
        for vid in self.id_list:
            yield self[vid]

#
# pre-process
#

def recon_3d_bin_pred(pred_path, save_path, pred_ornt="LPS", data_root="data/ribsegv2", preproc=True):
    """reconstruct the 1st stage binary segmentation to 3D volumes for all data
    The orientation should be consistent with their original image.
    Reconstructed voxel value: {-1: not predicted, 0: bg, 1: fg}
    Args:
        pred_path: str, path to the original fg-bg prediction (.npz)
        save_path: str, path to save the reconstructed volumes (.nii.gz)
        pred_ornt: str = "LPS", orientation of the predicted point clouds
        data_root: str = "data/ribsegv2"
        preproc: bool = True, use preprocessed data (see `preprocess`) or not
    """
    import open3d as o3d

    os.makedirs(save_path, exist_ok=True)
    for f in os.listdir(pred_path):
        if not f.endswith(".npz"):
            continue
        vid = int(f[:-4])
        print(vid, end='\r')
        if preproc:
            img_f = os.path.join(data_root, "pt_preproc", "{}-image.nii.gz".format(vid))
        else:
            img_f = os.path.join(data_root, "image", "RibFrac{}-image.nii.gz".format(vid))
        img_nii = nib.load(img_f)
        img_ornt = nib.aff2axcodes(img_nii.affine)

        data = np.load(os.path.join(pred_path, f))
        xyz = data["voxel_index"] # [n_batch, npt, 3] (or [n_batch, 3, npt])
        pred = data["pred"] # [n_batch, npt]
        # print(xyz.shape, pred.shape)
        assert 0 <= pred.min() and pred.max() <= 1, "Invalid binary prediction value range: [{}, {}]".format(pred.min(), pred.max())
        if pred.ndim > 1:
            pred = pred.reshape(-1)
            assert 3 == xyz.ndim, "voxel_index shape: {}".format(xyz.shape)
            if 3 == xyz.shape[1]: # [n_batch, 3, npt]
                xyz = xyz.transpose(0, 2, 1) # -> [n_batch, npt, 3]
            xyz = xyz.reshape(-1, xyz.shape[-1])

        need_reorient, axis_order_pred, _ = determine_reorient(img_ornt, pred_ornt)
        if need_reorient:
            # determine shape in predicted orientation <- used in reorient_points
            shape_pred = tuple(img_nii.shape[axis_order_pred[i]] for i in range(3))
            axes_range_pred = tuple((0, shape_pred[i]-1) for i in range(3))
            # reorient indices to original image orientation
            xyz = reorient_points(xyz, pred_ornt, img_ornt, axes_range_pred).astype(np.int32)

        # remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        xyz = xyz[ind]
        pred = pred[ind]

        # {-1: not predicted, 0: bg, 1: fg}
        pred_3d = np.zeros(img_nii.shape, dtype=np.int8) - 1
        pred_3d[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = pred.astype(np.int8)

        pred_nii = nib.Nifti1Image(pred_3d, img_nii.affine, img_nii.header)
        nib.save(pred_nii, os.path.join(save_path, "{}.nii.gz".format(vid)))


if "__main__" == __name__:
    recon_3d_bin_pred(
        "exp/ribsegv2/semseg-pt_v3m1_0_base-bin/result",
        "exp/ribsegv2/semseg-pt_v3m1_0_base-bin/recon-3d-binpred",
        preproc=True,
    )

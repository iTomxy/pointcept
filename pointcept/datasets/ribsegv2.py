import os, json
import numpy as np
import nibabel as nib
import torch
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import reorient_3dgrid, reorient_points
# from pointcept.utils.cache import shared_dict

"""
instance segmentation of ribs.
Adapted from ./scannet.py/ScanNetDataset
"""

# ignore index for both class & instance (?)
IGNORE_INDEX = -1


@DATASETS.register_module()
class Ribsegv2Dataset(DefaultDataset):
    """
    Use `strength` to store HU value to avoid being ignored in GridSample. See
        ./transform.py/index_operator/"index_valid_keys"
    for details.
    """
    class2id = np.asarray([0, 1]) # {0: background, 1: rib}
    ignore = [452, 485, 490]
    splits = {
        "train": [x for x in range(1, 421) if x not in (452, 485, 490)],
        "val": [x for x in range(421, 501) if x not in (452, 485, 490)],
        "test": [x for x in range(501, 661) if x not in (452, 485, 490)],
    }
    min_hu = 200

    def get_data_list(self):
        return self.splits[self.split]

    def get_data(self, idx):
        volume_id = self.data_list[idx % len(self.data_list)]
        image, label, cl = self.load_volume(volume_id)
        sieve_mask = self.sieve(image, label)
        # assert np.any(sieve_mask), "empty sieve_mask: {}".format(volume_id)
        segment = (label > 0).astype(np.int8)
        instance = np.where(label > 0, label, IGNORE_INDEX).astype(np.int8)
        data_dict = {
            "coord": np.argwhere(sieve_mask).astype(np.float32),
            "strength": image[sieve_mask][:, np.newaxis], # See ./transform.py/index_operator/"index_valid_keys".
            "segment": segment[sieve_mask],
            "instance": instance[sieve_mask],
            "skeleton": cl.astype(np.float32), # [c=24, n_cl_pt=500, 3]
        }
        # sampling = self.sample_index(data_dict["coord"].shape[0])
        # data_dict = {k: v[sampling] for k, v in data_dict.items()}
        data_dict["name"] = str(volume_id)
        return data_dict

    def sieve(self, image, label):
        return image > self.min_hu

    def load_volume(self, volume_id):
        fn_img = os.path.join(self.data_root, "image", "RibFrac{}-image.nii.gz".format(volume_id))
        fn_lab = os.path.join(self.data_root, "label", "RibFrac{}-rib-seg.nii.gz".format(volume_id))
        fn_cl = os.path.join(self.data_root, "centreline", "RibFrac{}.npz".format(volume_id))
        image_vol = nib.load(fn_img) # [H, W, L]
        label_vol = nib.load(fn_lab) # [H, W, L], in {0, ..., 24}
        cl = np.load(fn_cl)["cl"] # [c=24, n_cl_pt=500, 3]
        ori = nib.aff2axcodes(image_vol.affine)
        assert nib.aff2axcodes(label_vol.affine) == ori
        # spacing = tuple(map(float, image_vol.header.get_zooms())) # spacing in original axis order
        image = image_vol.get_fdata().astype(np.float32)
        shape = image.shape
        # image, spacing = reorient_3dgrid(image, ori, "LPS", spacing)
        image = reorient_3dgrid(image, ori, "LPS")
        label = reorient_3dgrid(label_vol.get_fdata().astype(np.uint8), ori, "LPS")
        cl = reorient_points(cl, ori, "LPS", ((0, shape[0]-1), (0, shape[1]-1), (0, shape[2]-1)))
        return image, label, cl#, spacing

    # def sample_index(self, n):
    #     """randomly sample self.npoints indices within `n`"""
    #     if n >= self.npoints:
    #         idx = np.random.choice(n, self.npoints, replace=False)
    #     elif n < self.npoints:
    #         idx = np.concatenate(
    #             np.arange(n),
    #             np.random.choice(n, self.npoints - n, replace=True)
    #         )

    #     return idx


@DATASETS.register_module()
class Ribsegv2DatasetFG(Ribsegv2Dataset):
    def __init__(self, bg_ratio=None, bg_ratio_rel_fg=None, *args, **kwargs):
        """fore-ground
        bg_ratio: None or float, use int(bg_ratio * n_bg_voxels) background voxels.
        bg_ratio_rel_fg: None or float: use int(bg_ratio_rel_fg * n_fg_voxels) background voxels.
        """
        self.bg_ratio = None if bg_ratio is None else max(0.0, min(bg_ratio, 1.0))
        self.bg_ratio_rel_fg = None if bg_ratio_rel_fg is None else max(0.0, min(bg_ratio_rel_fg, 1.0))
        super().__init__(*args, **kwargs)

    def sieve(self, image, label):
        """assume 0 is background"""
        hu_mask = image > self.min_hu
        fg_mask = label > 0
        sieve_mask = hu_mask & fg_mask
        if self.bg_ratio is not None:
            bg_mask = hu_mask & (0 == label)
            idx = np.where(bg_mask)
            n_bg = bg_mask.sum()
            n = int(self.bg_ratio * n_bg)
            shuffle_idx = np.random.permutation(n_bg)[: n]
            idx = tuple(_idx[shuffle_idx] for _idx in idx)
            sieve_mask[idx] = True # pack back these BG voxels
        elif self.bg_ratio_rel_fg is not None:
            bg_mask = hu_mask & (0 == label)
            idx = np.where(bg_mask)
            n = int(self.bg_ratio_rel_fg * sieve_mask.sum()) # relative to #{FG voxels}
            shuffle_idx = np.random.permutation(bg_mask.sum())[: n]
            idx = tuple(_idx[shuffle_idx] for _idx in idx)
            sieve_mask[idx] = True # pack back these BG voxels

        return sieve_mask

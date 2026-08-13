import os
import json

from torch.utils.data import Dataset

from ..builder import DATASETS
from ..defaults import DefaultDataset
from ..transform import Compose


def load_splits(data_root):
    """train/val/test vid lists from {data_root}/split.json, cached by
    `pointcept.datasets.cbai_hip.preproc.make_split`."""
    with open(os.path.join(data_root, "split.json")) as f:
        splits = json.load(f)
    splits["all"] = splits["train"] + splits["val"] + splits["test"]
    return splits


@DATASETS.register_module()
class CbaiHipDataset(DefaultDataset):
    """Semantic segmentation of hip CTs, on the cached point clouds.

    Each sample is one grid-subsampled volume loaded from the precomputed npz
    cache (`{data_root}/pt_preproc/{vid}.npz`, see `preproc.py`). Volume ids are
    DICOM UIDs (strings). The train/val/test assignment comes from
    `{data_root}/split.json`.

    Use `strength` to store HU value to avoid being ignored in GridSample. See
    `pointcept/datasets/transform.py` `index_operator`/"index_valid_keys".
    """

    def get_data_list(self):
        return load_splits(self.data_root)[self.split]

    def get_data(self, idx):
        vid = self.data_list[idx % len(self.data_list)]
        return {
            "name": vid,
            "index_valid_keys": ["coord", "strength", "segment", "voxel_index"],  # don't use tuple
            "npz": os.path.join(self.data_root, "pt_preproc", "{}.npz".format(vid)),
        }


@DATASETS.register_module()
class CbaiHipVolumeDataset(Dataset):
    """One item = one whole volume, for volume-wise testing.

    Pair it with a transform pipeline that grid-subsamples the volume and asks
    `GridSample` for `return_inverse=True`: the model then runs once over the
    grid points and `inverse` scatters that prediction back over all original
    points, so every point still gets a label. Stash the full-resolution label
    and voxel index with `Copy` into `origin_segment` / `origin_voxel_index`
    before `GridSample` (keys outside `index_valid_keys` survive the
    subsampling at full length).
    """
    INDEX_VALID_KEYS = ["coord", "strength", "segment", "voxel_index"]  # don't use tuple

    def __init__(self, split="test", data_root="data/cbai_hip", transform=None):
        super(CbaiHipVolumeDataset, self).__init__()
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)
        self.data_list = load_splits(data_root)[split]

    def get_data_list(self):
        return self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid = self.data_list[idx]
        data_dict = self.transform({
            "index_valid_keys": list(self.INDEX_VALID_KEYS),
            "npz": os.path.join(self.data_root, "pt_preproc", "{}.npz".format(vid)),
        })
        # after Collect, which keeps only the keys it was asked for
        data_dict["name"] = vid
        return data_dict

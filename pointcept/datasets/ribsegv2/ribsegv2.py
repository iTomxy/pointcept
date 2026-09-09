import os
from torch.utils.data import Dataset
from ..builder import DATASETS
from ..defaults import DefaultDataset
from ..transform import Compose
# from pointcept.utils.cache import shared_dict
from .base import IGNORE_VOLUMES, SPLITS, INCOMPLETE_VOLS

"""
instance segmentation of ribs.
Adapted from ./scannet.py/ScanNetDataset
"""

@DATASETS.register_module()
class Ribsegv2Dataset(DefaultDataset):
    """
    Use `strength` to store HU value to avoid being ignored in GridSample. See
        ./transform.py/index_operator/"index_valid_keys"
    for details.
    """
    def __init__(self, cache_root=None, bone_label_granularity=None, **kwargs):
        """
        cache_root: str = None, directory of preprocessed .npz volumes;
            defaults to {data_root}/pt_preproc. Point at
            <stage-1 log>/pt_preproc-binpred for a stage-2 run -- see
            preproc.preprocess_ptcloud.
        bone_label_granularity: str = None, optional ``coarse``/``fine``
            selector consumed by ``CombineBoneLabel``. Sample metadata keeps
            both TotalSegmentator phases on one inherited PTv3 pipeline.
        """
        # set before super().__init__, which calls self.get_data_list() at
        # the end and get_data() may run before this returns.
        self.cache_root = os.path.expanduser(cache_root) if cache_root else None
        self.bone_label_granularity = bone_label_granularity
        super(Ribsegv2Dataset, self).__init__(**kwargs)

    def get_data_list(self):
        return SPLITS[self.split]

    def get_data(self, idx):
        volume_id = self.data_list[idx % len(self.data_list)]
        cache_root = self.cache_root or os.path.join(self.data_root, "pt_preproc")
        data = {
            "name": str(volume_id),
            "index_valid_keys": ["coord", "strength", "segment", "voxel_index"], # don't use tuple
            # "intensity": os.path.join(self.data_root, "image", "RibFrac{}-image.nii.gz".format(volume_id)),
            # "segment": os.path.join(self.data_root, "label", "RibFrac{}-rib-seg.nii.gz".format(volume_id)),
            # "intensity": os.path.join(self.data_root, "preproc_crop", "{}-image.nii.gz".format(volume_id)),
            # "segment": os.path.join(self.data_root, "preproc_crop", "{}-label.nii.gz".format(volume_id)),
            "npz": os.path.join(cache_root, "{}.npz".format(volume_id)),
            # legacy of the online-sieving pipeline: the sieve is now applied offline by
            # preproc.preprocess_ptcloud(binpred_path=...), and under ReadNpz this key stays
            # a string that nothing consumes.
            # "sieve_mask": os.path.join(self.data_root, "binpred", "{}.nii.gz".format(volume_id)),
        }
        if self.bone_label_granularity is not None:
            data["bone_label_granularity"] = self.bone_label_granularity
        return data


@DATASETS.register_module()
class Ribsegv2VolumeDataset(Dataset):
    """One item = one whole volume, for volume-wise testing.

    Pair it with a transform pipeline that grid-subsamples the volume and asks
    `GridSample` for `return_inverse=True`: the model then runs once over the
    ~1e5 grid points and `inverse` scatters that prediction back over all ~2e6
    original points, so every point still gets a label. Stash the
    full-resolution label and voxel index with `Copy` into `origin_segment` /
    `origin_voxel_index` before `GridSample` (keys outside `index_valid_keys`
    survive the subsampling at full length).
    """
    INDEX_VALID_KEYS = ["coord", "strength", "segment", "voxel_index"] # don't use tuple

    def __init__(self,
        split="test",
        data_root="data/ribsegv2",
        transform=None,
        add_trainval_incomplete=False, # also test on incomplete volumes from train & val
        test_all_incomplete=False, # test on incomplete volumes only, from every split
        cache_root=None, # str, dir of preprocessed .npz volumes; see docstring below
        bone_label_granularity=None, # consumed by CombineBoneLabel
    ):
        """
        cache_root: str = None, directory of preprocessed .npz volumes; defaults
            to {data_root}/pt_preproc. Point at <stage-1 log>/pt_preproc-binpred
            for a stage-2 run -- see preproc.preprocess_ptcloud.
        """
        super(Ribsegv2VolumeDataset, self).__init__()
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)
        self.cache_root = os.path.expanduser(cache_root) if cache_root else os.path.join(data_root, "pt_preproc")
        self.bone_label_granularity = bone_label_granularity

        if test_all_incomplete:
            print("Test with all incomplete volumes from train, val and test set.")
            id_list = list(INCOMPLETE_VOLS)
        else:
            id_list = SPLITS[split]

        if add_trainval_incomplete:
            print("Test with test volumes (complete & incomplete) + incomplete volumes from train & val.")
            id_list = sorted(set(id_list).union(INCOMPLETE_VOLS))

        self.data_list = [x for x in id_list if x not in IGNORE_VOLUMES]

    def get_data_list(self):
        return self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        volume_id = self.data_list[idx]
        source = {
            "index_valid_keys": list(self.INDEX_VALID_KEYS),
            "npz": os.path.join(self.cache_root, "{}.npz".format(volume_id)),
            # legacy of the online-sieving pipeline: the sieve is now applied offline by
            # preproc.preprocess_ptcloud(binpred_path=...), and under ReadNpz this key stays
            # a string that nothing consumes.
            # "sieve_mask": os.path.join(self.data_root, "binpred", "{}.nii.gz".format(volume_id)),
        }
        if self.bone_label_granularity is not None:
            source["bone_label_granularity"] = self.bone_label_granularity
        data_dict = self.transform(source)
        # after Collect, which keeps only the keys it was asked for
        data_dict["name"] = str(volume_id)
        return data_dict

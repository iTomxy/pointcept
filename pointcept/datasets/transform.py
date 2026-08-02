"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random, os
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
# import scipy.stats
from scipy.ndimage import binary_dilation, generate_binary_structure
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
from scipy.spatial import cKDTree
import torch
import copy
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


def index_operator(data_dict, index, duplicate=False):
    """index selection operator for keys in `index_valid_keys`
    custom these keys by "Update" transform in config

    NOTE: This affects the point order! This can in turns affect
    for example the gt & predicted instance association in InsSeg testing.
    """
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


def determine_reorient(src_ornt, trg_ornt):
    """Determine whether a reorientation is needed, calculate the operation parameters if yes.
    Orientation = axis order + direction along each axis. Correspondence:
        - x: L, R
        - y: A, P
        - z: S, I
    Input:
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
    Output:
        need: bool, False if no operation is needed (i.e. already the wanted orientation)
        axis_order: int[3], in {0, 1, 2}, the new axis order
        flip_flags: bool[3], whether this axis should be flipped, same order as trg_ornt
    """
    src_ornt = tuple(s.upper() for s in src_ornt)
    trg_ornt = tuple(s.upper() for s in trg_ornt)
    to_axis = {
        'L': 'X', 'R': 'X', 'X': 'X', # x-axis
        'A': 'Y', 'P': 'Y', 'Y': 'Y', # y-axis
        'S': 'Z', 'I': 'Z', 'Z': 'Z', # z-axis
    }
    assert len(set([to_axis[s] for s in src_ornt])) == 3, "Duplicated axes in `src_ornt': {}".format(src_ornt)
    assert len(set([to_axis[t] for t in trg_ornt])) == 3, "Duplicated axes in `trg_ornt': {}".format(trg_ornt)
    flag_diff = False
    for s, t, in zip(src_ornt, trg_ornt):
        if to_axis[s] != to_axis[t] or ( # diff axis order
            s != t and t not in "XYZ"    # same axis order, but diff direction
        ):
            flag_diff = True
            break

    if not flag_diff:
        return False, None, None

    opposites = {'L': 'R', 'R': 'L', 'P': 'A', 'A': 'P', 'I': 'S', 'S': 'I'}
    axis_order = []
    flip_flags = []
    for t in trg_ornt:
        for src_axis, s in enumerate(src_ornt):
            if to_axis[s] == to_axis[t]:
                axis_order.append(src_axis)
                if t in "XYZ":
                    flip_flags.append(False)
                else: # t not in {X, Y, Z}
                    assert s not in "XYZ", "Cannot resolve: {} -> {}".format(s, t)
                    flip_flags.append(s == opposites[t])

                break # found corresponding axis pair

    # NOTE first perform axis re-ordering, then flip,
    # Because `flip_flags' is in the same order as `trg_ornt'.
    return True, axis_order, flip_flags


def reorient_3dgrid(vol, src_ornt, trg_ornt):
    """numpy-based reorientation, support using x, y, z to indicate axis order but keep direction
    Orientation = axis order + direction along each axis. Correspondence:
        - x: L, R
        - y: A, P
        - z: S, I
    Input:
        vol: np.ndarray, [H, W, L], original volume
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
    Output:
        result: np.ndarray, [H', W', L'], reoriented volume
    """
    flag_need, axis_order, flip_flags = determine_reorient(src_ornt, trg_ornt)
    if not flag_need:
        return vol

    # 1. Apply axis permutation
    result = np.transpose(vol, axis_order)
    # 2. Apply flips where needed
    for axis, flip in enumerate(flip_flags):
        if flip:
            result = np.flip(result, axis=axis)

    result = result.copy()  # ensure contiguous array
    return result


def reorient_points(points, src_ornt, trg_ornt, axes_range):
    """reorient a set of 3D points
    Input:
        points: [..., 3], numpy.ndarray
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
        axes_range: [(x1, x2), (y1, y2), (z1, z2)], range of point coordinates
            along each axis, all inclusive, in original axis order.
    Output:
        result: [..., 3], numpy.ndarray, reoriented points
    """
    flag_need, axis_order, flip_flags = determine_reorient(src_ornt, trg_ornt)
    if not flag_need:
        return points

    # 1. re-order axes
    result = points[..., axis_order]
    axes_range = [axes_range[i] for i in axis_order]
    # 2. flip axes
    for coord_axis, (flip, (_min, _max)) in enumerate(zip(flip_flags, axes_range)):
        if flip:
            result[..., coord_axis] = _min + _max - result[..., coord_axis]

    return result


def match_skeleton(points, skeleton):
    """match each point with its nearest skeleton (e.g. centre line of blood vessel) point
    Input:
        points: float[n, 3]
        skeleton: float[m, 3], points on the skeleton
    Output:
        matched_points: float[n, 3], nearest skeleton point for each input point
    """
    tree = cKDTree(skeleton)
    dist, idx = tree.query(points, k=1)
    return skeleton[idx]


def adjust_spacing(points, old_spacing, new_spacing):
    """(7 Oct 2025, iTom) NOT TESTED
    Adjust point coordinates for new spacing.

    Args:
        points: [n, 3] array of point coordinates
        old_spacing: [3] array of old spacing (z, y, x) or (h, w, l)
        new_spacing: [3] array of new spacing

    Returns:
        adjusted_points: [n, 3] array with adjusted coordinates
    """
    old_spacing = np.array(old_spacing)
    new_spacing = np.array(new_spacing)

    # Scale factor from old to new spacing
    scale_factor = old_spacing / new_spacing

    # Apply scaling
    adjusted_points = points * scale_factor

    return adjusted_points


def normalise_intensity(arr, clip_percentile=None):
    """clip extremes & normalise
    Args:
        arr: float numpy.ndarray
        clip_percentile: float[2] = None, in [0, 100], percentile to clip extreme values
    """
    if clip_percentile is not None:
        assert len(clip_percentile) == 2 and -0.01 < clip_percentile[0] < clip_percentile[1] < 100.01
        _min, _max = np.percentile(arr, [max(0, clip_percentile[0]), min(clip_percentile[1], 100)])
        arr = np.clip(arr, _min, _max) # truncate extreme values

    _mean, _std = np.mean(arr), np.std(arr)
    return (arr - _mean) / _std


def normalise_coord(points, centroid=None, radius=None, return_params=False):
    """normalise point cloud: (points - centroid) / radius
    Input:
        points: float[n, 3], numpy.ndarray
        centroid: float[3] = None
        radius: float = None
        return_params: bool = False, whether return `centroid` and `radius`
    """
    points = points.astype(np.float32)

    if centroid is None:
        centroid = np.mean(points, axis=0)
    else:
        centroid = np.asarray(centroid, dtype=np.float32)

    points = points - centroid

    if radius is None:
        radius = np.max(np.sqrt(np.sum(points**2, axis=1)))
    else:
        radius = float(radius)

    points = points / radius

    if return_params:
        return points, centroid, radius
    return points


@TRANSFORMS.register_module()
class BinarizeLabel:
    """binarize multi-class labels into foreground vs. background"""
    def __init__(self, coi):
        """
        coi: int|List[int], classes of interest, they will be set to 1, others to 0
        """
        self.coi = np.asarray([coi]).flatten()

    def __call__(self, data_dict):
        if "segment" in data_dict:
            data_dict["segment"] = np.isin(data_dict["segment"], self.coi).astype(np.uint8)
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift

        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class ConcatFeature:
    """concatenate several features (along axis-1) & put into the data dict
    Note: this is an in-place operation and may OVERWRITE the `dest_key` field if it already exists.
    """
    def __init__(self, src_keys, dest_key):
        """
        src_keys: List[str]: what features to concatenate
        dest_key: str, where to store the concatenated input
        """
        if isinstance(src_keys, str):
            src_keys = (src_keys,)
        self.src_keys = src_keys
        self.dest_key = dest_key

    def __call__(self, data_dict):
        """shape of features to be concatenated: [#points, #channels]"""
        feat_list = []
        for k in self.src_keys:
            feat_list.append(data_dict[k])

        if len(feat_list) == 1:
            comb_feat = feat_list[0]
        elif isinstance(feat_list[0], np.ndarray):
            comb_feat = np.concatenate([f.astype(np.float32) for f in feat_list], axis=1)
        else:
            assert isinstance(feat_list[0], torch.Tensor)
            comb_feat = torch.cat([f.float() for f in feat_list], dim=1)

        # NOTE this overwrites `dest_key` if it already exists!
        data_dict[self.dest_key] = comb_feat
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict


@TRANSFORMS.register_module()
class RandomPatchPoint:
    """randomly crop a 3D patch from a point cloud volume"""
    def __init__(self, patch_size, keys):
        """
        patch_size: float or float[3], patch size in three axes
        keys: List[str], crop what fields
        """
        assert isinstance(patch_size, (float, tuple, list))
        if isinstance(patch_size, float):
            patch_size = (patch_size,) * 3
        assert len(patch_size) == 3
        self.patch_size = patch_size

        if isinstance(keys, str):
            keys = [keys]
        if "coord" not in keys:
            keys.append("coord")
        self.keys = keys

    def __call__(self, data_dict):
        """
        coord: [npt, 3]
        segment: [npt]
        strength: [npt]
        """
        # choose one point as centroid
        npt, _ = data_dict["coord"].shape # [npt, 3]
        centroid = data_dict["coord"][np.random.randint(0, npt)]
        # calculate patch bbox around centroid
        _min = data_dict["coord"][:, :3].min(0) # [3]
        _max = data_dict["coord"][:, :3].max(0)
        volume_size = _max - _min
        patch_size = np.minimum(np.asarray(self.patch_size, dtype=data_dict["coord"].dtype), volume_size)

        patch_min = centroid - patch_size / 2
        patch_max = centroid + patch_size / 2

        # Shift the patch back into the volume if it crosses the boundary.
        patch_min = np.maximum(patch_min, _min)
        patch_max = patch_min + patch_size
        patch_max = np.minimum(patch_max, _max)
        patch_min = patch_max - patch_size

        coord = data_dict["coord"][:, :3]
        mask = np.ones(npt, dtype=bool)
        for axis in range(3):
            if np.isclose(patch_size[axis], volume_size[axis]):
                continue
            if axis < 2:
                mask &= (coord[:, axis] >= patch_min[axis]) & (coord[:, axis] < patch_max[axis])
            else:
                mask &= (coord[:, axis] >= patch_min[axis]) & (coord[:, axis] <= patch_max[axis])

        for key in self.keys:
            if key in data_dict:
                data_dict[key] = data_dict[key][mask]
        return data_dict


@TRANSFORMS.register_module()
class CT2PointCloud:
    """convert a CT scan (3D voxel grids) to point cloud
    This transform uses the `intensity` field to sieve voxels, assuming it is
    the raw UNnormalised HU value. But use it AFTER NormalizeIntensity, which
    can place the normalised intensity in `norm_intensity` and thus won't affect.
    """
    def __init__(self, hu_thres, coi=None, keys=["segment", "strength"], dilate_coi=None, dilate_iter=20):
        """
        hu_thres: float, HU threshold, only select voxels with intensity above
        coi: int|List[int] = None, classes of interest, if provided, only select voxels with class of interest
        keys: str|List[str] = ["segment"], apply the sieving to which field
        dilate_coi: List[int] = None, dilate the sieving mask around voxels of these classes
        dilate_iter: int = 20: dilate iteration
        """
        self.hu_thres = hu_thres
        if coi is not None:
            coi = np.asarray([coi]).flatten()
        self.coi = coi
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        # generate_binary_structure: `3` for 3D
        self.struct_3d = generate_binary_structure(3, 1) if dilate_coi is not None and dilate_iter > 0 else None
        self.dilate_coi = np.asarray([dilate_coi]).flatten() if dilate_coi is not None else None
        self.dilate_iter = dilate_iter

    def sieve(self, data_dict):
        """
        (potentially) used keys:
            intensity: float[H, W, L], sieve with HU thresholding
                In this codebase (Pointcept), I use `strength` as normalised intensity, `intensity` as the raw one.
            segment: int[H, W, L], if `coi` is specified, e.g. sieve only foreground voxels
            sieve_mask: int[H, W, L], specified sieving mask, e.g. fg-bg segmentation prediction from a model.
                Will keep only sieve_mask==1.
        """
        mask = data_dict["intensity"] > self.hu_thres
        if self.coi is not None:
            mask &= np.isin(data_dict["segment"], self.coi)

        if "sieve_mask" in data_dict and isinstance(data_dict["sieve_mask"], np.ndarray):
            assert data_dict["sieve_mask"].shape == mask.shape, \
                "Shape mismatch: data {} vs. sieve_mask {}".format(mask.shape, data_dict["sieve_mask"].shape)
            mask &= (1 == data_dict.pop("sieve_mask"))

        if self.struct_3d is not None:
            # dilate the mask to keep some surrounding non-bone voxels to help separate near-by bones
            mask_coi = np.isin(data_dict["segment"], self.dilate_coi)
            mask_coi = binary_dilation(mask_coi, structure=self.struct_3d, iterations=self.dilate_iter)
            mask |= mask_coi

        return mask

    def __call__(self, data_dict):
        """
        voxel_index: chosen voxels' 3D indecies (as initial points coordinates)
        """
        mask = self.sieve(data_dict)
        data_dict["voxel_index"] = np.argwhere(mask) # [npt, 3]
        for k in self.keys:
            if k in data_dict:
                data_dict[k] = data_dict[k][mask]
        return data_dict


@TRANSFORMS.register_module()
class CTDensityNoise(object):
    def __init__(self,
                 noise_std=10,  # Standard deviation in HU units
                 p=0.5):
        """
        Add Gaussian noise to CT intensity values (simulates scanner noise).

        Args:
            noise_std: Standard deviation of Gaussian noise in HU units
            p: Probability of applying the transform
        """
        self.noise_std = noise_std
        self.p = p

    def __call__(self, data_dict):
        if "strength" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.normal(
                0, self.noise_std, data_dict["strength"].shape
            ).astype(data_dict["strength"].dtype)
            data_dict["strength"] += noise
        return data_dict


@TRANSFORMS.register_module()
class CTIntensityVariation(object):
    def __init__(self,
                 intensity_shift_range=(-50, 50),  # HU units shift
                 intensity_scale_range=(0.95, 1.05),  # multiplicative scaling
                 gamma_range=(0.9, 1.1),  # gamma correction
                 p=0.8):
        """
        CT Intensity/Density variation for point clouds derived from CT scans.

        Args:
            intensity_shift_range: Additive shift in HU units (simulates different scanner calibration)
            intensity_scale_range: Multiplicative scaling (simulates different scanner gain)
            gamma_range: Gamma correction range (simulates different reconstruction kernels)
            p: Probability of applying the transform
        """
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, data_dict):
        if "strength" in data_dict.keys() and np.random.rand() < self.p:
            intensity = data_dict["strength"].copy()

            # Apply additive shift (simulates scanner calibration differences)
            if self.intensity_shift_range is not None:
                shift = np.random.uniform(
                    self.intensity_shift_range[0],
                    self.intensity_shift_range[1]
                )
                intensity += shift

            # Apply multiplicative scaling (simulates scanner gain differences)
            if self.intensity_scale_range is not None:
                scale = np.random.uniform(
                    self.intensity_scale_range[0],
                    self.intensity_scale_range[1]
                )
                intensity *= scale

            # Apply gamma correction (simulates different reconstruction kernels)
            if self.gamma_range is not None:
                gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
                # Normalize to [0,1], apply gamma, then scale back
                intensity_min, intensity_max = intensity.min(), intensity.max()
                if intensity_max > intensity_min:  # avoid division by zero
                    intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
                    intensity_norm = np.power(intensity_norm, gamma)
                    intensity = intensity_norm * (intensity_max - intensity_min) + intensity_min

            data_dict["strength"] = intensity

        return data_dict


@TRANSFORMS.register_module()
class DropRibPoint:
    """determinstic drop consecutive rib pairs from point cloud, used for controlled testing
    Adapted from RandomDropRibPoint, but fix drop depth, truncation direction and single mode.
    For the innest pair to drop, allow dropping only one rib of them.
    Rib class ID order is hard-coded (top-down, S->I):
    - left: 1 - 12
    - right: 13 - 24
    """
    def __init__(self, keys, drop_depth, begin_from='i', single='', min_npt=789):
        """
        Args:
            keys: List[str], fields (other than `label`) to apply on, e.g. coord, intensity.
            drop_depth: int, in [1, 11], maximum depth (num of consecutive pairs) to drop
            begin_from: str = 'i', in {'s', 'i'}, droppoing begins from which end along the IS-axis
                - 's': drop superior rib pairs, keep inferior ones
                - 'i': (opposite to 's')
            single: str = '',  in {'', 'l', 'r'}, whether to keep one rib from the innest pair to drop, while only dropping the other one.
                - '': drop the whole innest rib pair (keep none)
                - 'l': only drop the right rib in the innest pair to drop, keep the left one
                - 'r': drop left keep right
            min_npt: int = 789, minimum #points of left points after truncation.
                Skip the truncation if not enough points left.
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = set(keys)
        if "segment" not in self.keys:
            self.keys.add("segment")

        rib_pairs = tuple((i, i+12) for i in range(1, 12+1))
        drop_depth = int(drop_depth)
        assert 1 <= drop_depth <= len(rib_pairs) - 1

        single = single.lower()
        assert single in ('', 'l', 'r'), "{}: Unsupport `single`: expect {}, got {}".format(
            self.__class__.__name__, ('', 'l', 'r'), single
        )

        self.begin_from = begin_from.lower()
        if 's' == self.begin_from: # from S (top) to I (bottom)
            self.keep_ribs = np.asarray(rib_pairs[drop_depth: ])
            if 'l' == single: # keep left
                self.keep_ribs = np.append(self.keep_ribs, min(rib_pairs[drop_depth - 1])) # left <-> smaller id
            elif 'r' == single: # keep right
                self.keep_ribs = np.append(self.keep_ribs, max(rib_pairs[drop_depth - 1])) # right <-> bigger id
        elif 'i' == self.begin_from: # from I (bottom) to S (top)
            self.keep_ribs = np.asarray(rib_pairs[: - drop_depth])
            if 'l' == single: # keep left
                self.keep_ribs = np.append(self.keep_ribs, min(rib_pairs[- drop_depth]))
            elif 'r' == single: # keep right
                self.keep_ribs = np.append(self.keep_ribs, max(rib_pairs[- drop_depth]))
        else:
            raise ValueError("{}: Unsupport `begin_from`: expect {}, got {}".format(
                    self.__class__.__name__, ('s', 'i'), self.begin_from
                ))

        self.keep_ribs = np.append(self.keep_ribs, 0) # keep BG points
        self.min_npt = min_npt

    def __call__(self, data_dict):
        mask = np.isin(data_dict["segment"], self.keep_ribs)
        if mask.sum() >= self.min_npt:
            for k in self.keys:
                data_dict[k] = data_dict[k][mask]

        return data_dict


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class ExpandDims:
    """apply numpy.expand_dims"""
    def __init__(self, key_axes):
        """
        key_axes: List[(key<str>, axes<int|List[int]>)], how to expand dims on which key
        """
        assert isinstance(key_axes, (list, tuple))
        if isinstance(key_axes[0], str):
            key_axes = [key_axes]
        self.key_axes = key_axes

    def __call__(self, data_dict):
        for k, ax in self.key_axes:
            if k in data_dict:
                data_dict[k] = np.expand_dims(data_dict[k], axis=ax)

        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]

            data_dict = index_operator(data_dict, idx_unique)
            # (29 Sept 2025, iTom) also apply to matched_skeleton
            if "matched_skeleton" in data_dict:
                data_dict["matched_skeleton"] = data_dict["matched_skeleton"][idx_unique]

            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


@TRANSFORMS.register_module()
class LimitPoint:
    """randomly drop points only when the cloud exceeds `max_points`

    Unlike `SamplePoint`, a cloud smaller than the limit is left untouched
    instead of being padded with duplicated points. Place it after `GridSample`
    to keep the density uniform while capping memory. It re-orders points, so
    it invalidates a previously computed `inverse`.
    """
    def __init__(self, max_points):
        assert max_points is None or max_points > 0
        self.max_points = max_points

    def __call__(self, data_dict):
        if self.max_points is None:
            return data_dict

        assert "coord" in data_dict
        npt = data_dict["coord"].shape[0]
        if npt > self.max_points:
            idx = np.random.choice(npt, self.max_points, replace=False)
            data_dict = index_operator(data_dict, idx)

        return data_dict


@TRANSFORMS.register_module()
class MatchRibSkeleton:
    """Match rib instance points to their closest skeleton/centreline point.
    This relies on the original instance IDs before InstanceParser to correctly
    associate points with their respective skeletons. So copy `instance` to
    `origin_instance` BEFORE InstanceParser, which rearranges instances' ID.
    """
    def __call__(self, data_dict):
        """
        skeleton: float[n_cls, n_skeleton_pt, 3]. Background(0) is NOT included,
            so skeleton[0] is the centreline of rib1.
        origin_instance: int[npt], original instance ids (i.e. rib1-rib24) before InstanceParser.
        """
        assert "skeleton" in data_dict.keys()
        assert "origin_instance" in data_dict.keys(), \
            "Copy `instance` to `origin_instance` in advance before `InstanceParser."
        matched_skeleton = np.zeros_like(data_dict["coord"], dtype=np.float32) - 1.0
        for instance_id in np.unique(data_dict["origin_instance"]):
            if instance_id <= 0:
                continue
            mask = data_dict["origin_instance"] == instance_id
            if np.sum(mask) == 0:
                continue
            coord = data_dict["coord"][mask]
            skeleton = data_dict["skeleton"][instance_id - 1]  # rib1
            m_sk = match_skeleton(coord, skeleton)
            matched_skeleton[mask] = m_sk

        data_dict["matched_skeleton"] = matched_skeleton
        return data_dict


@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        center_height_scale=(0, 1),
        shared_global_view=False,
        view_keys=("coord", "origin_coord", "color", "normal"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        assert "coord" in view_keys

    def get_view(self, point, center, scale):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        size = int(np.random.uniform(*scale) * max_size)
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]

        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
        # get major global view
        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]
        # get global views: restrict the center of left global view within the major global view
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]

        global_views = [major_view] + global_views

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # reset cover mask if all points are sampled
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # augmentation and concat
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )
        for key in view_dict.keys():
            if "offset" not in key:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord:
    def __init__(self, centroid=None, radius=None):
        self.centroid = centroid
        self.radius = radius

    """will use `centroid` and `radius` to perform normalisation if provided in the data dict"""
    def __call__(self, data_dict):
        """data_dict["coord"]: [#points, 3]"""
        if "coord" in data_dict.keys():
            data_dict["coord"] = normalise_coord(
                data_dict["coord"],
                centroid=data_dict.get("centroid", self.centroid),
                radius=data_dict.get("radius", self.radius)
            )

        return data_dict


@TRANSFORMS.register_module()
class NormalizeIntensity(object):
    def __init__(self, clip_percentile=None, dest_key='strength'):
        """
        clip_percentile: float[2] = None, in [0, 100], percentile to clip extreme values
        dest_key: str = 'intensity', specify another if you want to keep the
            original `intensity` field intact.
        """
        self.clip_percentile = clip_percentile
        self.dest_key = dest_key

    def __call__(self, data_dict):
        if "intensity" in data_dict:
            data_dict[self.dest_key] = normalise_intensity(data_dict["intensity"], self.clip_percentile)

        return data_dict


@TRANSFORMS.register_module()
class NormalizeIntensityCached:
    """percentile-clipped z-score of the intensity, from PRE-COMPUTED statistics.

    Same result as `NormalizeIntensity(clip_percentile=...)` would give on the
    whole volume, but the four scalars come from the cache instead of being
    recomputed. That matters because the statistics must describe the *whole*
    volume: once the point cloud has been sieved to HU > threshold, they can no
    longer be derived from the points at hand. Caching four floats per volume
    rather than one float32 per point makes the cache ~22% smaller.

    Beware what this feature actually carries. The upper clip sits near the
    99.5th percentile of a mostly-air volume, i.e. ~500 HU, so 20-30% of the
    kept bone points saturate to one value and cortical density is erased.
    Within a volume the output spans ~0.2, while across volumes it spans ~2.5,
    so it encodes which scan a point came from more than which tissue it is.
    `WindowIntensity` is the better default; this exists to reproduce the old
    `norm_intensity` field for comparison.
    """
    def __init__(self, src_key="intensity", dest_key="strength",
                 stat_keys=("intensity_min", "intensity_max", "intensity_mean", "intensity_std")):
        """
        src_key: str = "intensity", field holding the raw HU value
        dest_key: str = "strength", where to store the normalised value
        stat_keys: str[4], data dict keys holding (clip_low, clip_high, mean, std)
            of the clipped volume, as written by `preprocess_ptcloud`
        """
        assert len(stat_keys) == 4
        self.src_key = src_key
        self.dest_key = dest_key
        self.stat_keys = tuple(stat_keys)

    def __call__(self, data_dict):
        if self.src_key not in data_dict:
            return data_dict

        missing = [k for k in self.stat_keys if k not in data_dict]
        assert not missing, "{}: missing cached statistics {}".format(self.__class__.__name__, missing)
        lo, hi, mean, std = (float(data_dict[k]) for k in self.stat_keys)
        v = np.clip(data_dict[self.src_key].astype(np.float32), lo, hi)
        data_dict[self.dest_key] = ((v - mean) / max(std, 1e-6)).astype(np.float32)
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class RandomApply:
    """randomly choose one candidate augmentation & apply"""
    def __init__(self, cfgs):
        self.transforms = []
        for t_cfg in cfgs:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        return random.choice(self.transforms)(data_dict)


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


@TRANSFORMS.register_module()
class RandomDropRibPoint:
    """randomly drop consecutive rib pairs from point cloud
    For the innest pair to drop, allow dropping only one rib of them.
    Rib class ID order is hard-coded (top-down, S->I):
    - left: 1 - 12
    - right: 13 - 24
    """
    def __init__(self, keys, max_drop_depth, begin_from='', allow_single=True, p=0.5, min_npt=789):
        """
        Args:
            keys: List[str], fields (other than `label`) to apply on, e.g. coord, intensity.
            max_drop_depth: int, in [1, 11], maximum depth (num of consecutive pairs) to drop
            begin_from: str = '', in {'', 's', 'i'}, droppoing begins from which end along the IS-axis
                - 's': drop superior rib pairs, keep inferior ones
                - 'i': (opposite to 's')
                - '': randomly chosen from {'s', 'i'}
            allow_single: bool = True, whether allow dropping only one rib from
                the innest pair to drop, while lefting the other one.
            p: float = 0.5, in [0, 1], application probability
            min_npt: int = 789, minimum #points of left points after truncation.
                Skip the truncation if not enough points left.
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = set(keys)
        if "segment" not in self.keys:
            self.keys.add("segment")
        self.rib_pairs = tuple((i, i+12) for i in range(1, 12+1))
        self.max_drop_depth = max(1, min(int(max_drop_depth), len(self.rib_pairs) - 1))
        self.begin_from = begin_from.lower()
        valid_bf = (
            's', # from S (top) to I (bottom)
            'i', # from I (bottom) to S (top)
            '',  # both
        )
        assert self.begin_from in valid_bf, "{}: Unsupport `begin_from`: expect {}, got {}".format(
            self.__class__.__name__, valid_bf, self.begin_from
        )
        self.allow_single = allow_single
        self.p = max(0, min(p, 1))
        self.min_npt = min_npt

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict

        nc = np.unique(data_dict["segment"]).size - 1 # `-1` excludes bg
        n_p = (nc + 1) // 2 # num of existing rib pairs
        if n_p < 2:
            return data_dict

        n_drop = random.randint(1, min(n_p - 1, self.max_drop_depth)) # closed interval
        rp = [t for t in self.rib_pairs]
        bf = self.begin_from if '' != self.begin_from else random.choice('si')
        if 's' == bf:
            innest_pair = rp[n_drop - 1]
            rp = rp[n_drop: ]
        else:
            innest_pair = rp[- n_drop]
            rp = rp[: - n_drop]

        cs_keep = np.asarray(rp).flatten()
        cs_keep = np.append(cs_keep, 0) # keep BG points
        if self.allow_single and random.random() < 0.5:
            cs_keep = np.append(cs_keep, random.choice(innest_pair))

        mask = np.isin(data_dict["segment"], cs_keep)
        if mask.sum() >= self.min_npt:
            for k in self.keys:
                data_dict[k] = data_dict[k][mask]

        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale

        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class RandomTruncateRibPoint:
    """random truncation specific for rib point cloud volume
    It truncates at rib upper/lower boundary (so that totally missing) or middle (so shape incomplete).
    Rib class ID order is hard-coded (top-down, S->I):
    - left: 1 - 12
    - right: 13 - 24
    Also, in RibSegV2 dataset, after converting coordinate to physical space (see ToPhysicalCoord),
    greater z-axis value corresponds to superior (upper), and smaller to inferior (lower).
    """
    def __init__(self, keys, max_drop_depth, begin_from='', pos='', p=0.5, is_axis=2, min_npt=789):
        """
        Args:
            keys: List[str], fields (other than `label`) to apply on, e.g. coord, intensity.
            max_drop_depth: int, in [1, 11], maximum depth (num of consecutive pairs) to drop
            begin_from: str = '', in {'', 's', 'i'}, droppoing begins from which end along the IS-axis
                - 's': truncate superior part, keep inferior part
                - 'i': (opposite to 's')
                - '': randomly chosen from {'s', 'i'}
            pos: str = '', in {'', 'b', 'm'}, truncation position modes
                - 'b': at boundary (top or bottom, dependent on truncation direction `begin_from`)
                - 'm': at rib middle
                - '': randomly chosen from {'b', 'm'}
            p: float = 0.5, in [0, 1], application probability
            is_axis: int = 2, which axis of `coord` is the position on the IS-axis (z-axis)
                The default 2 assumes the first 3 channels of `coord` are coordinates and
                the 3rd is its position on the IS-axis.
            min_npt: int = 789, minimum #points of left points after truncation.
                Skip the truncation if not enough points left.
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = set(keys)
        if "segment" not in self.keys:
            self.keys.add("segment")

        self.rib_pairs = tuple((i, i+12) for i in range(1, 12+1))
        self.max_drop_depth = max(1, min(int(max_drop_depth), len(self.rib_pairs) - 1))

        self.begin_from = begin_from.lower()
        valid_bf = (
            's', # from S (top) to I (bottom)
            'i', # from I (bottom) to S (top)
            '',  # both
        )
        assert self.begin_from in valid_bf, "{}: Unsupport `begin_from`: expect {}, got {}".format(
            self.__class__.__name__, valid_bf, self.begin_from
        )

        self.pos = pos.lower()
        valid_pos = (
            'b', # truncate at rib boundary
            'm', # at middle
            '',  # both
        )
        assert self.pos in valid_pos, "{}: Unsupport `pos`: expect {}, got {}".format(
            self.__class__.__name__, valid_pos, self.pos
        )

        self.p = max(0, min(p, 1))
        self.is_axis = is_axis
        self.min_npt = min_npt

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict

        # randomly choose an anchor rib, with respective to which the truncation applies
        nc = np.unique(data_dict["segment"]).size - 1 # `-1` excludes bg
        n_p = (nc + 1) // 2 # num of existing rib pairs
        if n_p < 2:
            return data_dict

        n_drop = random.randint(1, min(n_p - 1, self.max_drop_depth)) # closed interval
        anchor_pairs = [t for t in self.rib_pairs]
        bf = self.begin_from if '' != self.begin_from else random.choice('si')
        if 's' == bf:
            anchor_pairs = anchor_pairs[: n_drop]
        else:
            anchor_pairs = anchor_pairs[- n_drop: ]

        anchor_ribs = np.intersect1d(
            np.asarray(anchor_pairs).flatten(),
            np.unique(data_dict["segment"])
        )
        if anchor_ribs.size < 1:
            return data_dict

        anchor_rib = np.random.choice(anchor_ribs)
        # assumes 1st 3 channels are xyz, and 3rd is the position on IS-axis (z-axis)
        coord = data_dict["coord"]
        anchor_coord = coord[anchor_rib == data_dict["segment"]] # [m, 3+?]
        pos_mode = self.pos if '' != self.pos else random.choice("bm")
        if 'm' == pos_mode:
            cut_z = anchor_coord[:, self.is_axis].mean()
        elif 's' == bf: # truncate Superior, keep Inferior
            cut_z = anchor_coord[:, self.is_axis].min() # lower boundary to remove anchor rib
        else: # truncate Inferior, keep Superior
            cut_z = anchor_coord[:, self.is_axis].max() # upper boundary to remove anchor rib

        if 's' == bf: # truncate Superior (larger), keep Inferior (smaller)
            mask = coord[:, self.is_axis] < cut_z
        else: # truncate Inferior, keep Superior
            mask = coord[:, self.is_axis] > cut_z

        if mask.sum() >= self.min_npt:
            for k in self.keys:
                data_dict[k] = data_dict[k][mask]

        return data_dict


@TRANSFORMS.register_module()
class ReadNifti:
    """read .nii/.nii.gz data"""
    def __init__(self, keys=(("intensity", "f4"), ("segment", "i4")), meta_key=''):
        """
        keys: List[(path<str>, NumpyTypeCode<str>)] = (("intensity", "f4"), ("segment", "i4"))
            what fields are .nii files to be read, and should be what dtype.
        meta_key: str = '', find meta data (spacing, orientation) from the nifti
            corresponding to this key. Usually `intensity'.
        """
        self.keys = keys
        self.meta_key = meta_key

    def __call__(self, data_dict):
        for k, dt in self.keys:
            if k in data_dict and isinstance(data_dict[k], str):
                f = data_dict[k]
                assert os.path.isfile(f), "{}: No such file: {}".format(self.__class__.__name__, f)
                nii = nib.load(f)
                data_dict[k] = nii.get_fdata().astype(dt).copy()
                if k == self.meta_key:
                    data_dict["affine"] = nii.affine.copy() # for converting voxel indices to physical coordinates
                    data_dict["nifti_shape"] = nii.shape # for transforming affine at reorientation

        return data_dict


@TRANSFORMS.register_module()
class ReadNpz:
    """Load a precomputed point-cloud .npz cache.
    Replaces ReadNifti + NormalizeIntensity + CT2PointCloud. Expected npz keys:
    affine, label, intensity, voxel_index, nifti_shape and the
    intensity_{min,max,mean,std} scalars. Rebuild `coord` with ToPhysicalCoord.
    See ribsegv2/preproc.preprocess_ptcloud.
    `intensity` is a raw HU integer; turn it into a feature with `WindowIntensity`
    (fixed window) or `NormalizeIntensityCached` (cached percentile z-score),
    both of which cast to float themselves.
    """
    def __init__(self, path_key="npz", rename_keys={"label": "segment"}):
        """
        path_key: str = "npz", data_dict key that holds the .npz file path
        rename_keys: Dict[str, str] = None, rename keys in the loaded npz file
        """
        self.path_key = path_key
        self.rename_keys = rename_keys or {}

    def __call__(self, data_dict):
        path = data_dict.pop(self.path_key)
        assert os.path.isfile(path), "{}: No such file: {}".format(self.__class__.__name__, path)
        npz = np.load(path)

        d = {k: npz[k] for k in npz.files}
        for k_old, k_new in self.rename_keys.items():
            if k_old in d:
                d[k_new] = d.pop(k_old)
        data_dict.update(d)
        return data_dict


@TRANSFORMS.register_module()
class Reorient:
    """reorient a medical volume (3D voxel grids)"""
    def __init__(self, keys, new_ornt="LPS"):
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = keys
        self.new_ornt = tuple(t.upper() for t in new_ornt)

    def __call__(self, data_dict):
        old_ornt = tuple(s.upper() for s in nib.aff2axcodes(data_dict["affine"]))
        # whether spacing is 1) not available, or 2) already re-ordered
        # flag_spacing = "spacing" not in data_dict
        for k in self.keys:
            if k in data_dict:
                data_dict[k] = reorient_3dgrid(data_dict[k], old_ornt, self.new_ornt)

        # transform affine according
        if "affine" in data_dict and old_ornt != self.new_ornt:
            # Get the transform from original to target orientation
            transform = ornt_transform(axcodes2ornt(old_ornt), axcodes2ornt(self.new_ornt))
            # Create new affine for the transformed image
            orig_shape = data_dict.pop("nifti_shape")
            affine = nib.orientations.inv_ornt_aff(transform, orig_shape)
            data_dict["affine"] = np.dot(data_dict["affine"], affine)
            assert nib.aff2axcodes(data_dict["affine"]) == self.new_ornt, \
                "[{}] Orientation after transform {} does not match expection {}".format(
                    self.__class__.__name__, nib.aff2axcodes(data_dict["affine"]), self.new_ornt
                )
            # Reorientation can permute axes; retain the shape in the same
            # orientation as the arrays and voxel_index produced downstream.
            _, axis_order, _ = determine_reorient(old_ornt, self.new_ornt)
            data_dict["nifti_shape"] = tuple(orig_shape[i] for i in axis_order)

        # data_dict["orientation"] = self.new_ornt
        return data_dict


@TRANSFORMS.register_module()
class SamplePoint:
    """randomly sample a fix number of points from a volume"""
    def __init__(self, keys, npoints=0):
        """
        keys: List[str]: apply sampling to what keys in the data dict
        npoints: int = 0, sample how many points from a volume
        """
        self.npoints = npoints
        if isinstance(keys, str):
            keys = (keys,)
        assert len(keys) > 0
        self.keys = keys

    def __call__(self, data_dict):
        """
        Can use `sample_idx' to specify what points to sample. If not specified,
        fall back to random sampling. Useful in volume-wise testing where I want
        to loop over all points from the same volume.
        """
        idx = data_dict.pop("sample_idx", None)
        if idx is None:
            assert self.npoints > 0, "{}: Both `sample_idx' and `npoints' are not given".format(self.__class__.__name__)
        for k in self.keys:
            if k not in data_dict:
                continue
            v = data_dict[k]
            assert isinstance(v, np.ndarray), \
                "{} ({}) is not supported in SamplePoint (expect numpy.ndarray)".format(k, type(v))
            if idx is None:
                if v.shape[0] > self.npoints:
                    idx = np.random.choice(v.shape[0], self.npoints, replace=False)
                elif v.shape[0] < self.npoints:
                    idx = np.concatenate([
                        np.random.permutation(v.shape[0]),
                        np.random.choice(v.shape[0], self.npoints - v.shape[0], replace=True)
                    ])
                else:
                    idx = np.random.permutation(v.shape[0])

            data_dict[k] = v[idx]

        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            if np.issubdtype(data.dtype, np.unsignedinteger) and data.size > 0:
                # torch 1.x cannot construct tensors directly from uint16/32/64.
                # torch.from_numpy accepts only uint8 among the unsigned types,
                # but a cache narrowed by `np_smallest_dtype` holds uint16/uint32
                data = data.astype(np.int64)
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class ToPhysicalCoord:
    """convert voxel indeices to physical space coordinates"""
    def __call__(self, data_dict):
        # Backward compatibility for caches produced before affine replaced
        # the derived coord array.
        if "affine" not in data_dict:
            assert "coord" in data_dict, \
                "ToPhysicalCoord requires affine + voxel_index, or an existing coord"
            return data_dict
        data_dict["coord"] = nib.affines.apply_affine(
            data_dict["affine"],
            data_dict["voxel_index"].astype(np.float32)
        )
        return data_dict


@TRANSFORMS.register_module()
class Transpose:
    """numpy.transpose, torch.permute"""
    def __init__(self, keys, axes):
        """
        keys: List[str], apply on what fields
        axes: List[int], desired ordering of dimensions
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = keys
        self.axes = axes

    def __call__(self, data_dict):
        for k in self.keys:
            if isinstance(data_dict[k], np.ndarray):
                data_dict[k] = np.transpose(data_dict[k], self.axes)
            else:
                assert isinstance(data_dict[k], torch.Tensor)
                data_dict[k] = torch.permute(data_dict[k], self.axes)

        return data_dict


@TRANSFORMS.register_module()
class TruncateRibPoint:
    """determinstic truncation specific for rib point cloud volume, used for controlled testing
    Adapted from RandomTruncateRibPoint, but fix drop depth, truncation direction and position.
    It truncates at rib upper/lower boundary (so that totally missing) or middle (so shape incomplete).
    Rib class ID order is hard-coded (top-down, S->I):
    - left: 1 - 12
    - right: 13 - 24
    Also, in RibSegV2 dataset, after converting coordinate to physical space (see ToPhysicalCoord),
    greater z-axis value corresponds to superior (upper), and smaller to inferior (lower).
    """
    def __init__(self, keys, drop_depth, begin_from='i', pos='m', is_axis=2, min_npt=789):
        """
        Args:
            keys: List[str], fields (other than `label`) to apply on, e.g. coord, intensity.
            drop_depth: int, in [1, 11], maximum depth (num of consecutive pairs) to drop
            begin_from: str = 'i', in {'s', 'i'}, droppoing begins from which end along the IS-axis
                - 's': truncate superior part, keep inferior part
                - 'i': (opposite to 's')
            pos: str = 'm', in {'b', 'm'}, truncation position modes
                - 'b': at boundary (top or bottom, dependent on truncation direction `begin_from`)
                - 'm': at rib middle
            p: float = 0.5, in [0, 1], application probability
            is_axis: int = 2, which axis of `coord` is the position on the IS-axis (z-axis)
                The default 2 assumes the first 3 channels of `coord` are coordinates and
                the 3rd is its position on the IS-axis.
            min_npt: int = 789, minimum #points of left points after truncation.
                Skip the truncation if not enough points left.
        """
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = set(keys)
        if "segment" not in self.keys:
            self.keys.add("segment")

        rib_pairs = tuple((i, i+12) for i in range(1, 12+1))
        drop_depth = int(drop_depth)
        assert 1 <= drop_depth <= len(rib_pairs) - 1

        self.begin_from = begin_from.lower()
        if 's' == self.begin_from: # from S (top) to I (bottom)
            self.anchor_rib_pair = np.asarray(rib_pairs[drop_depth - 1])
            self.keep_ribs = np.asarray(rib_pairs[drop_depth: ]).flatten() # used for judgement in case anchor rib pair does not exist
        elif 'i' == self.begin_from: # from I (bottom) to S (top)
            self.anchor_rib_pair = np.asarray(rib_pairs[- drop_depth])
            self.keep_ribs = np.asarray(rib_pairs[: - drop_depth]).flatten() # used for judgement in case anchor rib pair does not exist
        else:
            raise ValueError("{}: Unsupport `begin_from`: expect {}, got {}".format(
                self.__class__.__name__, ('s', 'i'), self.begin_from
            ))

        self.keep_ribs = np.append(self.keep_ribs, 0) # keep BG points
        self.pos = pos.lower()
        valid_pos = (
            'b', # truncate at rib boundary
            'm', # at middle
        )
        assert self.pos in valid_pos, "{}: Unsupport `pos`: expect {}, got {}".format(
            self.__class__.__name__, valid_pos, self.pos
        )

        self.is_axis = is_axis
        self.min_npt = min_npt

    def __call__(self, data_dict):
        # assumes 1st 3 channels are xyz, and 3rd is the position on IS-axis (z-axis)
        coord = data_dict["coord"]
        anchor_mask = np.isin(data_dict["segment"], self.anchor_rib_pair)
        if not anchor_mask.any(): # anchor rib does not exist
            mask = np.isin(data_dict["segment"], self.keep_ribs)
        else:
            anchor_coord = coord[anchor_mask] # [m, 3+?]
            if 'm' == self.pos:
                cut_z = anchor_coord[:, self.is_axis].mean()
            elif 's' == self.begin_from: # truncate Superior, keep Inferior
                cut_z = anchor_coord[:, self.is_axis].min() # lower boundary to remove anchor rib
            else: # truncate Inferior, keep Superior
                cut_z = anchor_coord[:, self.is_axis].max() # upper boundary to remove anchor rib

            if 's' == self.begin_from: # truncate Superior (larger), keep Inferior (smaller)
                mask = coord[:, self.is_axis] < cut_z
            else: # truncate Inferior, keep Superior
                mask = coord[:, self.is_axis] > cut_z

        if mask.sum() >= self.min_npt:
            for k in self.keys:
                data_dict[k] = data_dict[k][mask]

        return data_dict


@TRANSFORMS.register_module()
class TypeCast:
    """cast numpy array to a specific dtype"""
    def __init__(self, key_type):
        """key_type: Dict[str, NumpyTypeCode], e.g. {"intensity": "f4", "label": "i4"}"""
        self.key_type = key_type

    def __call__(self, data_dict):
        for k, dt in self.key_type.items():
            if k in data_dict:
                data_dict[k] = data_dict[k].astype(dt)

        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class WindowIntensity:
    """map raw HU to [0, 1] through a FIXED window shared by every volume.

    Unlike `NormalizeIntensity`, the window is not derived from the volume being
    processed, so the same HU always maps to the same feature value. Per-volume
    percentile normalisation makes the feature depend on which other tissue
    happened to be in the field of view.
    """
    def __init__(self, window=(200.0, 1500.0), src_key="intensity", dest_key="strength"):
        """
        window: float[2], (low, high) HU bounds; values outside are clipped
        src_key: str = "intensity", field holding the raw HU value
        dest_key: str = "strength", where to store the windowed value
        """
        assert len(window) == 2 and window[0] < window[1]
        self.low, self.high = float(window[0]), float(window[1])
        self.src_key = src_key
        self.dest_key = dest_key

    def __call__(self, data_dict):
        if self.src_key in data_dict:
            v = np.clip(data_dict[self.src_key].astype(np.float32), self.low, self.high)
            data_dict[self.dest_key] = (v - self.low) / (self.high - self.low)

        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

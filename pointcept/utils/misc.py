"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os, math, re
import contextlib
import warnings
from collections import abc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.distributed as dist
from importlib import import_module


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def vis_confusion_matrix(conf_matrix, classes_name, save_file, title='Normalized Confusion Matrix Heatmap'):
    nc = len(classes_name)
    fig, ax = plt.subplots(figsize=(nc + 4, nc + 4))

    # Plot heatmap
    fmt = ".2f" if np.issubdtype(conf_matrix.dtype, np.floating) else "d"
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=classes_name, yticklabels=classes_name,
                square=True, cbar=False, ax=ax)

    for i in range(conf_matrix.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

    # Labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    # Save figure with transparent background
    plt.savefig(save_file, pad_inches=0.0, bbox_inches='tight')#, transparent=True)
    plt.close(fig)


@contextlib.contextmanager
def quiet_nan():
    """silence the 0/0 and all-NaN warnings raised by deliberately NaN-producing code

    Metrics use NaN to mark "not applicable to this sample" (e.g. a class absent
    from a volume) so it drops out of an average instead of counting as zero.
    Computing them therefore warns once per class per sample, which floods a log
    for no reason. Wrap those calls in this rather than muting numpy globally.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            yield


def nanmean(a, axis=None):
    """numpy.nanmean that returns NaN instead of warning on an all-NaN slice"""
    a = np.asarray(a, dtype=float)
    valid = ~np.isnan(a)
    total = np.where(valid, a, 0.0).sum(axis=axis)
    count = valid.sum(axis=axis)
    out = np.where(count > 0, total / np.clip(count, 1, None), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def to_jsonable(obj):
    """recursively cast numpy scalars & arrays to built-ins so json.dump cannot fail

    Worth a defensive pass whenever the payload is assembled from several
    helpers: numpy scalars (np.bool_ especially) are not JSON serialisable, and
    a TypeError at dump time throws away everything that produced the payload.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def calc_stat(lst, percentages=[], prec=None, scale=None):
    """list of statistics: median, mean, standard error, min, max, percentiles
    It can be useful when you want to know these statistics of a list and
    dump them in a json log/string.
    Input:
        lst: list of number
        percentages: List[float] = [], what percentiles (quantile) to cauculate
        prec: int|None = None, round to which decimal place if it is an int
        scale: int|float|None = None, scale the elements in `lst` if it is an int or float
            Use it when `lst` contains normalised number (i.e. in [0, 1]) and you want to
            present them in percentage (i.e. 0.xyz -> xy.z%)
    """
    if isinstance(scale, (int, float)):
        lst = list(map(lambda x: scale * x, lst))

    if not np.isfinite(np.asarray(lst, dtype=float)).any():
        # empty, or every entry is NaN: numpy would warn and return NaN anyway,
        # so answer with the same shape of dict quietly
        keys = ["min", "max", "mean", "std", "median"]
        # same clamping as the normal path below, so the keys match
        keys += ["p_{}".format(max(1e-7, min(p, 100 - 1e-7))) for p in percentages]
        return {k: math.nan for k in keys}

    ret = {
        "min": float(np.nanmin(lst)),
        "max": float(np.nanmax(lst)),
        "mean": float(np.nanmean(lst)),
        "std": float(np.nanstd(lst)),
        "median": float(np.nanmedian(lst))
    }
    if len(percentages) > 0:
        percentages = [max(1e-7, min(p, 100 - 1e-7)) for p in percentages]
        percentiles = np.nanpercentile(lst, percentages)
        for ptage, ptile in zip(percentages, percentiles):
            ret["p_{}".format(ptage)] = float(ptile)

    if isinstance(prec, int):
        ret = {k: round(v, prec) for k, v in ret.items()}

    return ret


def bootstrap_ci_mean_delta(delta, B=10000, alpha=0.05, seed=0):
    """Calculate the confidence interval of mean of delta/difference of two variables using bootstrap method.
    Args:
        delta: float[], the difference of two variables for each sample
        B: int = 10000, the number of bootstrap samples to draw
        alpha: float = 0.05, the confidence level is 1 - alpha
        seed: int = 0, random seed for reproducibility
    Returns:
        mean: float, the mean of delta
        lo: float, the lower bound of confidence interval
        hi: float, the upper bound of confidence interval
    """
    delta = np.asarray(delta, dtype=float)
    n = delta.size
    if 0 == n:
        return {
            'mean': float('nan'),
            'lb': float('nan'),
            'ub': float('nan'),
        }
    elif 1 == n:
        return {
            'mean': float(delta[0]),
            'lb': None, # use None to indicate degenerate case
            'ub': None,
        }

    rng = np.random.default_rng(seed)

    boot = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = delta[idx].mean()

    mean = delta.mean()
    lo, hi = np.quantile(boot, [alpha/2, 1 - alpha/2])
    return {
        'mean': float(mean),
        'lb': float(lo), # lower bound
        'ub': float(hi), # higher bound
    }


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def staging_dir(final_path, suffix=".tmp"):
    """Directory to build `final_path` in, so it only appears once complete.

    Half of the pair with `publish_dir`. A producer that writes through these
    two lets any consumer take the existence of `final_path` as proof the job
    finished, instead of counting entries and guessing at what the total should
    have been -- a guess only the producer can actually make.

    The staging directory is deliberately NOT cleared. Producers here skip
    outputs they already wrote, so an interrupted run resumes into it and costs
    only what it had not done; wiping it would turn every interruption into a
    full redo.
    Args:
        final_path: str, where the directory will end up
        suffix: str = ".tmp", appended to `final_path` to name the staging one.
            A sibling, so the rename in `publish_dir` stays within one filesystem.
    Returns:
        tmp_path: str, the staging directory, created if absent
    """
    tmp_path = final_path.rstrip(os.sep) + suffix
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def publish_dir(tmp_path, final_path):
    """Publish a staging directory under its final name. See `staging_dir`.

    `os.rename` within one filesystem is atomic, so `final_path` never exists
    half-written -- which is exactly what lets a consumer read its existence as
    completeness.

    When `final_path` already holds results the entries are moved across one by
    one instead. That happens when two runs deliberately share a directory --
    the volume-wise tester dumping a second split beside the first, say -- and
    it is NOT atomic, so do not race two producers onto one final path.
    Args:
        tmp_path: str, the staging directory, as returned by `staging_dir`
        final_path: str, where it should end up
    Returns:
        final_path: str
    """
    if not os.path.exists(final_path):
        os.rename(tmp_path, final_path)
        return final_path
    for name in os.listdir(tmp_path):
        os.replace(os.path.join(tmp_path, name), os.path.join(final_path, name))
    os.rmdir(tmp_path)
    return final_path


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass


def axcodes2dir(axcode):
    """Convert axcode string to direction cosine matrix.
    Ref: https://nipy.org/nibabel/reference/nibabel.orientations.html#nibabel.orientations.aff2axcodes
    Input:
        orientation: str|Tuple[char], e.g. "RAI", ('L', 'P', 'S')
    Output:
       direction: float[9]: serialised 3x3 direction matrix in row-major order
    """
    assert len(set(axcode)) == 3
    axis_map = {
        'R': [1, 0, 0], 'L': [-1, 0, 0],
        'A': [0, 1, 0], 'P': [0, -1, 0],
        'S': [0, 0, 1], 'I': [0, 0, -1]
    }
    direction = []
    for code in axcode:
        direction.extend(axis_map[code.upper()])

    return direction


def np2nifti(image, axcode, spacing=(1.0, 1.0, 1.0)):
    """Convert numpy array to nibabel.Nifti1Image.
    Input:
        image: [L, H, W], numpy array
        axcode: str[3], orientation of `image', e.g. "RAI"
        spacing: float[3], spacing of each axis
    Output:
        nibabel.Nifti1Image
    """
    affine = np.eye(4)
    direction = axcodes2dir(axcode)
    direction_matrix = np.array(direction).reshape(3, 3)
    for i in range(3):
        affine[:3, i] = direction_matrix[:, i] * spacing[i]

    return nib.Nifti1Image(image, affine=affine)


def np_smallest_dtype(arr, return_dtype=False):
    """decide the smallest suitable numpy integer dtype for an integer array
    Args:
        arr: numpy.ndarray of integer dtype
        return_dtype: bool = False, return the chosen dtype. If False, cast
            the input array to the chosen dtype and return it.
    Returns:
        if return_dtype:
            dtype: the smallest numpy ingeter dtype that suits the input array
        else:
            arr: the input array cast to the chosen dtype
    """
    assert np.issubdtype(arr.dtype, np.integer), 'Expect array with integer dtype, got {}'.format(arr.dtype)
    if 0 == arr.size:
        return np.uint8 if return_dtype else arr.astype(np.uint8)

    min_val = np.min(arr)
    max_val = np.max(arr)
    type_list = [np.uint8, np.uint16, np.uint32, np.uint64]
    if min_val < 0:
        type_list = [np.int8, np.int16, np.int32, np.int64]

    for d_type in type_list:
        if np.iinfo(d_type).min <= min_val and np.iinfo(d_type).max >= max_val:
            return d_type if return_dtype else arr.astype(d_type)

    raise ValueError('Could not find a dtype for the array.')


def to_dict(ed):
    """convert dict-like object (e.g. easydict.EasyDict, addict.Dict) to built-in dict for clean yaml"""
    d = {}
    for k, v in ed.items():
        if isinstance(v, dict): # EasyDict is also dict
            d[k] = to_dict(v)
        elif isinstance(v, (tuple, list)):
            d[k] = [to_dict(_v) if isinstance(_v, dict) else _v for _v in v]
        else:
            d[k] = v
    return d


def natural_sort_key(s, num_pattern=re.compile('([0-9]+)'), lower=False):
    """https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort"""
    if lower:
        return [int(text) if text.isdigit() else text.lower()
                for text in num_pattern.split(s)]
    else:
        return [int(text) if text.isdigit() else text#.lower()
                for text in num_pattern.split(s)]

import numpy as np

__all__ = ["relabel_ribs_anatomical"]

# --- NEW: helper to parse orientation -> axes & flags
def _derive_axes_flags_from_orientation(
    orientation: tuple[str, str, str],
) -> tuple[int, int, bool, bool]:
    """
    Given an orientation tuple for the volume axes (H, W, L), e.g. ('L','P','S'),
    infer:
      - lr_axis: which volume axis corresponds to Left/Right (X world)
      - si_axis: which volume axis corresponds to Superior/Inferior (Z world)
      - right_is_higher: does increasing index go toward RIGHT on lr_axis?
      - superior_is_higher: does increasing index go toward SUPERIOR on si_axis?

    Rules:
      orientation[0] is +H direction, orientation[1] is +W, orientation[2] is +L.
      Each element must be one of {'L','R','P','A','S','I'}.
    """
    if isinstance(orientation, (tuple, list)) and len(orientation) == 1:
        orientation = orientation[0]
    assert isinstance(orientation, (tuple, list, str)) and len(orientation) == 3, str(orientation)
    # if not (isinstance(orientation, (tuple, list)) and len(orientation) == 3):
    #     raise ValueError("orientation must be a 3-tuple like ('L','P','S').")
    ori = tuple(str(c).upper() for c in orientation)
    valid = set("LRPASI")
    if any(c not in valid for c in ori):
        raise ValueError("orientation characters must be in {'L','R','P','A','S','I'}.")

    # Map which axis is LR (X) and which is SI (Z)
    lr_axis = None
    si_axis = None
    for ax, c in enumerate(ori):
        if c in ("L", "R"):
            lr_axis = ax
        if c in ("S", "I"):
            si_axis = ax
    if lr_axis is None or si_axis is None:
        raise ValueError(
            "orientation must include one of {'L','R'} for LR axis and one of {'S','I'} for SI axis."
        )

    # Flags: whether increasing index goes toward RIGHT / SUPERIOR
    right_is_higher = (ori[lr_axis] == "R")
    superior_is_higher = (ori[si_axis] == "I")
    return lr_axis, si_axis, right_is_higher, superior_is_higher


# --- NEW: simple public API
def relabel_ribs_anatomical(
    inst: np.ndarray,
    orientation: tuple[str, str, str] = ("L", "P", "S"),
):
    """
    Simple API:
      - inst: integer instance volume shaped [H, W, L] (0 = background)
      - orientation: e.g. ('L','P','S') meaning +H=Left, +W=Posterior, +L=Superior.

    Output:
      - relabeled volume with Left ribs = 1..12 (superior->inferior),
        Right ribs = 13..24 (superior->inferior)
      - mapping {old_label -> new_label}
    """
    if inst.ndim != 3:
        raise AssertionError("Expected a 3D volume shaped [H, W, L].")

    lr_axis, si_axis, right_is_higher, superior_is_higher = _derive_axes_flags_from_orientation(orientation)
    return _relabel_ribs_anatomical_core(
        inst,
        lr_axis=lr_axis,
        si_axis=si_axis,
        right_is_higher=right_is_higher,
        superior_is_higher=superior_is_higher,
    )


# --- CORE: your prior implementation moved under-the-hood
def _relabel_ribs_anatomical_core(
    inst: np.ndarray,
    lr_axis: int,
    si_axis: int,
    right_is_higher: bool,
    superior_is_higher: bool,
):
    """
    Core relabel function (unchanged behavior):
      Left ribs  : labels  1..12 (superior -> inferior)
      Right ribs : labels 13..24 (superior -> inferior)
    """
    assert inst.ndim == 3, "Expected a 3D volume"
    relabeled = np.zeros_like(inst, dtype=inst.dtype)
    old_labels = np.array([l for l in np.unique(inst) if l != 0], dtype=int)
    if len(old_labels) == 0:
        return relabeled, {}

    # Compute centroids for each instance (in index space)
    centroids = {}
    for l in old_labels:
        a0, a1, a2 = np.where(inst == l)
        c0 = a0.mean() if a0.size > 0 else 0.0
        c1 = a1.mean() if a1.size > 0 else 0.0
        c2 = a2.mean() if a2.size > 0 else 0.0
        centroids[l] = np.array([c0, c1, c2])

    # Gather centroid coordinates along the chosen axes
    lr_vals = np.array([centroids[l][lr_axis] for l in old_labels])
    si_vals = np.array([centroids[l][si_axis] for l in old_labels])

    # Split ribs into left/right by 1D k-means (k=2) on lr coordinate
    c_left, c_right = lr_vals.min(), lr_vals.max()
    for _ in range(8):
        dist_left = np.abs(lr_vals - c_left)
        dist_right = np.abs(lr_vals - c_right)
        left_mask = dist_left <= dist_right
        if left_mask.sum() == 0 or left_mask.sum() == len(lr_vals):
            med = np.median(lr_vals)
            left_mask = lr_vals <= med
        c_left = lr_vals[left_mask].mean()
        c_right = lr_vals[~left_mask].mean()

    left_labels = old_labels[left_mask]
    right_labels = old_labels[~left_mask]

    # Ensure the 'right' cluster matches right_is_higher
    left_lr_mean = lr_vals[left_mask].mean() if left_labels.size else -np.inf
    right_lr_mean = lr_vals[~left_mask].mean() if right_labels.size else np.inf
    cluster_right_is_higher = right_lr_mean > left_lr_mean
    if cluster_right_is_higher != right_is_higher:
        left_labels, right_labels = right_labels, left_labels

    # Sort within each side by superior→inferior using si_vals
    def sort_by_superior(labels_subset):
        if labels_subset.size == 0:
            return []
        subset_idx = np.isin(old_labels, labels_subset)
        vals = si_vals[subset_idx]
        order = np.argsort(vals)
        if not superior_is_higher:
            order = order[::-1]
        return list(old_labels[subset_idx][order])

    left_sorted = sort_by_superior(left_labels)
    right_sorted = sort_by_superior(right_labels)

    # Assign new labels
    mapping = {}
    nid = 1
    for l in left_sorted:
        mapping[l] = nid
        nid += 1
        if nid > 12:
            break
    nid = 13
    for l in right_sorted:
        mapping[l] = nid
        nid += 1
        if nid > 24:
            break

    # Write relabeled volume
    for old, new in mapping.items():
        relabeled[inst == old] = new

    return relabeled, mapping

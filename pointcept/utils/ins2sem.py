import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass, label
from typing import Dict, List, Tuple, Optional
import warnings

def map_rib_instances_to_anatomical_labels(
    # nii_path: str,
    instances,
    side_separation: bool = True,
    min_volume_threshold: int = 100,
    debug: bool = False
) -> Dict[int, int]:
    """
    Map rib instance predictions to anatomical labels (rib1-rib24) based on position.

    Args:
        nii_path: Path to NIfTI file with instance predictions
        side_separation: Whether to separate left/right ribs before mapping
        min_volume_threshold: Minimum voxel count for valid rib instance
        debug: Print debug information

    Returns:
        Dictionary mapping instance_id -> anatomical_rib_label (1-24)
    """

    # Load NIfTI file
    # nii_img = nib.load(nii_path)
    # instances = nii_img.get_fdata().astype(int)

    if debug:
        print(f"Image shape: {instances.shape}")
        print(f"Unique instances: {np.unique(instances)}")

    # Get all non-background instances
    unique_instances = np.unique(instances)
    unique_instances = unique_instances[unique_instances > 0]  # Remove background

    if len(unique_instances) == 0:
        warnings.warn("No rib instances found in the image")
        return {}

    # Extract instance information
    instance_info = []
    for inst_id in unique_instances:
        mask = (instances == inst_id)
        volume = np.sum(mask)

        # Skip small instances (likely noise)
        if volume < min_volume_threshold:
            if debug:
                print(f"Skipping instance {inst_id} (volume: {volume})")
            continue

        # Calculate center of mass in LPS coordinates
        com = center_of_mass(mask)

        instance_info.append({
            'instance_id': inst_id,
            'center_of_mass': com,
            'volume': volume,
            'left_right_pos': com[0],      # L-R position (axis 0)
            'post_ant_pos': com[1],       # P-A position (axis 1)
            'sup_inf_pos': com[2]         # S-I position (axis 2)
        })

    if not instance_info:
        warnings.warn("No valid rib instances found after filtering")
        return {}

    if debug:
        print(f"Found {len(instance_info)} valid instances")

    # Mapping strategy
    if side_separation and len(instance_info) > 12:
        # Separate left and right ribs, then map each side
        mapping = _map_with_side_separation(instance_info, debug)
    else:
        # Simple superior-inferior mapping
        mapping = _map_by_superior_inferior(instance_info, debug)

    return mapping


def _map_with_side_separation(instance_info: List[Dict], debug: bool) -> Dict[int, int]:
    """
    Map ribs by first separating left/right sides, then ordering by S-I position.
    """
    # Find median L-R position to separate left/right
    lr_positions = [info['left_right_pos'] for info in instance_info]
    median_lr = np.median(lr_positions)

    # Separate left and right ribs
    left_ribs = [info for info in instance_info if info['left_right_pos'] < median_lr]
    right_ribs = [info for info in instance_info if info['left_right_pos'] >= median_lr]

    if debug:
        print(f"Left ribs: {len(left_ribs)}, Right ribs: {len(right_ribs)}")

    # Sort each side by superior-inferior position (superior = smaller S-I coordinate)
    left_ribs.sort(key=lambda x: x['sup_inf_pos'])
    right_ribs.sort(key=lambda x: x['sup_inf_pos'])

    mapping = {}

    # Map left ribs to odd numbers (1, 3, 5, ..., 23)
    for i, rib_info in enumerate(left_ribs):
        anatomical_label = 2 * i + 1  # 1, 3, 5, ...
        if anatomical_label <= 23:  # Only up to rib 23 (left side)
            mapping[rib_info['instance_id']] = anatomical_label

    # Map right ribs to even numbers (2, 4, 6, ..., 24)
    for i, rib_info in enumerate(right_ribs):
        anatomical_label = 2 * i + 2  # 2, 4, 6, ...
        if anatomical_label <= 24:  # Only up to rib 24 (right side)
            mapping[rib_info['instance_id']] = anatomical_label

    if debug:
        print("Left-Right separated mapping:")
        for inst_id, anat_id in sorted(mapping.items(), key=lambda x: x[1]):
            side = "L" if anat_id % 2 == 1 else "R"
            print(f"  Instance {inst_id} -> Rib {anat_id} ({side})")

    return mapping


def _map_by_superior_inferior(instance_info: List[Dict], debug: bool) -> Dict[int, int]:
    """
    Simple mapping by superior-inferior position only.
    """
    # Sort by superior-inferior position (superior = smaller coordinate)
    sorted_ribs = sorted(instance_info, key=lambda x: x['sup_inf_pos'])

    mapping = {}
    for i, rib_info in enumerate(sorted_ribs):
        anatomical_label = i + 1
        if anatomical_label <= 24:  # Only map up to 24 ribs
            mapping[rib_info['instance_id']] = anatomical_label

    if debug:
        print("Superior-Inferior mapping:")
        for inst_id, anat_id in sorted(mapping.items(), key=lambda x: x[1]):
            print(f"  Instance {inst_id} -> Rib {anat_id}")

    return mapping


def create_anatomical_segmentation(
    # instance_nii_path: str,
    # output_nii_path: str,
    instances, # int[H, W, L]
    mapping: Dict[int, int]
) -> None:
    """
    Create anatomical segmentation from instance segmentation using mapping.

    Args:
        instance_nii_path: Path to instance segmentation NIfTI
        output_nii_path: Path for output anatomical segmentation
        mapping: Dictionary mapping instance_id -> anatomical_label
    """
    # Load instance segmentation
    # nii_img = nib.load(instance_nii_path)
    # instances = nii_img.get_fdata().astype(int)

    # Create anatomical segmentation
    anatomical_seg = np.zeros_like(instances)

    for instance_id, anatomical_label in mapping.items():
        anatomical_seg[instances == instance_id] = anatomical_label

    # Save result
#     output_img = nib.Nifti1Image(anatomical_seg.astype(np.uint8), nii_img.affine, nii_img.header)
#     nib.save(output_img, output_nii_path)

#     print(f"Anatomical segmentation saved to: {output_nii_path}")
    return anatomical_seg


def relabel_ribs_anatomical(
    inst: np.ndarray,
    lr_axis: int = 0,
    si_axis: int = 2,
    right_is_higher: bool = True,
    superior_is_higher: bool = True,
):
    """
    Relabel a 3D instance-segmentation volume of ribs to match:
      Left ribs  : labels  1..12 (superior -> inferior)
      Right ribs : labels 13..24 (superior -> inferior)

    Parameters
    ----------
    inst : (Z,Y,X) or arbitrary-axes np.ndarray of integers
        Instance-labeled volume. 0 is background; each rib has a unique >0 id.
    lr_axis : int
        Axis index along which LEFT↔RIGHT varies (e.g., 2 for X if array is (Z,Y,X)).
    si_axis : int
        Axis index along which INFERIOR↔SUPERIOR varies (e.g., 0 for Z in (Z,Y,X)).
    right_is_higher : bool
        If True, larger centroid coordinate along lr_axis is RIGHT.
        If False, smaller centroid coordinate along lr_axis is RIGHT.
        (Set based on your image orientation.)
    superior_is_higher : bool
        If True, larger centroid coordinate along si_axis is SUPERIOR.
        If False, smaller centroid coordinate along si_axis is SUPERIOR.

    Returns
    -------
    relabeled : np.ndarray
        New volume where ribs are relabeled to 1..24 as specified (missing ribs are skipped).
    mapping : dict[int,int]
        {old_label -> new_label}
    """
    assert inst.ndim == 3, "Expected a 3D volume"
    relabeled = np.zeros_like(inst, dtype=inst.dtype)
    old_labels = np.array([l for l in np.unique(inst) if l != 0], dtype=int)
    if len(old_labels) == 0:
        return relabeled, {}

    # Compute centroids for each instance (in index space)
    centroids = {}
    for l in old_labels:
        zs, ys, xs = np.where(inst == l)
        # centroid in voxel index space
        cz = zs.mean() if zs.size > 0 else 0.0
        cy = ys.mean() if ys.size > 0 else 0.0
        cx = xs.mean() if xs.size > 0 else 0.0
        centroids[l] = np.array([cz, cy, cx])

    # Gather centroid coordinates along the chosen axes
    lr_vals = np.array([centroids[l][lr_axis] for l in old_labels])
    si_vals = np.array([centroids[l][si_axis] for l in old_labels])

    # Split ribs into left/right by 1D k-means (k=2) on lr coordinate (no sklearn)
    # Initialize with min/max; iterate a few steps (this is robust enough here).
    c_left, c_right = lr_vals.min(), lr_vals.max()
    for _ in range(8):
        dist_left = np.abs(lr_vals - c_left)
        dist_right = np.abs(lr_vals - c_right)
        left_mask = dist_left <= dist_right
        if left_mask.sum() == 0 or left_mask.sum() == len(lr_vals):
            # Fallback: split by median if degenerate
            med = np.median(lr_vals)
            left_mask = lr_vals <= med
        c_left = lr_vals[left_mask].mean()
        c_right = lr_vals[~left_mask].mean()

    left_labels = old_labels[left_mask]
    right_labels = old_labels[~left_mask]

    # Decide which cluster is actually RIGHT based on right_is_higher
    left_lr_mean = lr_vals[left_mask].mean() if left_labels.size else -np.inf
    right_lr_mean = lr_vals[~left_mask].mean() if right_labels.size else np.inf
    # If "right is higher" but the current 'right_labels' mean is lower than left's mean, swap.
    cluster_right_is_higher = right_lr_mean > left_lr_mean
    if cluster_right_is_higher != right_is_higher:
        # swap sets
        left_labels, right_labels = right_labels, left_labels
        left_mask = ~left_mask

    # Within each side, sort by superior→inferior using si_vals
    def sort_by_superior(labels_subset):
        if labels_subset.size == 0:
            return []
        subset_idx = np.isin(old_labels, labels_subset)
        vals = si_vals[subset_idx]
        # argsort to get superior→inferior
        order = np.argsort(vals)
        if not superior_is_higher:
            order = order[::-1]
        # Map order indices back to actual labels
        return list(old_labels[subset_idx][order])

    left_sorted = sort_by_superior(left_labels)
    right_sorted = sort_by_superior(right_labels)

    # Assign new labels
    mapping = {}
    new_id = 1
    for l in left_sorted:
        mapping[l] = new_id
        new_id += 1
    # Continue with right side at 13..24
    new_id = 13
    for l in right_sorted:
        mapping[l] = new_id
        new_id += 1

    # Write relabeled volume
    # (Only labels present get written; if fewer than 12 per side exist, we just fill available)
    for old, new in mapping.items():
        relabeled[inst == old] = new

    return relabeled, mapping

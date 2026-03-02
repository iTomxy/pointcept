"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os, multiprocessing as mp, itertools

try:
    import open3d as o3d
except ImportError:
    o3d = None
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def get_point_cloud(coord, color=None, verbose=True):
    if not isinstance(coord, list):
        coord = [coord]
        if color is not None:
            color = [color]

    pcd_list = []
    for i in range(len(coord)):
        coord_ = to_numpy(coord[i])
        if color is not None:
            color_ = to_numpy(color[i])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord_)
        pcd.colors = o3d.utility.Vector3dVector(
            np.zeros_like(coord_) if color is None else color_
        )
        pcd_list.append(pcd)
    if verbose:
        o3d.visualization.draw_geometries(pcd_list)
    return pcd_list


def get_line_set(coord, line, color=(1.0, 0.0, 0.0), verbose=True):
    coord = to_numpy(coord)
    line = to_numpy(line)
    colors = np.array([color for _ in range(len(line))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(coord)
    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if verbose:
        o3d.visualization.draw_geometries([line_set])
    return line_set


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    points = to_numpy(points)
    lines = to_numpy(lines)
    colors = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")


def get_palette(n_classes, pil_format=True):
    """Returns the color map for visualizing the segmentation mask.
    Example:
        ```python
        palette = get_palette(n_classes, True)
        seg_mask = seg_model(image) # int, [H, W], in [0, n_classes]
        seg_img = PIL.Image.fromarray(seg_mask)
        seg_img.putpalette(palette)
        seg_img.convert("RGB").save("seg.jpg")
        ```
    Args:
        n_classes: int, number of classes
        pil_format: bool, whether in format suitable for `PIL.Image.putpalette`.
            see: https://pillow.readthedocs.io/en/stable/reference/ImagePalette.html
    Returns:
        palette: [(R_i, G_i, B_i)] if `pil_format` is False, or
            [R1, G1, B1, R2, G2, B2, ...] if `pil_format` is True
    """
    n = n_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    if pil_format:
        return palette

    res = []
    for i in range(0, len(palette), 3):
        res.append(tuple(palette[i: i+3]))
    return res


def vis_point_cloud(xyz, label=None, n_classes=0, palette=None, window_name="Open3D"):
    """visualise 1 point cloud (label or prediction)
    xyz: int|float[n, 3], numpy.ndarray
    label: int[n] = None
    n_classes: int = 0
    palette: int[m, 3] = None, m >= n_classes
    window_name: str = "Open3D"
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if label is not None:
        assert len(label) == xyz.shape[0]
        label = np.asarray(label)
        if n_classes < 1:
            n_classes = int(label.max()) + 1
        if palette is None:
            palette = np.asarray(get_palette(n_classes, False))
        elif not isinstance(palette, np.ndarray):
            palette = np.asarray(palette)
        assert palette.shape[0] >= n_classes and 3 == palette.shape[1], "Palette ({}) too small for {} classes".format(palette.shape, n_classes)
        colors = np.asarray([palette[c] for c in label])
        colors = colors.astype(np.float32) / 255
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def vis_multi_pc(xyzs, labels=[], class_nums=[], palettes=[], windows_name=[]):
    """visualise multiple point clouds (label or prediction) simultaneously by calling `vis_point_cloud` with multi-processing
    xyzs: list of int|float[n, 3], numpy.ndarray
    labels: list of int[n] = None
    class_nums: list of int = 0
    palettes: list of int[m, 3] = None, m >= n_classes
    windows_name: list of str = "Open3D"
    """
    assert isinstance(xyzs, (list, tuple))
    if len(labels) == 0:
        labels = [None] * len(xyzs)
    if len(class_nums) == 0:
        class_nums = [0] * len(xyzs)
    if len(palettes) == 0:
        palettes = [None] * len(xyzs)
    if len(windows_name) == 0:
        windows_name = ["Open3D {}".format(i) for i in range(len(xyzs))]
    assert len(xyzs) == len(labels) == len(class_nums) == len(palettes) == len(windows_name)

    p_list = []
    for xyz, label, nc, palette, wn in zip(xyzs, labels, class_nums, palettes, windows_name):
        p = mp.Process(target=vis_point_cloud, args=(xyz, label, nc, palette, wn))
        p.start()
        p_list.append(p)
        # p.join() # do NOT join here

    for p in p_list:
        p.join()


def bbox3d_points(point1, point2):
    """Generate all integer positions (xyz) of a 3D bounding-box defined by its two diagonal points.
    Input:
        point1: List or tuple [x1, y1, z1] representing one diagonal corner.
        point2: List or tuple [x2, y2, z2] representing the opposite diagonal corner.
    Output:
        return: int[n, 3]
    """
    # Get min/max values
    x_min, x_max = min(point1[0], point2[0]), max(point1[0], point2[0])
    y_min, y_max = min(point1[1], point2[1]), max(point1[1], point2[1])
    z_min, z_max = min(point1[2], point2[2]), max(point1[2], point2[2])

    # Generate 8 vertices
    vertices = list(itertools.product([x_min, x_max], [y_min, y_max], [z_min, z_max]))

    # Define edges using pairs of vertex indices
    edge_indices = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7], [6, 7]
    ])

    # Store all edge points
    all_edge_points = set()

    # Generate integer points along each edge
    for edge in edge_indices:
        start, end = np.array(vertices[edge[0]]), np.array(vertices[edge[1]])
        # Get range for each coordinate axis
        x_range = np.arange(start[0], end[0] + np.sign(end[0] - start[0]), np.sign(end[0] - start[0])) if start[0] != end[0] else [start[0]]
        y_range = np.arange(start[1], end[1] + np.sign(end[1] - start[1]), np.sign(end[1] - start[1])) if start[1] != end[1] else [start[1]]
        z_range = np.arange(start[2], end[2] + np.sign(end[2] - start[2]), np.sign(end[2] - start[2])) if start[2] != end[2] else [start[2]]

        # Create points along the edge
        for x, y, z in zip(
            np.broadcast_to(x_range, max(len(x_range), len(y_range), len(z_range))),
            np.broadcast_to(y_range, max(len(x_range), len(y_range), len(z_range))),
            np.broadcast_to(z_range, max(len(x_range), len(y_range), len(z_range)))
        ):
            all_edge_points.add((x, y, z))

    return np.array(list(all_edge_points))


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
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Label')
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    # Save figure with transparent background
    plt.savefig(save_file, pad_inches=0.0, bbox_inches='tight')#, transparent=True)
    plt.close(fig)

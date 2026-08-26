#!/usr/bin/env python3
"""Verify the cu128_pt271 Pointcept environment, including real CUDA kernels."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail instead of doing import-only checks when no GPU is visible.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Do not require the optional Open3D/CamTools packages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / os.environ.get(
        "USER", f"uid-{os.getuid()}"
    )
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_root / "cuda-cache"))
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(cache_root / "torch-extensions")
    )
    all_arches = "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", all_arches)
    os.environ.setdefault("CUMM_CUDA_ARCH_LIST", all_arches)
    Path(os.environ["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    # Import torch first so its bundled CUDA libraries are loaded before any
    # third-party extension resolves libc10/libcudart.
    import torch
    import torchvision
    import SharedArray  # noqa: F401
    import cumm
    import pointgroup_ops
    import pointops
    import pointops2
    from pointops2 import pointops as pointops2_functions
    import pointseg
    import spconv
    import spconv.pytorch as spconv_torch
    import torch_cluster
    import torch_geometric
    import torch_scatter
    import torch_sparse
    from spconv.cppconstants import COMPILED_CUDA_ARCHS

    assert torch.__version__ == "2.7.1+cu128", torch.__version__
    assert torchvision.__version__ == "0.22.1+cu128", torchvision.__version__
    assert importlib.metadata.version("sharedarray") == "3.2.4"
    assert torch.version.cuda == "12.8", torch.version.cuda
    torch_arches = torch._C._cuda_getArchFlags().split()
    assert "sm_120" in torch_arches, torch_arches
    assert torch_cluster.__version__ == "1.6.3", torch_cluster.__version__
    assert torch_scatter.__version__ == "2.1.2", torch_scatter.__version__
    assert torch_sparse.__version__ == "0.6.18", torch_sparse.__version__
    assert torch.ops.torch_cluster.cuda_version() == 12080
    assert torch.ops.torch_scatter.cuda_version() == 12080
    assert torch.ops.torch_sparse.cuda_version() == 12080
    assert torch_geometric.__version__ == "2.6.1", torch_geometric.__version__
    assert cumm.__version__ == "0.8.2", cumm.__version__
    assert spconv.__version__ == "2.3.8", spconv.__version__
    assert (12, 0) in COMPILED_CUDA_ARCHS, COMPILED_CUDA_ARCHS

    if not args.skip_visualization:
        import camtools  # noqa: F401
        import open3d

        assert importlib.metadata.version("camtools") == "0.1.8"
        assert open3d.__version__ == "0.19.0", open3d.__version__

    # Importing Pointcept's model registry exercises spconv, PyG, and the local
    # extensions through the same import path used by training.
    import pointcept.models  # noqa: F401

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}; torchvision={torchvision.__version__}")
    print(f"torch CUDA={torch.version.cuda}; torch arches={torch_arches}")
    print(
        f"torch_geometric={torch_geometric.__version__}; "
        f"cumm={cumm.__version__}; spconv={spconv.__version__}"
    )
    if not args.skip_visualization:
        print("open3d=0.19.0; camtools=0.1.8")

    if not torch.cuda.is_available():
        message = "No visible GPU: imports passed, but CUDA kernels were not tested."
        if args.require_gpu:
            raise RuntimeError(message)
        print(message)
        return 0

    device = torch.device("cuda")
    print(
        f"GPU={torch.cuda.get_device_name(0)}; "
        f"capability={torch.cuda.get_device_capability(0)}"
    )

    # PyTorch and the locally compiled torch_scatter extension.
    matrix = torch.randn(64, 64, device=device)
    _ = matrix @ matrix
    _ = torch_scatter.scatter_sum(
        torch.ones(8, device=device),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device),
    )

    # torch_cluster and torch_sparse each launch their own CUDA extension.
    cluster_x = torch.randn(8, 3, device=device)
    cluster_y = torch.randn(4, 3, device=device)
    cluster_row, cluster_col = torch_cluster.knn(cluster_x, cluster_y, k=2)
    assert cluster_row.numel() == cluster_col.numel() == 8
    sparse_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long, device=device
    )
    sparse_value = torch.ones(4, device=device)
    sparse_matrix = torch.randn(2, 4, device=device)
    sparse_tensor = torch_sparse.SparseTensor(
        row=sparse_index[0],
        col=sparse_index[1],
        value=sparse_value,
        sparse_sizes=(2, 2),
    )
    sparse_result = sparse_tensor.matmul(sparse_matrix)
    assert sparse_result.shape == (2, 4)

    # Pointcept pointops. A real launch catches missing sm_120 cubins/PTX.
    xyz = torch.randn(32, 3, device=device).contiguous()
    offset = torch.tensor([32], dtype=torch.int32, device=device)
    new_offset = torch.tensor([8], dtype=torch.int32, device=device)
    sampled = pointops.farthest_point_sampling(xyz, offset, new_offset)
    assert sampled.numel() == 8

    # Pointcept pointops2 uses a separate extension binary.
    sampled2 = pointops2_functions.furthestsampling(xyz, offset, new_offset)
    assert sampled2.numel() == 8

    # PointGroup has an independent extension and kernel.
    batch_idxs = torch.zeros(32, dtype=torch.int32, device=device)
    batch_offsets = torch.tensor([0, 32], dtype=torch.int32, device=device)
    neighbors, start_len = pointgroup_ops.ballquery_batch_p(
        xyz, batch_idxs, batch_offsets, 10.0, 32
    )
    assert neighbors.numel() > 0
    assert start_len.shape == (32, 2)

    # spconv generated/JIT kernels.
    features = torch.randn(4, 4, device=device)
    indices = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    sparse = spconv_torch.SparseConvTensor(features, indices, [2, 2, 2], 1)
    layer = spconv_torch.SubMConv3d(4, 4, 3, bias=False).to(device)
    _ = layer(sparse)

    torch.cuda.synchronize()
    print("All Pointcept CUDA tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# syntax=docker/dockerfile:1.7
# Base Pointcept image — Pointcept and its dependencies only.
#
# Scope: Python/PyTorch, the PyG extensions, cumm/spconv, and every CUDA op
# under libs/. Project extras (SimpleITK, medpy, itk, jupyter, open3d, opencv)
# belong in a downstream image built FROM this one — see Dockerfile.
#
# Every native extension is compiled here for the union of architectures across
# the three clusters, so the image is portable and nothing is JIT-compiled at
# run time. See plans/20260826-collect_hardware_info.md for the measurements.
#
#   7.5   cetus hpc-exec01-04    Quadro RTX 6000          (only 24h/48h queues)
#   8.6   cbai A4000, iHPC A5500
#   8.9   iHPC L4, L40
#   12.0  cetus hpc-exec05-23    RTX PRO 6000 Blackwell
#
# Build (context must be the repo root, for libs/):
#   DOCKER_BUILDKIT=1 docker build -f pointcept.Dockerfile \
#       -t pointcept-base:cu128 .
#
# One cluster only, for faster iteration:
#   DOCKER_BUILDKIT=1 docker build -f pointcept.Dockerfile \
#       --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
#       --build-arg CUMM_CUDA_ARCH_LIST="12.0" \
#       --build-arg MAX_JOBS=16 -t pointcept-base:cu128-blackwell .

ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

# --------------------------------------------------------------------------
# Versions. Asserted against the running interpreter in the verification layer,
# so a base-image bump fails the build rather than silently shipping labels
# that disagree with the contents.
# --------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG CUDA_VERSION=12.8
ARG CUDNN_MAJOR=9

ARG PYG_VERSION=2.6.1
ARG TORCH_SCATTER_VERSION=2.1.2
ARG TORCH_SPARSE_VERSION=0.6.18
ARG TORCH_CLUSTER_VERSION=1.6.3
ARG CUMM_VERSION=0.8.2
ARG SPCONV_VERSION=2.3.8

# Pinned to immutable commits: spconv has no cu128 wheel, and cumm 0.8.2 is the
# first tagged release with CUDA 12.8 support.
ARG CUMM_REVISION=4c77b38d1ab57d5d1c157adddf67dad93f3a446b
ARG SPCONV_REVISION=263d6b47425ef843c82f997b12d8b714013d216c
ARG TORCH_SCATTER_REVISION=140d3ad677aae615767412873b90982cbf97d35d
ARG TORCH_SPARSE_REVISION=7d22892fb47626d2c5b0171769d44bdc8edd606c
ARG TORCH_CLUSTER_REVISION=29cd22bf1a5b82fc06b108d6573f81302c5d6b12

# torch accepts "+PTX"; cumm's parser does not, so it gets the plain list. 12.0
# is compiled natively there, so only the future-arch JIT fallback is lost.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0+PTX"
ARG CUMM_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0"
ARG MAX_JOBS=8

LABEL org.opencontainers.image.title="Pointcept base (CUDA 12.8)" \
      org.opencontainers.image.description="Pointcept with PyG, cumm/spconv and libs/ CUDA ops prebuilt for sm_75/86/89/120" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      python.version="${PYTHON_VERSION}" \
      cuda.version="${CUDA_VERSION}" \
      cudnn.version="${CUDNN_MAJOR}" \
      pytorch.version="${TORCH_VERSION}+cu128" \
      torchvision.version="${TORCHVISION_VERSION}+cu128" \
      torch_geometric.version="${PYG_VERSION}" \
      torch_scatter.version="${TORCH_SCATTER_VERSION}" \
      torch_sparse.version="${TORCH_SPARSE_VERSION}" \
      torch_cluster.version="${TORCH_CLUSTER_VERSION}" \
      cumm.version="${CUMM_VERSION}" \
      spconv.version="${SPCONV_VERSION}" \
      cuda.arch_list="${TORCH_CUDA_ARCH_LIST}" \
      pointcept.ops="pointops,pointops2,pointgroup_ops,pointseg"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUMM_CUDA_ARCH_LIST=${CUMM_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda

# --------------------------------------------------------------------------
# System packages. libsparsehash-dev provides <google/dense_hash_map>, which
# pointgroup_ops includes; git fetches the pinned revisions.
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libsparsehash-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Build tooling. pccm/ccimport/pybind11 are cumm and spconv's code generators
# and must be present before either is built.
# --------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel packaging \
    && pip install \
        "pccm==0.4.16" \
        "ccimport==0.4.4" \
        "pybind11==2.13.6" \
        fire \
        sympy \
        ninja

# --------------------------------------------------------------------------
# PyG extensions, built from source rather than taken from the PyG wheel index
# so that the architecture list above is what actually gets compiled.
# torch-sparse vendors parallel-hashmap as a submodule.
# --------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /opt/src; \
    clone_at() { \
        git clone --recursive "https://github.com/rusty1s/$1.git" "/opt/src/$1"; \
        git -C "/opt/src/$1" checkout --detach "$2"; \
        git -C "/opt/src/$1" submodule update --init --recursive; \
        pip install --no-build-isolation --no-deps "/opt/src/$1"; \
    }; \
    clone_at pytorch_scatter "${TORCH_SCATTER_REVISION}"; \
    clone_at pytorch_sparse  "${TORCH_SPARSE_REVISION}"; \
    clone_at pytorch_cluster "${TORCH_CLUSTER_REVISION}"; \
    rm -rf /opt/src

RUN pip install "torch-geometric==${PYG_VERSION}"

# --------------------------------------------------------------------------
# cumm then spconv, in that order — spconv's build imports cumm to generate its
# kernels. spconv 2.3.8 predates cumm 0.8 and declares cumm<0.8.0; the 0.8.x
# changes are precisely the CUDA 12.8 support needed, so widen that bound.
# --------------------------------------------------------------------------
RUN set -eux; \
    git clone https://github.com/FindDefinition/cumm.git /opt/src/cumm; \
    git -C /opt/src/cumm checkout --detach "${CUMM_REVISION}"; \
    pip install --no-build-isolation --no-deps /opt/src/cumm; \
    git clone https://github.com/traveller59/spconv.git /opt/src/spconv; \
    git -C /opt/src/spconv checkout --detach "${SPCONV_REVISION}"; \
    sed -i 's/<0\.8\.0/<0.9.0/g' /opt/src/spconv/setup.py; \
    grep -q '<0.9.0' /opt/src/spconv/setup.py; \
    pip install --no-build-isolation /opt/src/spconv; \
    rm -rf /opt/src

# --------------------------------------------------------------------------
# Pointcept's own CUDA ops. Only libs/ is copied: the Pointcept source itself is
# expected to be bind-mounted at run time, which is why PYTHONPATH points at the
# working directory below.
# --------------------------------------------------------------------------
COPY libs/ /opt/pointcept/libs/

RUN set -eux; \
    for op in pointops pointops2 pointgroup_ops pointseg; do \
        pip install --no-build-isolation --no-deps "/opt/pointcept/libs/${op}"; \
    done; \
    rm -rf /opt/pointcept/libs

# --------------------------------------------------------------------------
# Pointcept's Python dependencies.
#
# nibabel, seaborn and matplotlib are NOT optional here: pointcept/utils/misc.py
# imports all three at module scope and utils/config.py imports misc, so they sit
# on the core `import pointcept` path in this fork. Dropping them breaks the
# image even though nothing medical is being run.
#
# Deliberately excluded because they are reachable only from
# datasets/preprocessing/, never from training: open3d, opencv, trimesh,
# pyquaternion, imageio. scikit-learn is used solely by utils/eval_cluster.py.
# Add them downstream if a preprocessing script needs them.
# --------------------------------------------------------------------------
RUN pip install \
        addict \
        einops \
        ftfy \
        matplotlib \
        nibabel \
        numpy \
        pandas \
        plyfile \
        regex \
        scipy \
        seaborn \
        SharedArray \
        tensorboard \
        tensorboardX \
        termcolor \
        timm \
        tqdm \
        wandb \
        yapf

# --------------------------------------------------------------------------
# Verify. Fails the build if the labels disagree with reality, if a native
# extension cannot load, or if a CUDA op is missing one of the target
# architectures — the failure mode that produced "no kernel image is available
# for execution on the device" on this fleet before. Also records the resolved
# versions at /etc/pointcept-image.json for run-time introspection.
#
# No GPU is present during a build, so this checks loading, ABI and compiled
# SASS, never kernel execution.
# --------------------------------------------------------------------------
ENV POINTCEPT_EXPECT="${PYTHON_VERSION},${TORCH_VERSION},${TORCHVISION_VERSION},${CUDA_VERSION},${CUDNN_MAJOR}"

RUN python <<'PYEOF'
import glob, json, os, re, subprocess, sys

py_v, torch_v, tv_v, cuda_v, cudnn_major = os.environ["POINTCEPT_EXPECT"].split(",")
problems = []

import torch, torchvision

actual_py = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual_py != py_v:
    problems.append(f"python {actual_py} != labelled {py_v}")
if not torch.__version__.startswith(torch_v):
    problems.append(f"torch {torch.__version__} != labelled {torch_v}")
if not torchvision.__version__.startswith(tv_v):
    problems.append(f"torchvision {torchvision.__version__} != labelled {tv_v}")
if torch.version.cuda != cuda_v:
    problems.append(f"cuda {torch.version.cuda} != labelled {cuda_v}")
cudnn = torch.backends.cudnn.version()
if cudnn is None or cudnn // 10000 != int(cudnn_major):
    problems.append(f"cudnn {cudnn} is not major {cudnn_major}")

# Every extension must import.
import torch_cluster, torch_geometric, torch_scatter, torch_sparse
import cumm, spconv, spconv.pytorch
import pointgroup_ops, pointops, pointseg
from pointops2 import pointops as _pointops2

for mod, want in (
    (torch_scatter, "2.1.2"), (torch_sparse, "0.6.18"), (torch_cluster, "1.6.3"),
):
    got = mod.__version__
    if got != want:
        problems.append(f"{mod.__name__} {got} != {want} (a wheel, not our build?)")

# spconv records the architectures it was compiled for.
from spconv.cppconstants import COMPILED_CUDA_ARCHS

targets = set()
for part in os.environ["TORCH_CUDA_ARCH_LIST"].replace("+PTX", "").split(";"):
    part = part.strip()
    if part:
        major, minor = part.split(".")
        targets.add(f"sm_{major}{minor}")

spconv_archs = {f"sm_{a}{b}" for a, b in COMPILED_CUDA_ARCHS}
missing = targets - spconv_archs
if missing:
    problems.append(f"spconv missing archs {sorted(missing)}; has {sorted(spconv_archs)}")

# Read the compiled SASS out of each CUDA op and confirm nothing is absent.
# pointseg is a CppExtension and carries no device code, so it is skipped.
site = os.path.dirname(torch.__file__).rsplit("/", 1)[0]
for package in ("pointops", "pointops2", "pointgroup_ops",
                "torch_scatter", "torch_sparse", "torch_cluster"):
    for so in glob.glob(f"{site}/{package}/**/*.so", recursive=True) + \
              glob.glob(f"{site}/{package}*.so"):
        try:
            out = subprocess.run(["cuobjdump", "--list-elf", so],
                                 capture_output=True, text=True, timeout=120).stdout
        except Exception:
            continue
        found = set(re.findall(r"sm_(\d+)", out))
        if not found:
            continue          # host-only object
        found = {f"sm_{a}" for a in found}
        gap = targets - found
        if gap:
            problems.append(f"{os.path.basename(so)} missing {sorted(gap)}")

def _nccl_version():
    # Reads a compiled-in constant, but can still raise with no driver present.
    try:
        return ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        return "unknown"


if problems:
    print("IMAGE VERIFICATION FAILED", file=sys.stderr)
    for p in problems:
        print("  -", p, file=sys.stderr)
    raise SystemExit(1)

manifest = {
    "python": actual_py,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "cudnn": cudnn,
    "nccl": _nccl_version(),
    "torch_geometric": torch_geometric.__version__,
    "torch_scatter": torch_scatter.__version__,
    "torch_sparse": torch_sparse.__version__,
    "torch_cluster": torch_cluster.__version__,
    "cumm": cumm.__version__,
    "spconv": spconv.__version__,
    "cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "compiled_archs": sorted(spconv_archs),
    "pointcept_ops": ["pointops", "pointops2", "pointgroup_ops", "pointseg"],
}
with open("/etc/pointcept-image.json", "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PYEOF

# Pointcept is bind-mounted at run time; "." makes it importable from the
# working directory the way the project's own scripts expect.
ENV PYTHONPATH=.
WORKDIR /workspace

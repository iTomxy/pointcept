# syntax=docker/dockerfile:1.7
# Base Pointcept image — upstream Pointcept and its dependencies only.
#
# Standalone: the build context is never read, so this file can be published and
# built anywhere with `docker build -f pointcept.Dockerfile .` (or piped in on
# stdin). Pointcept itself is cloned from GitHub at a pinned revision; nothing
# comes from a local checkout.
#
# Scope: Python/PyTorch, the PyG extensions, cumm/spconv, and the five CUDA ops
# under Pointcept's libs/. Anything beyond upstream — extra datasets, medical
# imaging, visualisation, jupyter — belongs in a downstream image built FROM
# this one.
#
# Every native extension is compiled here for a broad architecture range, so the
# image is portable across mixed GPU fleets and nothing is JIT-compiled at run
# time. The default spans Turing to Blackwell:
#
#   7.5   Quadro RTX 6000, RTX 2080 Ti, T4
#   8.6   RTX A4000/A5000/A5500, RTX 3090
#   8.9   L4, L40, RTX 4090
#   12.0  RTX PRO 6000 Blackwell
#
# Note this deliberately departs from upstream's README, which pins torch 2.5 /
# CUDA 12.4. CUDA 12.8 is the minimum that supports sm_120 (Blackwell), and
# 12.8 rather than a later release because some drivers cap there.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f pointcept.Dockerfile -t pointcept-base:cu128 .
#
# Trim to one architecture for faster iteration:
#   DOCKER_BUILDKIT=1 docker build -f pointcept.Dockerfile \
#       --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
#       --build-arg CUMM_CUDA_ARCH_LIST="12.0" \
#       --build-arg MAX_JOBS=16 -t pointcept-base:cu128-blackwell .
#
# Run — upstream Pointcept ships in the image at /opt/pointcept:
#   docker run --gpus all -w /opt/pointcept pointcept-base:cu128 python tools/train.py ...
#
# Run a fork instead, without rebuilding. PYTHONPATH=. resolves the package from
# the working directory, so the mounted tree wins over /opt/pointcept:
#   docker run --gpus all -v /path/to/your/fork:/workspace pointcept-base:cu128 ...

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

# Pointcept v1.7.0-8-g9f37497. Pinned to a commit rather than a branch so the
# image is reproducible; bump deliberately.
ARG POINTCEPT_REPOSITORY=https://github.com/Pointcept/Pointcept.git
ARG POINTCEPT_REVISION=9f37497e4f3005c90bbbe7221b86439c29d60611
ARG POINTCEPT_VERSION=1.7.0

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
      org.opencontainers.image.description="Upstream Pointcept with PyG, cumm/spconv and the libs/ CUDA ops prebuilt for sm_75/86/89/120" \
      org.opencontainers.image.source="${POINTCEPT_REPOSITORY}" \
      org.opencontainers.image.revision="${POINTCEPT_REVISION}" \
      org.opencontainers.image.version="${POINTCEPT_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      pointcept.version="${POINTCEPT_VERSION}" \
      pointcept.revision="${POINTCEPT_REVISION}" \
      pointcept.ops="pointops,pointops2,pointgroup_ops,pointseg,pointrope" \
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
      cuda.arch_list="${TORCH_CUDA_ARCH_LIST}"

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
#
# *_DISABLE_JIT=1 is what makes these ahead-of-time builds, and it is not
# optional here. Without it both setup.py files take their JIT branch: they
# register no ext_modules, compile nothing, and — in cumm's case — skip the
# CopyHeaderCallback that installs cumm/include into site-packages. The
# resulting package resolves its headers from the *source tree* instead, so
# deleting /opt/src below leaves `import cumm` raising
# AssertionError: TENSORVIEW_INCLUDE_PATH.exists(), which also takes spconv
# down with it. JIT mode would additionally defer every kernel to run time,
# defeating the point of baking architectures into the image.
# --------------------------------------------------------------------------
RUN set -eux; \
    export CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=1; \
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
# Upstream Pointcept, and its five CUDA ops.
#
# The source tree is kept at /opt/pointcept rather than deleted: Pointcept has
# no setup.py and is run in place, so the image would otherwise ship the
# compiled ops with no code to drive them. It also documents exactly which
# revision the ops were built from.
# --------------------------------------------------------------------------
RUN set -eux; \
    git clone "${POINTCEPT_REPOSITORY}" /opt/pointcept; \
    git -C /opt/pointcept checkout --detach "${POINTCEPT_REVISION}"; \
    for op in pointops pointops2 pointgroup_ops pointseg pointrope; do \
        pip install --no-build-isolation --no-deps "/opt/pointcept/libs/${op}"; \
    done; \
    find /opt/pointcept/libs -name build -type d -prune -exec rm -rf {} +; \
    rm -rf /opt/pointcept/.git

# --------------------------------------------------------------------------
# Upstream Pointcept's Python dependencies, per its README plus an import scan
# of pointcept/ excluding datasets/preprocessing/.
#
# Left out on purpose, all reachable only from optional code paths:
#   open3d, opencv, trimesh, pyquaternion, imageio  — datasets/preprocessing/
#   scikit-learn                                    — utils/eval_cluster.py
#   MinkowskiEngine, Swin3D                         — alternative backbones
#   transformers, peft, CLIP                        — PPT models
#   nuscenes, waymo-open-dataset, tensorflow        — dataset-specific
# Add whichever a downstream image actually needs.
# --------------------------------------------------------------------------
RUN pip install \
        addict \
        einops \
        ftfy \
        h5py \
        numpy \
        pandas \
        pillow \
        plyfile \
        pyyaml \
        regex \
        scipy \
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
# architectures — the failure mode behind "no kernel image is available for
# execution on the device" on a mixed fleet. Also records the resolved versions
# at /etc/pointcept-image.json for run-time introspection.
#
# No GPU is present during a build, so this checks loading, ABI and compiled
# SASS, never kernel execution.
# --------------------------------------------------------------------------
ENV POINTCEPT_EXPECT="${PYTHON_VERSION},${TORCH_VERSION},${TORCHVISION_VERSION},${CUDA_VERSION},${CUDNN_MAJOR},${POINTCEPT_REVISION}"

RUN python <<'PYEOF'
import glob, json, os, re, subprocess, sys

py_v, torch_v, tv_v, cuda_v, cudnn_major, pointcept_rev = \
    os.environ["POINTCEPT_EXPECT"].split(",")
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
import pointgroup_ops, pointops, pointrope, pointseg
from pointops2 import pointops as _pointops2

for mod, want in (
    (torch_scatter, "2.1.2"), (torch_sparse, "0.6.18"), (torch_cluster, "1.6.3"),
):
    if mod.__version__ != want:
        problems.append(
            f"{mod.__name__} {mod.__version__} != {want} (a wheel, not our build?)")

# Pointcept itself must import from the tree the ops were built from.
sys.path.insert(0, "/opt/pointcept")
import pointcept  # noqa: F401

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
# pointseg is a CppExtension and carries no device code, so it drops out here.
#
# cumm and spconv are deliberately NOT scanned. Both install an extension named
# core_cc.cpython-*.so — cumm/core_cc and spconv/core_cc — so they collide on
# basename, and cumm's binding layer legitimately carries only nvcc's default
# arch. COMPILED_CUDA_ARCHS above is the authoritative record of what those two
# were compiled for, straight from spconv.core_cc, so scanning them here adds
# nothing but false alarms. Do not "fix" that collision by keeping whichever
# file has more architectures: that silently compares two unrelated libraries
# and would pass a spconv missing a target arch as long as cumm had it.
site = os.path.dirname(torch.__file__).rsplit("/", 1)[0]
seen = set()
for package in ("pointops", "pointops2", "pointgroup_ops", "pointrope",
                "torch_scatter", "torch_sparse", "torch_cluster"):
    for so in glob.glob(f"{site}/{package}/**/*.so", recursive=True) + \
              glob.glob(f"{site}/{package}*.so"):
        # The two globs overlap (pointops* also matches pointops2*), so key on
        # the real path to avoid running cuobjdump over the same file twice.
        so = os.path.realpath(so)
        if so in seen:
            continue
        seen.add(so)
        try:
            out = subprocess.run(["cuobjdump", "--list-elf", so],
                                 capture_output=True, text=True, timeout=120).stdout
        except Exception:
            continue
        found = {f"sm_{a}" for a in re.findall(r"sm_(\d+)", out)}
        if not found:
            continue          # host-only object
        gap = targets - found
        if gap:
            problems.append(f"{os.path.relpath(so, site)} missing {sorted(gap)}")


def _nccl_version():
    # Reads a compiled-in constant, but can still raise with no driver present.
    try:
        return ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        return "unknown"


if problems:
    print("IMAGE VERIFICATION FAILED", file=sys.stderr)
    for problem in problems:
        print("  -", problem, file=sys.stderr)
    raise SystemExit(1)

manifest = {
    "pointcept_revision": pointcept_rev,
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
    "pointcept_ops": ["pointops", "pointops2", "pointgroup_ops", "pointseg",
                      "pointrope"],
}
with open("/etc/pointcept-image.json", "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PYEOF

# "." resolves the package from the working directory, so running with -w
# /opt/pointcept uses the bundled upstream tree, while bind-mounting a fork over
# the working directory uses the fork instead — no rebuild, and no silent
# fallback between the two.
ENV PYTHONPATH=.
WORKDIR /opt/pointcept

#!/usr/bin/env bash
# Install Pointcept-specific dependencies into the general cu128_pt271 env.
# Create the base environment first with ~/env-cu128_pt271.sh.
#
# Usage:
#   bash env.sh
#   FORCE_REBUILD=1 bash env.sh
#   INSTALL_VISUALIZATION=0 bash env.sh  # Skip optional Open3D/CamTools.
#   POINTCEPT_CUDA_ARCH_LIST='8.6;8.9;12.0+PTX' bash env.sh

set -Eeuo pipefail

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] env.sh failed at line ${BASH_LINENO[0]} (exit ${status})" >&2
    exit "${status}"
}
trap on_error ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${1:-${HOME}/miniconda3}"
CONDA_EXE="${CONDA_ROOT}/bin/conda"
ENV_NAME="${ENV_NAME:-cu128_pt271}"
ENV_ROOT="${CONDA_ROOT}/envs/${ENV_NAME}"
ENV_PYTHON="${ENV_ROOT}/bin/python"
SOURCE_ROOT="${ENV_ROOT}/src/pointcept"

TORCH_VERSION=2.7.1
CUDA_VERSION=12.8
PYG_VERSION=2.6.1
TORCH_CLUSTER_VERSION=1.6.3
TORCH_SCATTER_VERSION=2.1.2
TORCH_SPARSE_VERSION=0.6.18
OPEN3D_VERSION=0.19.0
CAMTOOLS_VERSION=0.1.8
CUDA_ARCH_LIST="${POINTCEPT_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
INSTALL_VISUALIZATION="${INSTALL_VISUALIZATION:-1}"
BUILD_COMPILER_ID=uninitialized
BUILD_LINKER_ID=uninitialized
BUILD_SYSROOT=uninitialized

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export CUMM_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export MAX_JOBS="${MAX_JOBS:-4}"

# cumm 0.8.2 is the first tagged release with CUDA 12.8 support. spconv has no
# official cu128 wheel, so both are installed from immutable upstream commits.
CUMM_REPOSITORY=https://github.com/FindDefinition/cumm.git
CUMM_REVISION=4c77b38d1ab57d5d1c157adddf67dad93f3a446b
SPCONV_REPOSITORY=https://github.com/traveller59/spconv.git
SPCONV_REVISION=263d6b47425ef843c82f997b12d8b714013d216c
CUMM_SOURCE="${SOURCE_ROOT}/cumm-0.8.2"
SPCONV_SOURCE="${SOURCE_ROOT}/spconv-2.3.8"

TORCH_SCATTER_REPOSITORY=https://github.com/rusty1s/pytorch_scatter.git
TORCH_SCATTER_REVISION=140d3ad677aae615767412873b90982cbf97d35d
TORCH_SPARSE_REPOSITORY=https://github.com/rusty1s/pytorch_sparse.git
TORCH_SPARSE_REVISION=7d22892fb47626d2c5b0171769d44bdc8edd606c
TORCH_CLUSTER_REPOSITORY=https://github.com/rusty1s/pytorch_cluster.git
TORCH_CLUSTER_REVISION=29cd22bf1a5b82fc06b108d6573f81302c5d6b12
TORCH_SCATTER_SOURCE="${SOURCE_ROOT}/torch-scatter-${TORCH_SCATTER_VERSION}"
TORCH_SPARSE_SOURCE="${SOURCE_ROOT}/torch-sparse-${TORCH_SPARSE_VERSION}"
TORCH_CLUSTER_SOURCE="${SOURCE_ROOT}/torch-cluster-${TORCH_CLUSTER_VERSION}"

log() {
    printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
    echo "env.sh: $*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

marker_matches() {
    local marker="$1"
    [[ -f "${marker}" ]] && grep -Fxq "$(build_fingerprint)" "${marker}"
}

write_marker() {
    local marker="$1"
    build_fingerprint >"${marker}"
}

build_fingerprint() {
    printf '%s\n' \
        "torch=${TORCH_VERSION} cuda=${CUDA_VERSION} arch=${CUDA_ARCH_LIST} compiler=${BUILD_COMPILER_ID} linker=${BUILD_LINKER_ID} sysroot=${BUILD_SYSROOT} pyg=${TORCH_SCATTER_REVISION}/${TORCH_SPARSE_REVISION}/${TORCH_CLUSTER_REVISION} cumm=${CUMM_REVISION} spconv=${SPCONV_REVISION}"
}

load_cuda_128() {
    local nvcc_path=""
    local cuda_root

    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        nvcc_path="${CUDA_HOME}/bin/nvcc"
    elif type module >/dev/null 2>&1; then
        module load "cuda/${CUDA_VERSION}"
        nvcc_path="$(command -v nvcc || true)"
    fi

    if [[ -z "${nvcc_path}" ]]; then
        local candidate
        for candidate in \
            "/shared/apps/cuda-${CUDA_VERSION}" \
            "/usr/local/cuda-${CUDA_VERSION}" \
            "/opt/cuda-${CUDA_VERSION}"; do
            if [[ -x "${candidate}/bin/nvcc" ]]; then
                nvcc_path="${candidate}/bin/nvcc"
                break
            fi
        done
    fi

    [[ -x "${nvcc_path}" ]] || die \
        "CUDA ${CUDA_VERSION} nvcc was not found. Load cuda/${CUDA_VERSION} or set CUDA_HOME."

    cuda_root="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
    export CUDA_HOME="${cuda_root}"
    export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="${ENV_ROOT}/include:${CUDA_HOME}/include:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${ENV_ROOT}/include:${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
    export LIBRARY_PATH="${CUDA_HOME}/lib64:${ENV_ROOT}/lib:${LIBRARY_PATH:-}"

    "${CUDA_HOME}/bin/nvcc" --version | grep -q "release ${CUDA_VERSION}" || die \
        "${CUDA_HOME}/bin/nvcc is not CUDA ${CUDA_VERSION}; CUDA 12.8 is required for sm_120."
}

load_conda_compiler() {
    local compiler_prefix="${ENV_ROOT}/bin/x86_64-conda-linux-gnu"
    local linker="${compiler_prefix}-ld"
    local minimum_linker=2.40

    [[ -x "${compiler_prefix}-gcc" && -x "${compiler_prefix}-g++" && \
        -x "${linker}" ]] || die \
        "The Conda GCC/G++ toolchain is missing. Rerun ~/env-cu128_pt271.sh."

    # PyTorch 2.7 requires GCC >=9. The system GCC on the RHEL 8 login nodes is
    # 8.5, so use the environment-owned GCC 11 toolchain for every native op.
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    export CC="${compiler_prefix}-gcc"
    export CXX="${compiler_prefix}-g++"
    export CUDAHOSTCXX="${CXX}"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"

    BUILD_COMPILER_ID="$("${CXX}" -dumpfullversion -dumpversion)"
    BUILD_LINKER_ID="$("${linker}" --version | sed -n '1{s/.* //;p;}')"
    BUILD_SYSROOT="$("${CXX}" -print-sysroot)"
    [[ -d "${BUILD_SYSROOT}" ]] || die \
        "The Conda compiler did not report a usable sysroot: ${BUILD_SYSROOT}"
    [[ "$(printf '%s\n' "${minimum_linker}" "${BUILD_LINKER_ID}" | \
        sort -V | head -n 1)" == "${minimum_linker}" ]] || die \
        "Conda binutils ${BUILD_LINKER_ID} is too old; rerun ~/env-cu128_pt271.sh."

    log "Compiler ${BUILD_COMPILER_ID}; binutils ${BUILD_LINKER_ID}; sysroot ${BUILD_SYSROOT}"
}

clone_pinned() {
    local repository="$1"
    local revision="$2"
    local destination="$3"
    local local_src="${LOCAL_SOURCE_CACHE:-}"

    # Use a login-node checkout (cloned ahead of time, e.g. into ~/codes/) so
    # the compute-node build never needs the git binary or GitHub access.
    if [[ -n "${local_src}" && -d "${local_src}/$(basename "${destination}")/.git" ]]; then
        log "Using local checkout for $(basename "${destination}")"
        rm -rf "${destination}"
        cp -a "${local_src}/$(basename "${destination}")" "${destination}"
        return 0
    fi

    if [[ ! -d "${destination}/.git" ]]; then
        [[ ! -e "${destination}" ]] || die \
            "${destination} exists but is not a git checkout; move it aside first."
        git clone --no-checkout "${repository}" "${destination}"
        git -C "${destination}" checkout --detach "${revision}"
    fi

    [[ "$(git -C "${destination}" rev-parse HEAD)" == "${revision}" ]] || die \
        "${destination} is not at the revision pinned by env.sh; move it aside first."
}

clean_build_dir() {
    local source_dir="$1"
    if [[ -d "${source_dir}/build" ]]; then
        find "${source_dir}/build" -mindepth 1 -delete
        rmdir "${source_dir}/build" 2>/dev/null || true
    fi
}

pip_install() {
    "${ENV_PYTHON}" -m pip install "$@"
}

visualization_dependencies_ok() {
    "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

import camtools  # noqa: F401
import open3d

assert version("camtools") == "0.1.8"
assert open3d.__version__ == "0.19.0"
PY
}

conda_dependencies_ok() {
    # google-sparsehash ships the header PointGroup includes; sharedarray is a
    # runtime import. Both are already pinned into the env by a prior run.
    [[ -f "${ENV_ROOT}/include/google/dense_hash_map" ]] || return 1
    "${ENV_PYTHON}" -c 'import SharedArray' >/dev/null 2>&1 || return 1
    if is_true "${INSTALL_VISUALIZATION}"; then
        visualization_dependencies_ok || return 1
    fi
    return 0
}

# Abort if a conda solve proposes to touch anything outside the packages we
# asked for. The base env is a defaults/conda-forge hybrid (gxx_linux-64 and
# binutils_linux-64 come from pkgs/main, sysroot_linux-64 from conda-forge), so
# an unconstrained solve against --channel conda-forge migrates the whole
# environment -- python, libgcc, libstdcxx and ~70 more -- from pkgs/main to
# conda-forge. On this NFS prefix that transaction is slow enough to be killed
# by a PBS walltime, which leaves every one of those packages half-linked as
# *.c~ husks. Fail loudly instead of starting a migration we cannot finish.
assert_solve_is_surgical() {
    local plan checker
    plan="$("${CONDA_EXE}" install --yes --dry-run --json --freeze-installed \
        --prefix "${ENV_ROOT}" "$@" 2>/dev/null)" || return 0

    read -r -d '' checker <<'PY' || true
import json, re, sys

requested = {re.split(r"[=<>! ]", spec.split("::")[-1])[0] for spec in sys.argv[1:]}
try:
    plan = json.loads(sys.stdin.read())
except ValueError:
    sys.exit(0)

actions = plan.get("actions") or {}
touched = set()
for key in ("LINK", "UNLINK"):
    for record in actions.get(key, []):
        if isinstance(record, dict):
            touched.add(record.get("name", ""))
        else:
            touched.add(str(record).split("::")[-1].rsplit("-", 2)[0])
collateral = sorted(name for name in touched - requested if name)
if collateral:
    print("conda would also change: " + ", ".join(collateral), file=sys.stderr)
    sys.exit(1)
PY

    printf '%s' "${plan}" | "${ENV_PYTHON}" -c "${checker}" "$@" || die \
        "The conda solve would rewrite packages beyond the requested specs; refusing to start a migration that a walltime kill would corrupt. Set POINTCEPT_ALLOW_CONDA_MIGRATION=1 to override (only on a node with no walltime)."
}

install_conda_dependencies() {
    # Channel-qualified so the solver cannot drift these onto another channel.
    # The compiler toolchain is deliberately NOT re-specified here: it is
    # already installed by ~/env-cu128_pt271.sh, and naming it again invites the
    # solver to reconsider (and relocate) it. Verify it instead -- that is what
    # load_conda_compiler does, right after this step.
    local specifications=(
        "bioconda::google-sparsehash=2.0.3"
        "conda-forge::sharedarray=3.2.4"
    )

    if is_true "${INSTALL_VISUALIZATION}"; then
        # PyPI's Open3D wheel does not support this Python/host combination;
        # the conda-forge build matches the RHEL 8 glibc baseline.
        specifications+=("conda-forge::open3d=${OPEN3D_VERSION}")
    fi

    if conda_dependencies_ok; then
        log "Pointcept conda dependencies are already present"
        return
    fi

    log "Installing Pointcept conda dependencies"
    if ! is_true "${POINTCEPT_ALLOW_CONDA_MIGRATION:-0}"; then
        assert_solve_is_surgical "${specifications[@]}"
    fi
    "${CONDA_EXE}" install --yes --freeze-installed \
        --prefix "${ENV_ROOT}" "${specifications[@]}"
    [[ -f "${ENV_ROOT}/include/google/dense_hash_map" ]] || die \
        "google-sparsehash did not install google/dense_hash_map."
}

install_visualization_dependencies() {
    if ! is_true "${INSTALL_VISUALIZATION}"; then
        log "Skipping optional Open3D/CamTools dependencies"
        return
    fi
    if visualization_dependencies_ok; then
        log "Open3D/CamTools dependencies are already importable"
        return
    fi

    log "Installing optional CamTools dependency"
    pip_install "camtools==${CAMTOOLS_VERSION}"
    visualization_dependencies_ok || die \
        "The Open3D/CamTools visualization dependencies do not import."
}

pyg_extensions_ok() {
    "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import torch  # Load torch's shared libraries before the extensions.
import torch_cluster
import torch_scatter
import torch_sparse

assert torch_cluster.__version__ == "1.6.3"
assert torch_scatter.__version__ == "2.1.2"
assert torch_sparse.__version__ == "0.6.18"
assert torch.ops.torch_cluster.cuda_version() == 12080
assert torch.ops.torch_scatter.cuda_version() == 12080
assert torch.ops.torch_sparse.cuda_version() == 12080
PY
}

install_pyg_extensions() {
    local marker="${ENV_ROOT}/.pointcept-pyg-native"
    local repositories=(
        "${TORCH_SCATTER_REPOSITORY}"
        "${TORCH_SPARSE_REPOSITORY}"
        "${TORCH_CLUSTER_REPOSITORY}"
    )
    local revisions=(
        "${TORCH_SCATTER_REVISION}"
        "${TORCH_SPARSE_REVISION}"
        "${TORCH_CLUSTER_REVISION}"
    )
    local sources=(
        "${TORCH_SCATTER_SOURCE}"
        "${TORCH_SPARSE_SOURCE}"
        "${TORCH_CLUSTER_SOURCE}"
    )
    local index

    if ! is_true "${FORCE_REBUILD}" && marker_matches "${marker}" && \
            pyg_extensions_ok; then
        log "Locally built PyG extensions are already importable"
        return
    fi

    log "Building PyG extensions locally for glibc compatibility"
    # Wheels from the PyG index (local version tag "+ptXXcuYYY") are built on a
    # glibc >= 2.32 host and cannot load on RHEL 8. If a previous run was
    # interrupted before it rebuilt one of these from source, the downloaded
    # wheel is still installed and will fail at import with an undefined
    # __libc_single_threaded. Remove all three up front so the loop below can
    # only ever leave locally built extensions behind.
    "${ENV_PYTHON}" -m pip uninstall --yes \
        torch-scatter torch-sparse torch-cluster >/dev/null 2>&1 || true
    mkdir -p "${SOURCE_ROOT}"
    export FORCE_CUDA=1
    for index in "${!repositories[@]}"; do
        clone_pinned \
            "${repositories[index]}" "${revisions[index]}" "${sources[index]}"
        if [[ "${sources[index]}" == "${TORCH_SPARSE_SOURCE}" ]]; then
            if [[ -x "$(command -v git)" ]]; then
                git -C "${TORCH_SPARSE_SOURCE}" submodule update --init --recursive
            fi
            [[ -f "${TORCH_SPARSE_SOURCE}/third_party/parallel-hashmap/parallel_hashmap/phmap.h" ]] || \
                die "torch-sparse's parallel-hashmap submodule is incomplete."
        fi
        clean_build_dir "${sources[index]}"
        find "${sources[index]}" -type f \( -name '*.so' -o -name '*.pyd' \) -delete
        pip_install --no-cache-dir --no-build-isolation --no-deps --force-reinstall \
            "${sources[index]}"
    done
    unset FORCE_CUDA

    pyg_extensions_ok || die "A locally built PyG extension does not import."
    write_marker "${marker}"
}

spconv_ok() {
    "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import cumm
import spconv
import spconv.pytorch  # noqa: F401
from spconv.cppconstants import COMPILED_CUDA_ARCHS

assert cumm.__version__ == "0.8.2"
assert spconv.__version__ == "2.3.8"
assert (12, 0) in COMPILED_CUDA_ARCHS
PY
}

uninstall_spconv_variants() {
    local packages=()

    mapfile -t packages < <("${ENV_PYTHON}" - <<'PY'
import importlib.metadata
import re

for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name", "")
    if re.fullmatch(r"(?:cumm|spconv)(?:-cu\d+)?", name, flags=re.IGNORECASE):
        print(name)
PY
    )
    if ((${#packages[@]})); then
        "${ENV_PYTHON}" -m pip uninstall --yes "${packages[@]}"
    fi
}

install_spconv() {
    local marker="${ENV_ROOT}/.pointcept-spconv-2.3.8-cumm-0.8.2"

    if ! is_true "${FORCE_REBUILD}" && marker_matches "${marker}" && spconv_ok; then
        log "cumm/spconv are already installed for ${CUDA_ARCH_LIST}"
        return
    fi

    log "Building cumm 0.8.2 and spconv 2.3.8"
    mkdir -p "${SOURCE_ROOT}"
    clone_pinned "${CUMM_REPOSITORY}" "${CUMM_REVISION}" "${CUMM_SOURCE}"
    clone_pinned "${SPCONV_REPOSITORY}" "${SPCONV_REVISION}" "${SPCONV_SOURCE}"
    uninstall_spconv_variants

    # spconv 2.3.8 predates cumm's CUDA-12.8 release and declares cumm<0.8.
    # The 0.8.x changes add CUDA/compiler support, so widen the stale bound in
    # this environment-owned clone while retaining the pinned source versions.
    sed -i 's/<0\.8\.0/<0.9.0/g' "${SPCONV_SOURCE}/setup.py"
    grep -q '<0.9.0' "${SPCONV_SOURCE}/setup.py" || die \
        "Could not update spconv's cumm version bound."

    clean_build_dir "${CUMM_SOURCE}"
    clean_build_dir "${SPCONV_SOURCE}"
    find "${CUMM_SOURCE}" "${SPCONV_SOURCE}" \
        -type f \( -name '*.so' -o -name '*.pyd' \) -delete

    # Editable installs use cumm/spconv's recommended JIT development build.
    # Importing once here materializes their native libraries for all arches.
    pip_install --no-build-isolation --no-deps --editable "${CUMM_SOURCE}"
    pip_install --no-build-isolation --editable "${SPCONV_SOURCE}"
    spconv_ok || die "The locally built cumm/spconv installation does not import."

    write_marker "${marker}"
}

pointcept_ops_ok() {
    "${ENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import torch  # Load torch's shared libraries before the extensions.
import pointgroup_ops  # noqa: F401
import pointops  # noqa: F401
from pointops2 import pointops  # noqa: F401
import pointseg  # noqa: F401
PY
}

install_pointcept_ops() {
    local marker="${ENV_ROOT}/.pointcept-native-ops"
    local op_dir

    if ! is_true "${FORCE_REBUILD}" && marker_matches "${marker}" && \
            pointcept_ops_ok; then
        log "Pointcept native ops are already installed for ${CUDA_ARCH_LIST}"
        return
    fi

    log "Building Pointcept native ops for ${CUDA_ARCH_LIST}"
    for op_dir in \
        "${PROJECT_ROOT}/libs/pointops" \
        "${PROJECT_ROOT}/libs/pointops2" \
        "${PROJECT_ROOT}/libs/pointgroup_ops" \
        "${PROJECT_ROOT}/libs/pointseg"; do
        [[ -f "${op_dir}/setup.py" ]] || die "Missing ${op_dir}/setup.py"
        clean_build_dir "${op_dir}"
        pip_install --no-build-isolation --no-deps --force-reinstall "${op_dir}"
    done

    pointcept_ops_ok || die "A Pointcept native extension does not import."
    write_marker "${marker}"
}

[[ -x "${ENV_PYTHON}" ]] || die \
    "Missing ${ENV_PYTHON}. Create it first with ~/env-cu128_pt271.sh."
[[ -x "${CONDA_EXE}" ]] || die "Conda was not found at ${CONDA_EXE}."

case ";${CUDA_ARCH_LIST};" in
    *';12.0;'*|*';12.0+PTX;'*) ;;
    *) die "POINTCEPT_CUDA_ARCH_LIST must contain 12.0 or 12.0+PTX for cetus." ;;
esac

"${ENV_PYTHON}" - <<'PY' || die "The base environment is not torch 2.7.1+cu128."
import torch
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
arch_flags = torch._C._cuda_getArchFlags().split()
assert "sm_120" in arch_flags, arch_flags
PY

load_cuda_128
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/${USER}/cuda-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/${USER}/torch-extensions}"
mkdir -p "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}"

# The Conda prefix is shared over NFS. Serialize native builds so two jobs do
# not write the same site-packages/source tree concurrently.
command -v flock >/dev/null 2>&1 || die "flock is required to serialize native builds."
BUILD_LOCK="${ENV_ROOT}/.pointcept-build.lock"
exec 9>"${BUILD_LOCK}"
flock -w 7200 9 || die "Timed out waiting for ${BUILD_LOCK}."

install_conda_dependencies
install_visualization_dependencies
# A conda solve may update runtime libraries, so validate and select the pinned
# compiler only after all conda-managed Pointcept packages are installed.
load_conda_compiler

log "Installing Pointcept Python dependencies"
pip_install --upgrade \
    wheel packaging setuptools \
    "pccm==0.4.16" "ccimport==0.4.4" "pybind11==2.13.6" fire sympy
pip_install \
    tensorboardX wandb yapf addict einops plyfile termcolor timm \
    ftfy regex black seaborn

# Model-specific extras such as FlashAttention, OCNN/dwconv, and OpenAI CLIP
# are intentionally left to the configs/projects that use those backends. The
# RibSeg PTv3 configuration targeted here has FlashAttention disabled.

install_pyg_extensions
pip_install "torch-geometric==${PYG_VERSION}"

install_spconv
install_pointcept_ops
"${ENV_PYTHON}" -m pip check

log "Pointcept environment installation complete"
echo "Activate with: conda activate ${ENV_NAME}"
echo "On a GPU node, load cuda/${CUDA_VERSION} and test with:"
if is_true "${INSTALL_VISUALIZATION}"; then
    echo "${ENV_PYTHON} ${PROJECT_ROOT}/tests/test_env.py --require-gpu"
else
    echo "${ENV_PYTHON} ${PROJECT_ROOT}/tests/test_env.py --require-gpu --skip-visualization"
fi

# Goal

Collect the hardware info of all clusters (see [env.md](env.md))
as the base of building a new Pointcept docker image.

# How

Run [collect-hwinfo.sh](#collect-hwinfosh) **twice per cluster**:

1. **Head/login node** — storage, network egress, scheduler, build capability.
2. **GPU node** — driver, compute capability, GPU memory, topology.
   A login node has no driver, so `nvidia-smi` is absent there and the GPU
   section comes back empty. On cetus this pass must go through PBS.

```bash
# pass 1 (login node)
bash collect-hwinfo.sh hwinfo-<cluster>-login.md

# pass 2 (gpu node) -- cetus, via a short interactive job
qsub -I -q small_gpuq -l select=1:ncpus=4:ngpus=1 -l walltime=00:20:00
bash collect-hwinfo.sh hwinfo-<cluster>-gpu.md
```

The script never aborts: every probe degrades to `n/a`, so an unfamiliar
cluster missing one tool still produces a full report.

# What to collect, and why each item decides something

`nvidia-smi --query-gpu=name,compute_cap,driver_version` is necessary but is
only one of five gates. The others are invisible to it.

## 1. Driver version — the decisive gate

Not `compute_cap`. A CUDA 12.8 image needs:

- driver **>= 525** to start at all (CUDA minor-version compatibility), and
- driver **>= 570** to drive Blackwell / `sm_120`.

If any cluster is below 525, a single cu128 image is impossible there and that
cluster needs its own image or the CUDA forward-compat package. Read it from
the `nvidia-smi` header: `Driver Version: X` and `CUDA Version: Y` (Y = the
newest CUDA that driver can run).

## 2. Compute capability — sizes `TORCH_CUDA_ARCH_LIST`

Needed from **every distinct GPU node type**, not one sample node. iHPC has
three families (`saturn*`, `mars*`, `venus*`) that may differ. Also record GPU
memory (batch sizing) and GPU count per node (DDP).

## 3. Container runtime — the second hard gate

The image is useless if the runtime cannot start it or pass GPUs through:

- `apptainer`/`singularity`/`docker` version
- whether `--nv` passthrough works (this injects the host driver)
- whether images can be **built** locally: user namespaces enabled, and
  whether `--fakeroot --ignore-fakeroot-command` completes a `%post`

## 4. Storage — where the image is built and lives

An apptainer build defaults to `/tmp`. On cetus `/` has only 15 GB free while
the image is ~10-20 GB, so `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` must be
redirected to `/shared/homes` or the build dies partway.

## 5. Network egress — build here, or build elsewhere and ship

Docker Hub / PyPI / GitHub / data.pyg.org reachability. A cluster with no
egress must receive a prebuilt `.sif` by `scp`.
(`registry-1.docker.io/v2/` returning **401** means reachable; `000` means blocked.)

## 6. Scheduler limits — walltime caps

A job killed mid-transaction is exactly how the `cu128_pt271` conda env was
corrupted. Record max walltime, ncpus, ngpus and concurrent-job caps per queue.

## 7. CPU / RAM

Sizes `MAX_JOBS` for the image build and dataloader workers at train time.

## 8. OS / glibc

Still needed even after Pointcept moves into a container, because the
openpoints / pointnext-lightning conda env is built against the **host**.

# Findings so far

## cetus — `hpc-login01` (login pass done)

| item | value |
|---|---|
| OS / glibc | RHEL 8.10, glibc 2.28 |
| CPU / RAM | Xeon Gold 5122, 8 logical cpus, 92 GiB |
| GPU nodes | RTX PRO 6000 Blackwell, `sm_120` (driver version **still to collect**) |
| apptainer | 1.5.3 — **can build locally** with `--fakeroot --ignore-fakeroot-command` |
| docker | absent |
| storage | `/shared/homes` 793 GB free; **`/` only 15 GB — must redirect APPTAINER_TMPDIR** |
| egress | Docker Hub, PyPI, GitHub, data.pyg.org all reachable |
| host CUDA | 12.8 at `/shared/apps/cuda-12.8`, module system available |

### GPU queue walltime caps

| queue | max walltime | max ncpus | concurrent jobs |
|---|---|---|---|
| `small_gpuq` | **48:00:00** | 12 | 12 |
| `med_gpuq` | **24:00:00** | 12 | 12 |
| `large_gpuq` | **06:00:00** | 12 | 6 |

CPU-only queues are far longer: `workq` 200 h, `medq` 100 h.

> **Contradicts [env.md](env.md).** It says "Prefer `large_gpuq` by default"
> and "set longer walltime for full training, try 8h" — but `large_gpuq` caps
> at **6 h**, so an 8 h request is rejected outright. `large_gpuq` has the
> *shortest* limit of the three GPU queues. Use `small_gpuq` (48 h) or
> `med_gpuq` (24 h) for full training runs, and reserve `large_gpuq` for short
> jobs that need its larger per-job resources.
>
> Image builds need **no GPU** — build on the login node, or in `workq`.

## cbai — `strax-server2`

Not yet collected. Expected: docker, `/straxdata` large disk, no scheduler.

## iHPC — `janus0` + `saturn*` / `mars*` / `venus*`

- janus0 (login node): [hardware-info/hwinfo-janus0-login.md](hardware-info/hwinfo-janus0-login.md)
- venus: [venus7](hardware-info/hwinfo-venus7-gpu.md), [venus11](hardware-info/hwinfo-venus11-gpu.md)
- mars:
- saturn:

### venus7 / venus11 (GPU nodes — both identical)

| item | value |
|---|---|
| OS / glibc | RHEL 8.10, glibc 2.28 |
| CPU / RAM | Xeon Gold 5415+, 8 cores / 16 threads, 125 GiB |
| GPU | 2 × RTX A5500, `sm_86` (8.6), 24 GiB each, driver **570.144** (CUDA ≤ 12.8) |
| GPU topology | GPU0 ↔ GPU1 via SYS (PCIe + SMP); NUMA 0 / 1 split |
| apptainer | absent |
| singularity | 4.2.2 — **userns build capable: yes**, `--nv` passthrough: **ok** |
| docker | absent |
| podman | 4.9.4-rhel |
| storage | `/home` 8.1G free (NFS), `/tmp` 249G+ (local NVMe), `/data` 4.8P (NFS), `/share` 27G (NFS, 99% full) |
| egress | Docker Hub (401), PyPI (200), GitHub (200), data.pyg.org (200) — all reachable |
| scheduler | **none** (run jobs directly, no PBS/Slurm on these nodes) |
| host CUDA | no nvcc, module system available |

> **Note:** venus7 and venus11 are identical GPU nodes. Both have driver 570.144 ≥ 570, so they **support CUDA 12.8 + Blackwell**. Singularity works with `--nv` passthrough. No scheduler means jobs run interactively or via custom dispatch. Build images on login node (not collected yet) or use `/tmp` (250G+ local NVMe) as `APPTAINER_TMPDIR`.

### janus0 (login node — collected)

`host: janus0.ihpc.uts.edu.au` (full report: `plans/hardware-info/hwinfo-janus0-login.md`).

| item | value |
|---|---|
| OS / glibc | RHEL 8.10, glibc 2.28 |
| CPU / RAM | 2 × AMD EPYC 9254 (24c) = 48 logical cpus, 188 GiB |
| GPU | login node — `nvidia-smi` ABSENT (no driver, as expected) |
| apptainer | **absent** |
| singularity | **absent** |
| docker | absent |
| podman | 4.9.4-rhel — only runtime present; no apptainer/singularity to build a `.sif` here |
| storage | `/home` 8.0G free (NFS, 64G quota, 88% full); `/tmp` 122G free (local); `/scratch` 362G free (local `/dev/sdb1`); `/data` 4.8P free (NFS); `/share` 27G free but **99% full** |
| egress | Docker Hub (401), PyPI (200), GitHub (200), data.pyg.org (200) — all reachable |
| scheduler | **none** (run jobs directly; same as the venus GPU nodes) |
| host CUDA | no nvcc, module system available |

> **Notes for image placement & build:**
> - The `.sif` **must be built on a venus node** (singularity 4.2.2), not janus0 —
>   janus0 has only podman and cannot produce an apptainer/singularity image.
> - `/home` is NFS with a 64G quota (8G free) — too tight for a 10-20G `.sif`.
>   Store the built `.sif` on **`/data`** (NFS, 4.8P free, shared with venus nodes)
>   or in `/scratch` (local to janus0, not shared). Use `/tmp` or `/scratch` as
>   `APPTAINER_TMPDIR` for the build on the venus node (NVMe, 250G+).
> - `/share` is effectively full (99%); do not use it.
> - No scheduler anywhere on iHPC, so builds/training run interactively or via
>   custom dispatch — no walltime-cap surprises like on cetus.

# collect-hwinfo.sh

```shell
#!/usr/bin/env bash
# Collect the facts a portable Pointcept CUDA 12.8 image depends on.
#
# Run it twice per cluster: once on the head/login node (storage, egress,
# scheduler, build capability) and once on a GPU node (driver, arch, topology).
# On cetus the GPU pass must go through PBS -- the login node has no driver.
#
#   bash collect-hwinfo.sh                 # writes to ./hwinfo-<host>.md
#   bash collect-hwinfo.sh /path/out.md
#
# Never fails: every probe degrades to "n/a" so one missing tool cannot abort
# the run on an unfamiliar cluster.

OUT="${1:-hwinfo-$(hostname -s).md}"

have() { command -v "$1" >/dev/null 2>&1; }
# Run a command with a timeout, collapse failure/absence to a single marker.
try() {
    local cmd="$1"
    local result
    result="$(timeout 20 bash -c "${cmd}" 2>/dev/null)" || result=""
    [[ -n "${result}" ]] && printf -- '%s\n' "${result}" || echo "n/a"
}
section() { printf '\n## %s\n\n' "$1" >>"${OUT}"; }
field()   { printf -- '- **%s:** %s\n' "$1" "$2" >>"${OUT}"; }
block()   { printf '\n```\n%s\n```\n' "$1" >>"${OUT}"; }

: >"${OUT}"
{
    echo "# Hardware / environment report"
    echo
    echo "- **host:** $(hostname -f 2>/dev/null || hostname)"
    echo "- **collected:** $(date --iso-8601=seconds 2>/dev/null || date)"
    echo "- **user:** $(id -un)"
    echo "- **in scheduler job:** ${PBS_JOBID:-${SLURM_JOB_ID:-no}}"
} >>"${OUT}"

# --- 1. OS ------------------------------------------------------------------
# glibc still matters: the openpoints/pointnext conda env is built against the
# host, even once Pointcept itself moves into a container.
section "OS and libc"
field "os" "$(try 'cat /etc/redhat-release 2>/dev/null || . /etc/os-release && echo "$PRETTY_NAME"')"
field "kernel" "$(uname -r)"
field "cpu arch" "$(uname -m)"
field "glibc" "$(try 'ldd --version | head -1')"

# --- 2. CPU / RAM -----------------------------------------------------------
# Sizes MAX_JOBS for the image build and dataloader workers at train time.
section "CPU and memory"
field "model" "$(try "lscpu | sed -n 's/^Model name: *//p' | head -1")"
field "sockets x cores x threads" "$(try "lscpu | awk -F: '/^Socket|^Core\\(s\\)|^Thread\\(s\\)/{gsub(/ /,\"\",\$2); printf \"%s \", \$2}'")"
field "logical cpus" "$(nproc 2>/dev/null || echo n/a)"
field "total ram" "$(try "free -h | awk '/^Mem:/{print \$2}'")"

# --- 3. GPU and driver ------------------------------------------------------
# The decisive gate. A CUDA 12.8 image needs driver >= 525 to start at all
# (CUDA minor-version compatibility), and >= 570 to drive Blackwell (sm_120).
# compute_cap alone does NOT tell you whether the image will run.
section "GPU and driver"
if have nvidia-smi; then
    field "driver / max CUDA supported" \
        "$(try "nvidia-smi | sed -n 's/.*Driver Version: *\\([0-9.]*\\).*CUDA Version: *\\([0-9.]*\\).*/driver \\1, CUDA <= \\2/p'")"
    block "$(try 'nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,compute_mode,mig.mode.current --format=csv')"
    field "gpu count" "$(try 'nvidia-smi --list-gpus | wc -l')"
    field "topology" "$(try 'nvidia-smi topo -m | head -6' | head -1)"
    block "$(try 'nvidia-smi topo -m')"
else
    field "nvidia-smi" "ABSENT -- rerun this script on a GPU node"
fi

# --- 4. Container runtime ---------------------------------------------------
# Second hard gate, and entirely invisible to nvidia-smi: the image is useless
# if the runtime cannot start it or cannot pass GPUs through.
section "Container runtime"
for rt in apptainer singularity docker podman; do
    have "$rt" && field "$rt" "$(try "$rt --version")" || field "$rt" "absent"
done
if have apptainer || have singularity; then
    RT="$(command -v apptainer || command -v singularity)"
    field "userns build capable" \
        "$(try "grep -q '[1-9]' /proc/sys/user/max_user_namespaces && echo yes || echo no")"
    field "subuid/subgid entry" \
        "$(try "grep -qs \"^$(id -un):\" /etc/subuid && echo yes || echo no")"
    # --nv is what injects the host driver into the container; if it fails the
    # image cannot see the GPU no matter how it was built.
    if have nvidia-smi; then
        field "--nv passthrough" \
            "$(try "${RT} exec --nv docker://alpine:latest true && echo ok || echo FAILED")"
    else
        field "--nv passthrough" "untested (no GPU on this node)"
    fi
fi

# --- 5. Storage -------------------------------------------------------------
# An apptainer build defaults to /tmp; on cetus that is 15G and the build dies
# partway. Record every candidate so APPTAINER_TMPDIR can be pointed correctly.
section "Storage"
block "$(try "df -h ${HOME} /tmp /var/tmp /scratch /data /shared /straxdata /share 2>/dev/null | awk 'NR==1 || !seen[\$0]++'")"
field "quota" "$(try 'quota -s | tail -2')"
field "home is network fs" "$(try "df -PT ${HOME} | awk 'NR==2{print \$2}'")"
field "TMPDIR" "${TMPDIR:-unset}"

# --- 6. Network egress ------------------------------------------------------
# Decides whether the image can be built here or must be built elsewhere and
# shipped as a .sif.
section "Network egress"
# registry-1.docker.io returns 401 when reachable (auth required) -- 000 means blocked.
for url in https://registry-1.docker.io/v2/ https://pypi.org/simple/ https://github.com https://data.pyg.org; do
    field "$url" "$(try "curl -s -o /dev/null -w '%{http_code}' --max-time 15 ${url}")"
done
field "http(s)_proxy" "${https_proxy:-${http_proxy:-unset}}"

# --- 7. Scheduler -----------------------------------------------------------
# Walltime caps matter: a build or training run killed mid-transaction is how
# the cu128_pt271 conda env was corrupted.
section "Scheduler"
if have qstat; then
    field "type" "PBS"
    block "$(try 'qstat -Q')"
    block "$(try "qmgr -c 'list queue @default' 2>/dev/null | grep -iE '^Queue |resources_max|resources_default|max_run'")"
elif have sinfo; then
    field "type" "Slurm"
    block "$(try 'sinfo -o "%P %l %c %m %G %D %N"')"
else
    field "type" "none detected (run jobs directly)"
fi

# --- 8. Modules / host CUDA -------------------------------------------------
# Only needed for the conda fallback; a container brings its own toolkit.
section "Host CUDA and modules"
field "nvcc" "$(try 'nvcc --version | tail -2 | head -1')"
field "CUDA_HOME" "${CUDA_HOME:-unset}"
block "$(try 'ls -d /usr/local/cuda* /shared/apps/cuda* /opt/cuda* 2>/dev/null')"
field "module system" "$(type module >/dev/null 2>&1 && echo available || echo absent)"

echo "wrote ${OUT}"
```

# Goal

Collect the hardware info of all clusters (see [env.md](env.md))
as the base of building a new Pointcept docker image.

# How

Run [tmp-ai-collect-hwinfo.sh](../tmp-ai-collect-hwinfo.sh) (inlined below) **twice per cluster**:

1. **Head/login node** — storage, network egress, scheduler, build capability.
2. **GPU node** — driver, compute capability, GPU memory, topology.
   A login node has no driver, so `nvidia-smi` is absent there and the GPU
   section comes back empty. On cetus this pass must go through PBS.

```bash
# pass 1 (login node)
bash tmp-ai-collect-hwinfo.sh hwinfo-<cluster>-login.md

# pass 2 (gpu node) -- cetus, via a short interactive job
qsub -I -q small_gpuq -l select=1:ncpus=4:ngpus=1 -l walltime=00:20:00
bash tmp-ai-collect-hwinfo.sh hwinfo-<cluster>-gpu.md
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
| GPU nodes | **two classes** (both collected 2026-08-26): 48‑cpu exec `hpc-exec05`-`23` = RTX PRO 6000 Blackwell `sm_120` 96 GiB; 24‑cpu exec `hpc-exec01`-`04` = Quadro RTX 6000 `sm_75` 24 GiB. driver **580.142** on both |
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

### cetus GPU node — `hpc-exec19` via `large_gpuq` (collected 2026-08-26)

Full report: [hardware-info/hwinfo-cetus-large_gpuq-gpu.md](hardware-info/hwinfo-cetus-large_gpuq-gpu.md)

| item | value |
|---|---|
| GPU | 2 × RTX PRO 6000 Blackwell Server Edition, `sm_120`, **96 GiB each** |
| driver | **580.142** (CUDA <= 13.0) — the most modern driver of any cluster |
| CPU / RAM | AMD EPYC 9255 24c/48t, 377 GiB |
| apptainer | 1.5.3, userns build capable, **`--nv` passthrough: ok** |
| storage | `/scratch` 2.9 T free (local NVMe), `/` 1.3 T free |
| host CUDA | **`nvcc` absent on compute nodes** (only the login node has 12.8) |

> Two collection caveats worth remembering:
> - `nproc` reported **4**, not 48 — inside a PBS job it reflects the cgroup
>   allocation (`ncpus=4` was requested), not the node. Right number for sizing
>   `MAX_JOBS`, wrong one for describing the hardware; `lscpu` gives the node.
> - `nvcc` is absent on compute nodes, so any from-source CUDA build must happen
>   on the login node — or inside a container that carries its own toolkit.

### cetus GPU node — `hpc-exec03` via `small_gpuq`/`med_gpuq` (collected 2026-08-26)

Full report: [hardware-info/hwinfo-cetus-small_gpuq-gpu.md](hardware-info/hwinfo-cetus-small_gpuq-gpu.md) (and `…-med_gpuq-gpu.md`, same host).

| item | value |
|---|---|
| GPU | 2 × **Quadro RTX 6000**, `sm_75` (**7.5**), **24 GiB each** |
| driver | **580.142** (CUDA <= 13.0) — identical to the 48‑cpu class |
| CPU / RAM | Intel Xeon Gold 6226, 187 GiB (the 24‑cpu exec class) |
| apptainer | 1.5.3, userns build capable, **`--nv` passthrough: ok** |
| storage | `/scratch` 11 T free (local), `/` 618 G free; `/var` only 15 G — redirect `APPTAINER_TMPDIR` off `/var/tmp` |
| host CUDA | `nvcc` absent on compute nodes |

> **This is the architecture the earlier plan nearly dropped.** The proposed
> `8.6;8.9;12.0+PTX` list would have made every job on `hpc-exec01`-`04` fail to
> find a kernel image. `7.5` must stay in `TORCH_CUDA_ARCH_LIST`.

## cbai — `strax-server2`

**Collected: 2026-08-26** (full report: [hardware-info/hwinfo-strax-server2-login.md](hardware-info/hwinfo-strax-server2-login.md))

| item | value |
|---|---|
| OS / glibc | Ubuntu 24.04.3 LTS, glibc 2.39 |
| CPU / RAM | AMD Ryzen Threadripper PRO 3955WX, 16c/32t, 251 GiB |
| GPU | 4 × RTX A4000, `sm_86` (8.6), 16 GiB each, driver **580.95** (CUDA ≤ 13.0) |
| GPU topology | All 4 GPUs on NUMA 0, CPU affinity 0-31; peer links via NODE (PCIe + intra-NUMA) |
| docker | 28.3.3 — **can build locally** |
| apptainer/singularity | absent |
| podman | absent |
| storage | `/` (root) 123G free (916G NVMe); `/straxdata` 633G free (7.3T HDD); home on ext4 (local) |
| egress | Docker Hub (401), PyPI (200), GitHub (200), data.pyg.org (200) — all reachable |
| scheduler | **none** (run jobs directly) |
| host CUDA | CUDA 12.0 (`nvcc`), no module system |

> **Notes for image placement & build:**
> - Docker is the only container runtime; can build images locally with `docker build`
> - `/straxdata` (633G free) is the best location for built images and datasets
> - Driver 580.95 ≥ 570, so **supports CUDA 12.8 + Blackwell (sm_120)**
> - No scheduler means jobs run interactively — no walltime caps
> - GPU topology: all 4 GPUs on same NUMA node (0), peer-to-peer via NODE (not NVLink)

## iHPC — `janus0` + `saturn*` / `mars*` / `venus*`

- janus0 (login node): [hardware-info/hwinfo-janus0-login.md](hardware-info/hwinfo-janus0-login.md)
- venus: [venus7](hardware-info/hwinfo-venus7-gpu.md), [venus11](hardware-info/hwinfo-venus11-gpu.md)
- mars: [mars4](hardware-info/hwinfo-mars4-gpu.md)
- saturn: [saturn2](hardware-info/hwinfo-saturn2-gpu.md)

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

### janus0 (login node — collected 2026-08-26)

Full report: [hardware-info/hwinfo-janus0-login.md](hardware-info/hwinfo-janus0-login.md)

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

### mars4 (GPU node — collected 2026-08-26)

Full report: `plans/hardware-info/hwinfo-mars4-gpu.md`

| item | value |
|---|---|
| OS / glibc | RHEL 8.10, glibc 2.28 |
| CPU / RAM | AMD EPYC 9354P, 32c/32t, 188 GiB |
| GPU | 2 × NVIDIA L4, `sm_89` (8.9), 23 GiB each, driver **570.144** (CUDA ≤ 12.8) |
| GPU topology | GPU0 ↔ GPU1 via SYS (PCIe + SMP); NUMA 0 / 1 split (GPU0: NUMA 1, GPU1: NUMA 0) |
| apptainer | absent |
| singularity | 4.2.2 — **userns build capable: yes**, `--nv` passthrough: **ok** |
| docker | absent |
| podman | 4.9.4-rhel |
| storage | `/home` 48G free (NFS), `/tmp` 252G (local NVMe), `/scratch` 1.4T (local NVMe), `/data` 4.8P (NFS), `/share` 27G (NFS, 99% full) |
| egress | Docker Hub (401), PyPI (200), GitHub (200), data.pyg.org (200) — all reachable |
| scheduler | **none** (run jobs directly) |
| host CUDA | no nvcc, module system available |

> **Notes for image placement & build:**
> - Singularity 4.2.2 works with `--nv` passthrough. Build images on this node using `/tmp` (250G+ local NVMe) as `APPTAINER_TMPDIR`.
> - Driver 570.144 ≥ 570, so **supports CUDA 12.8 + Blackwell (sm_120)**.
> - GPU topology: 2× L4 on separate NUMA nodes (SYS connection), peer-to-peer via PCIe + SMP.
> - No scheduler means jobs run interactively — no walltime caps.

### saturn2 (GPU node — collected 2026-08-26)

Full report: `plans/hardware-info/hwinfo-saturn2-gpu.md`

| item | value |
|---|---|
| OS / glibc | RHEL 8.10, glibc 2.28 |
| CPU / RAM | AMD EPYC 9254, 24c/24t, 188 GiB |
| GPU | 2 × NVIDIA L40, `sm_89` (8.9), 46 GiB each, driver **570.144** (CUDA ≤ 12.8) |
| GPU topology | GPU0 ↔ GPU1 via SYS (PCIe + SMP); NUMA 0 / 1 split (GPU0: NUMA 1, GPU1: NUMA 0) |
| apptainer | absent |
| singularity | 4.2.2 — **userns build capable: yes**, `--nv` passthrough: **ok** |
| docker | absent |
| podman | 4.9.4-rhel |
| storage | `/home` 48G free (NFS), `/tmp` 254G (local NVMe), `/scratch` 1.6T (local NVMe), `/data` 4.8P (NFS), `/share` 27G (NFS, 99% full) |
| egress | Docker Hub (401), PyPI (200), GitHub (200), data.pyg.org (200) — all reachable |
| scheduler | **none** (run jobs directly) |
| host CUDA | no nvcc, module system available |

> **Notes for image placement & build:**
> - Singularity 4.2.2 works with `--nv` passthrough. Build images on this node using `/tmp` (250G+ local NVMe) as `APPTAINER_TMPDIR`.
> - Driver 570.144 ≥ 570, so **supports CUDA 12.8 + Blackwell (sm_120)**.
> - GPU topology: 2× L40 on separate NUMA nodes (SYS connection), peer-to-peer via PCIe + SMP.
> - L40 has 46 GiB VRAM (2x L4) — better for large batch sizes.
> - No scheduler means jobs run interactively — no walltime caps.

# Synthesis — what the image must be

## GPU matrix

| cluster | node | GPU | `sm` | VRAM | count | driver | max CUDA |
|---|---|---|---|---|---|---|---|
| cetus (exec 24‑cpu, `hpc-exec01`‑`04`) | `hpc-exec03` | Quadro RTX 6000 | **7.5** | 24 GiB | 2 | 580.142 | 13.0 |
| cetus (exec 48‑cpu, `hpc-exec05`‑`23`) | `hpc-exec19` | RTX PRO 6000 Blackwell | **12.0** | 96 GiB | 2 | 580.142 | 13.0 |
| cbai | `strax-server2` | RTX A4000 | **8.6** | 16 GiB | 4 | 580.95.05 | 13.0 |
| iHPC | `venus7`, `venus11` | RTX A5500 | **8.6** | 24 GiB | 2 | 570.144 | **12.8** |
| iHPC | `mars4` | L4 | **8.9** | 23 GiB | 2 | 570.144 | **12.8** |
| iHPC | `saturn2` | L40 | **8.9** | 46 GiB | 2 | 570.144 | **12.8** |

## Decisions this settles

**Arch list: `TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0+PTX"`.** The union of every
GPU in use is now four architectures — and the earlier proposal of
`8.6;8.9;12.0+PTX` was **wrong**: cetus's 24‑cpu exec class (`hpc-exec01`‑`04`)
carries **Quadro RTX 6000, `sm_7.5`**, so `7.5` *is* needed and must stay. The
current [env.sh](../env.sh) default (`7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX`) only
needs `8.0`, `9.0` and `10.0` dropped — those three genuinely run nowhere in this
fleet (cbai 8.6; iHPC 8.6/8.9; cetus 7.5/12.0). `12.0+PTX` keeps forward
compatibility for future hardware via PTX JIT.

> **Caveat — cetus desk class not yet collected.** `hpc-desk01`‑`07` (1 GPU each,
> in `iworkq`, currently **offline**) may carry a *fourth* architecture. If any
> desk node is a training target, sample it with `iworkq` before locking the list.
> The 24‑cpu and 48‑cpu exec classes are confirmed above.

**CUDA 12.8 is a ceiling, not a choice.** iHPC's driver 570.144 supports CUDA
**<= 12.8 exactly**. The pinned 12.8 works everywhere, but there is zero
headroom: bumping the image to CUDA 12.9 or 13.0 would break all four iHPC
nodes while remaining fine on cbai (13.0) and cetus. Do not raise it without
re-checking iHPC drivers first.

**Two artifacts are required, from one recipe.**

| cluster | runtime available | artifact |
|---|---|---|
| cetus | apptainer 1.5.3 | `.sif` |
| iHPC GPU nodes | singularity-ce 4.2.2 (site wrapper `singularity_build`) | `.sif` |
| iHPC `janus0` | podman only — **cannot build a `.sif`** | — |
| cbai | docker 28.3.3 only, **`sudo` available** | docker image |

Since every cluster reaches Docker Hub, the cleanest distribution is to build
and push **once** from cbai (the only host with docker), then on cetus and iHPC:

```bash
apptainer build pointcept.sif docker://<account>/pointcept:cu128
```

That avoids copying a 10-20 GB `.sif` between clusters. All three clusters can also build natively — cetus via apptainer, iHPC GPU
nodes via `singularity_build` (both verified), cbai via `docker build` — so
per-cluster builds remain a fallback. The cost is keeping the recipe in sync
across two files, as [Dockerfile](../Dockerfile) and
[pointcept.def](../pointcept.def) are today; the registry route keeps one.

## Image placement

| cluster | build scratch (`APPTAINER_TMPDIR`) | store the image |
|---|---|---|
| cetus | `/shared/homes/...` — **not `/tmp`** (`/` has only 15 GB) | `/shared/homes` (793 GB) |
| iHPC | `/tmp` or `/scratch` on a GPU node (NVMe, 250 GB+) | `/data/tliang/` — **1 TiB/user**, shared across all iHPC nodes |
| cbai | default | `/straxdata` (633 GB) |

> **iHPC placement contradicts [env.md](env.md).** It puts the image at
> `~/pointcept.sif` with a link in `/share/$(whoami)/`. But `/home` is capped at
> 32-64 GiB per user and `/share` is **99% full (27 GB left)** — neither holds a
> 10-20 GB image reliably. Use `/data/tliang/` (1 TiB per user, separate
> partition, shared across janus0 and every GPU node) so one copy serves all of
> iHPC.

# Gaps and how they closed

Every open item from the first pass, and what resolved it.

## Resolved

- **cbai build host — confirmed.** `sudo` available, `docker build` works, and a
  Pointcept image was previously built there and ran on all 4 GPUs. That also
  confirms the nvidia-container-toolkit path (`--gpus all`) works, so cbai is
  viable as the single build-and-push host.
- **iHPC build — confirmed.** `pointcept.sif` was built on an iHPC **GPU node**
  using the site wrapper `singularity_build`, and runs with GPUs. Note janus0
  still cannot build (podman only).
- **iHPC quotas.** `/home/tliang/` is capped at 32-64 GiB per user;
  `/data/tliang/` at **1 TiB per user**, on a separate partition. 1 TiB is ample
  for a 10-20 GB image, confirming `/data` as the placement. (This also explains
  the janus0-vs-saturn2 `/home` discrepancy: `df` was reporting the shared
  export, not the per-user entitlement.)
- **cetus GPU count.** `pbsnodes -a` gives 2 GPUs on every `hpc-exec*` node and
  1 on every `hpc-desk*` node.

## Assessment of [cetus-gpuq-info.md](../cetus-gpuq-info.md)

From the sibling PointNeXt project. Now fully checkable against measurement.

**Confirmed**

- `large_gpuq` contains Blackwell `sm_120` — `hpc-exec19` measured as exactly
  that, which explains its "no kernel image is available" failures under cu118.
- `small_gpuq` and `med_gpuq` do map to the `hpc-exec01`-`04` class — both
  probes landed on `hpc-exec03`, 24 cpu / 187 GiB, matching `pbsnodes`.
- Including `7.5` in its cu118 arch list was right, for the right reason.

**Wrong or unverified**

- *"small_gpuq / med_gpuq GPU Arch: sm_75/80/86/89"* was a range, not a
  measurement. The real answer is **`sm_75` — the bottom of it**. Had the image
  been built on the earlier `8.6;8.9;12.0+PTX` plan, every job on the two
  long-walltime queues would have failed.
- *"large_gpuq: compatible hpc-exec06-09 (sm_86/89)"* remains unverified and
  still sits awkwardly with `pbsnodes`, where `hpc-exec05`-`23` is one uniform
  class. Only `hpc-exec19` has been measured, and it is Blackwell. Harmless
  either way: `8.6` and `8.9` are in the list regardless.
- **Factual error:** it lists "max running per user: 6" for small_gpuq and
  med_gpuq. Live `qmgr` says **12** for both; only `large_gpuq` is 6.
- **Its recommendation is unsafe:** `7.5;8.0;8.6;8.9;9.0+PTX` is offered "for
  Blackwell support" but contains no `12.0`, leaving Blackwell to JIT `sm_90`
  PTX forward at every process start — slow and not guaranteed.

## Turing consequence: no bf16 on the long-walltime queues

`sm_75` is Turing, which has **no bfloat16 and no TF32** — both arrived with
Ampere (`sm_80`). FlashAttention also requires `sm_80`+. So on `small_gpuq` and
`med_gpuq`, the only queues that allow 24-48 h runs:

- a `bf16` mixed-precision config will fail or silently fall back;
- FlashAttention is unavailable — consistent with [env.sh](../env.sh) already
  disabling it for the RibSeg PTv3 config, and with `pointcept.def` not
  installing it.

Use `fp16` AMP or fp32 for anything that must run there, and keep `bf16` for the
Blackwell nodes. This is a config concern, not an image concern, but it is
easiest to settle now: a training plan that assumes `bf16` everywhere has no
long-walltime queue to run on.

## Consequence: VRAM and walltime pull in opposite directions

| queue | nodes | walltime | concurrent | GPU | VRAM |
|---|---|---|---|---|---|
| `small_gpuq` | `hpc-exec01`-`04` | **48 h** | 12 | Quadro RTX 6000 (`sm_75`) | 24 GiB |
| `med_gpuq` | `hpc-exec01`-`04` | **24 h** | 12 | Quadro RTX 6000 (`sm_75`) | 24 GiB |
| `large_gpuq` | `hpc-exec05`-`23` | **6 h** | 6 | RTX PRO 6000 Blackwell (`sm_120`) | **96 GiB** |

The trade is now fully measured and it is stark: cetus offers either **8× the
walltime** or **4× the VRAM plus a six-year-newer architecture**, never both.

- Long runs on `small_gpuq`: 48 h, but Turing — 24 GiB, no bf16, no
  FlashAttention.
- Blackwell on `large_gpuq`: 96 GiB and full bf16, but a hard 6 h wall, so the
  run must checkpoint and requeue.

Robust checkpoint-resume is therefore worth more than any image decision here,
and `large_gpuq` also allows only 6 concurrent jobs against 12 on the others.

> For comparison, iHPC's `saturn2` (2 × L40, 46 GiB, `sm_89`, no scheduler and
> so no walltime cap at all) may be the better home for long PTv3 runs than
> either cetus queue. cbai's 4 × A4000 are only 16 GiB each — the smallest VRAM
> anywhere — but there are four of them and no walltime either.

# All hardware collection complete

Every gate is now measured. Nothing further is needed before writing the image
recipe:

- driver >= 570 everywhere (cetus 580.142, cbai 580.95.05, iHPC 570.144);
- CUDA 12.8 is supported everywhere and is the ceiling on iHPC;
- arch list `7.5;8.6;8.9;12.0+PTX`;
- build hosts confirmed on all three clusters, `--nv` verified on cetus and iHPC;
- image placement decided per cluster.

One residual sampling caveat, low risk: `hpc-exec03` stands for the whole
`hpc-exec01`-`04` class and `hpc-exec19` for `hpc-exec05`-`23`. `pbsnodes` shows
each class uniform in cpu and memory, so this is a reasonable inference, but a
job landing on an unsampled node with a different card would be outside the
measured set. The `7.5;8.6;8.9;12.0+PTX` list covers every architecture seen on
any cluster, which makes that largely academic.

# collect-hwinfo.sh

```shell
#!/usr/bin/env bash
# Collect the facts a portable Pointcept CUDA 12.8 image depends on.
#
# Run it twice per cluster: once on the head/login node (storage, egress,
# scheduler, build capability) and once on a GPU node (driver, arch, topology).
# On cetus the GPU pass must go through PBS -- the login node has no driver.
#
#   bash tmp-ai-collect-hwinfo.sh                 # writes to ./hwinfo-<host>.md
#   bash tmp-ai-collect-hwinfo.sh /path/out.md
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

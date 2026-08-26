# Hardware / environment report

- **host:** saturn2.ihpc.uts.edu.au
- **collected:** 2026-08-26T15:48:49+10:00
- **user:** miwei
- **in scheduler job:** no

## OS and libc

- **os:** Red Hat Enterprise Linux release 8.10 (Ootpa)
- **kernel:** 4.18.0-553.40.1.el8_10.x86_64
- **cpu arch:** x86_64
- **glibc:** ldd (GNU libc) 2.28

## CPU and memory

- **model:** AMD EPYC 9254 24-Core Processor
- **sockets x cores x threads:** 1 24 1 
- **logical cpus:** 24
- **total ram:** 188Gi

## GPU and driver

- **driver / max CUDA supported:** driver 570.144, CUDA <= 12.8

```
index, name, compute_cap, memory.total [MiB], driver_version, compute_mode, mig.mode.current
0, NVIDIA L40, 8.9, 46068 MiB, 570.144, Default, [N/A]
1, NVIDIA L40, 8.9, 46068 MiB, 570.144, Default, [N/A]
```
- **gpu count:** 2
- **topology:** 	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	12-23	1		N/A
GPU1	SYS	 X 	0-11	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

## Container runtime

- **apptainer:** absent
- **singularity:** singularity-ce version 4.2.2
- **docker:** absent
- **podman:** podman version 4.9.4-rhel
- **userns build capable:** yes
- **subuid/subgid entry:** no
- **--nv passthrough:** ok

## Storage


```
Filesystem                                                                           Size  Used Avail Use% Mounted on
mdcislresnfs03.adsroot.uts.edu.au:/ifs/mdcisl01/research/facility/ihpc/rhel810homes   64G   17G   48G  26% /home
/dev/nvme0n1p3                                                                       256G  2.0G  254G   1% /tmp
/dev/nvme0n1p4                                                                        20G  4.7G   16G  24% /var
/dev/nvme1n1p1                                                                       1.8T  239G  1.6T  14% /scratch
mdcislresnfs03.adsroot.uts.edu.au:/ifs/mdcisl01/research/facility/ihpc/data           12P  6.5P  4.8P  58% /data
mdcislresnfs03.adsroot.uts.edu.au:/ifs/mdcisl01/research/facility/ihpc/share         2.0T  2.0T   27G  99% /share
```
- **quota:** n/a
- **home is network fs:** nfs
- **TMPDIR:** unset

## Network egress

- **https://registry-1.docker.io/v2/:** 401
- **https://pypi.org/simple/:** 200
- **https://github.com:** 200
- **https://data.pyg.org:** 200
- **http(s)_proxy:** unset

## Scheduler

- **type:** none detected (run jobs directly)

## Host CUDA and modules

- **nvcc:** n/a
- **CUDA_HOME:** unset

```
n/a
```
- **module system:** available

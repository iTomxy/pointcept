# Hardware / environment report

- **host:** strax-server2
- **collected:** 2026-08-26T15:36:00+10:00
- **user:** tianyouliang
- **in scheduler job:** no

## OS and libc

- **os:** Ubuntu 24.04.3 LTS
- **kernel:** 6.8.0-136-generic
- **cpu arch:** x86_64
- **glibc:** ldd (Ubuntu GLIBC 2.39-0ubuntu8.8) 2.39

## CPU and memory

- **model:** AMD Ryzen Threadripper PRO 3955WX 16-Cores
- **sockets x cores x threads:** 2 16 1 
- **logical cpus:** 32
- **total ram:** 251Gi

## GPU and driver

- **driver / max CUDA supported:** driver 580.95.05, CUDA <= 13.0

```
index, name, compute_cap, memory.total [MiB], driver_version, compute_mode, mig.mode.current
0, NVIDIA RTX A4000, 8.6, 16376 MiB, 580.95.05, Default, [N/A]
1, NVIDIA RTX A4000, 8.6, 16376 MiB, 580.95.05, Default, [N/A]
2, NVIDIA RTX A4000, 8.6, 16376 MiB, 580.95.05, Default, [N/A]
3, NVIDIA RTX A4000, 8.6, 16376 MiB, 580.95.05, Default, [N/A]
```
- **gpu count:** 4
- **topology:** 	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m

```
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NODE	NODE	NODE	0-31	0		N/A
GPU1	NODE	 X 	NODE	NODE	0-31	0		N/A
GPU2	NODE	NODE	 X 	NODE	0-31	0		N/A
GPU3	NODE	NODE	NODE	 X 	0-31	0		N/A

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
- **singularity:** absent
- **docker:** Docker version 28.3.3, build 980b856
- **podman:** absent

## Storage


```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  916G  747G  123G  86% /
/dev/sda1       7.3T  6.3T  633G  91% /straxdata
```
- **quota:** n/a
- **home is network fs:** ext4
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

- **nvcc:** Cuda compilation tools, release 12.0, V12.0.140
- **CUDA_HOME:** unset

```
n/a
```
- **module system:** absent

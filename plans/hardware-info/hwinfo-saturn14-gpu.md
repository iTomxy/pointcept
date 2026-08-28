# Hardware / environment report

- **host:** saturn14.ihpc.uts.edu.au
- **collected:** 2026-08-28T13:52:52+10:00
- **user:** miwei
- **in scheduler job:** no

## OS and libc

- **os:** Red Hat Enterprise Linux release 8.10 (Ootpa)
- **kernel:** 4.18.0-553.40.1.el8_10.x86_64
- **cpu arch:** x86_64
- **glibc:** ldd (GNU libc) 2.28

## CPU and memory

- **model:** Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz
- **sockets x cores x threads:** 1 12 2 
- **logical cpus:** 24
- **total ram:** 187Gi

## GPU and driver

- **driver / max CUDA supported:** driver 570.144, CUDA <= 12.8

```
index, name, compute_cap, memory.total [MiB], driver_version, compute_mode, mig.mode.current
0, Tesla V100-PCIE-32GB, 7.0, 32768 MiB, 570.144, Default, [N/A]
1, Tesla V100-PCIE-32GB, 7.0, 32768 MiB, 570.144, Default, [N/A]
```
- **gpu count:** 2
- **topology:** 	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m

```
	[4mGPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	SYS	0,2,4,6,8,10	0		N/A
GPU1	SYS	 X 	1,3,5,7,9,11	1		N/A

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
/dev/sdb1                                                                            256G  2.4G  254G   1% /tmp
/dev/sda3                                                                             20G  4.9G   16G  25% /var
/dev/sdb2                                                                            2.0T   45G  1.9T   3% /scratch
mdcislresnfs03.adsroot.uts.edu.au:/ifs/mdcisl01/research/facility/ihpc/data           12P  6.6P  4.7P  59% /data
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

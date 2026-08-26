# Hardware / environment report

- **host:** hpc-exec03.hpc.uts.edu.au
- **collected:** 2026-08-26T16:48:44+10:00
- **user:** u24647087
- **in scheduler job:** 66616.hpc-head01

## OS and libc

- **os:** Red Hat Enterprise Linux release 8.10 (Ootpa)
- **kernel:** 4.18.0-553.126.1.el8_10.x86_64
- **cpu arch:** x86_64
- **glibc:** ldd (GNU libc) 2.28

## CPU and memory

- **model:** Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz
- **sockets x cores x threads:** 1 12 2 
- **logical cpus:** 4
- **total ram:** 187Gi

## GPU and driver

- **driver / max CUDA supported:** driver 580.142, CUDA <= 13.0

```
index, name, compute_cap, memory.total [MiB], driver_version, compute_mode, mig.mode.current
0, Quadro RTX 6000, 7.5, 24576 MiB, 580.142, Default, [N/A]
1, Quadro RTX 6000, 7.5, 24576 MiB, 580.142, Default, [N/A]
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

- **apptainer:** apptainer version 1.5.3-3.el8
- **singularity:** apptainer version 1.5.3-3.el8
- **docker:** absent
- **podman:** absent
- **userns build capable:** yes
- **subuid/subgid entry:** no
- **--nv passthrough:** ok

## Storage


```
Filesystem                 Size  Used Avail Use% Mounted on
10.37.24.191:/cetus_homes  1.0T  231G  794G  23% /shared/homes
/dev/sda5                  1.1T  476G  618G  44% /
/dev/sda3                   16G  1.7G   15G  11% /var
/dev/sdb1                   11T  322G   11T   3% /scratch
```
- **quota:** n/a
- **home is network fs:** nfs4
- **TMPDIR:** /var/tmp/pbs.66616.hpc-head01

## Network egress

- **https://registry-1.docker.io/v2/:** 401
- **https://pypi.org/simple/:** 200
- **https://github.com:** 200
- **https://data.pyg.org:** 200
- **http(s)_proxy:** unset

## Scheduler

- **type:** PBS

```
Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
interq               0     2 yes yes     0     2     0     0     0     0 Exe*
medq                 0     0 yes yes     0     0     0     0     0     0 Exe*
smallq               0     1 yes yes     0     0     1     0     0     0 Exe*
workq                0    48 yes yes     0    38     2     0     0     0 Exe*
iworkq               0    19 yes yes     6     1     0    12     0     0 Exe*
small_gpuq           0    17 yes yes    10     4     3     0     0     0 Exe*
med_gpuq             0    11 yes yes     9     1     1     0     0     0 Exe*
large_gpuq           0    41 yes yes     7    27     5     0     0     0 Exe*
teachingq            0     0 yes yes     0     0     0     0     0     0 Exe*
cyber                0     1 yes yes     0     1     0     0     0     0 Exe*
```

```
Queue interq
    resources_max.mem = 100gb
    resources_max.ncpus = 16
    resources_max.walltime = 02:00:00
    resources_default.walltime = 01:00:00
    max_run = [u:PBS_GENERIC=2]
Queue medq
    resources_max.mem = 64gb
    resources_max.ncpus = 16
    resources_max.walltime = 100:00:00
    resources_default.walltime = 48:00:00
    max_run = [u:PBS_GENERIC=30]
Queue smallq
    resources_max.mem = 32gb
    resources_max.ncpus = 4
    resources_max.walltime = 24:00:00
    resources_default.walltime = 24:00:00
    max_run = [u:PBS_GENERIC=60]
Queue workq
    resources_max.walltime = 200:00:00
    resources_default.walltime = 120:00:00
    max_run = [u:PBS_GENERIC=20]
Queue iworkq
    resources_max.ndesktops = 1
    resources_max.walltime = 04:00:00
    resources_default.arch = linux
    resources_default.place = free
Queue small_gpuq
    resources_max.ncpus = 12
    resources_max.walltime = 48:00:00
    resources_default.mem = 10gb
    resources_default.walltime = 10:00:00
    max_run = [u:PBS_GENERIC=12]
Queue med_gpuq
    resources_max.ncpus = 12
    resources_max.walltime = 24:00:00
    resources_default.mem = 10gb
    resources_default.walltime = 12:00:00
    max_run = [u:PBS_GENERIC=12]
Queue large_gpuq
    resources_max.ncpus = 12
    resources_max.walltime = 06:00:00
    resources_default.mem = 10gb
    resources_default.walltime = 06:00:00
    max_run = [u:PBS_GENERIC=6]
Queue teachingq
    resources_max.mem = 50gb
    resources_max.ncpus = 60
    resources_default.walltime = 00:15:00
    max_run = [u:PBS_GENERIC=6]
Queue cyber
    resources_max.ncpus = 64
    resources_default.mem = 10gb
    resources_default.walltime = 200:00:00
    max_run = [u:PBS_GENERIC=100]
```

## Host CUDA and modules

- **nvcc:** n/a
- **CUDA_HOME:** unset

```
n/a
```
- **module system:** available

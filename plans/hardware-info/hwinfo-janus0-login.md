# Hardware / environment report

- **host:** janus0.ihpc.uts.edu.au
- **collected:** 2026-08-26T15:30:06+10:00
- **user:** tliang
- **in scheduler job:** no

## OS and libc

- **os:** Red Hat Enterprise Linux release 8.10 (Ootpa)
- **kernel:** 4.18.0-553.40.1.el8_10.x86_64
- **cpu arch:** x86_64
- **glibc:** ldd (GNU libc) 2.28

## CPU and memory

- **model:** AMD EPYC 9254 24-Core Processor
- **sockets x cores x threads:** 2 24 1 
- **logical cpus:** 48
- **total ram:** 188Gi

## GPU and driver

- **nvidia-smi:** ABSENT -- rerun this script on a GPU node

## Container runtime

- **apptainer:** absent
- **singularity:** absent
- **docker:** absent
- **podman:** podman version 4.9.4-rhel

## Storage


```
Filesystem                                                                           Size  Used Avail Use% Mounted on
mdcislresnfs03.adsroot.uts.edu.au:/ifs/mdcisl01/research/facility/ihpc/rhel810homes   64G   57G  8.0G  88% /home
/dev/sda3                                                                            128G  6.5G  122G   6% /tmp
/dev/sda4                                                                             20G  1.5G   19G   8% /var
/dev/sdb1                                                                            447G   86G  362G  20% /scratch
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

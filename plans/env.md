Statement of several possible computing server environment.

# Servers

Agents are possible to be launched on 3 servers.
Determine by host name and go for the corresponding detail statements.

1. [iHPC](#ihpc): `janus0` is the login node, others (`saturn*`, `mars*` and `venus*`) are gpu nodes.
2. [cetus](#cetus): `hpc-login01`, login node.
3. [cbai](#cbai): `strax-server2`, gpu node.

# iHPC

- Use singularity, not conda env.
    - Image is `~/pointcept.sif`, also soft-linked at `/share/$(whoami)/pointcept.sif`.
    - Refer to [singh.sh](../singu.sh).

## gpu

- Do not run experiments on the login node `janus0`.
- Prefer using saturn cuz the gpu memories on them are larger than mars and venus.
- Each gpu node has 2 gpus.
Use both of them for distributed training.
If not both gpus are of low usage (e.g. gpu memory or load <30%),
or cpu, memory are busy/heavily occupied by other users
do not just wait there or compete for the limited resource,
but 1) record progress and handover instructions,
and 2) tell me with e-mail, let me switch to a free gpu node.

## disk

- The home directory `~` disk space is limited to 64GiB.
We should avoid putting logs, predictions and checkpoints there.
- The large disk space is /data/`whoami`/,
which is soft-linked to ~/data/.
The root log path, where logs/predictions/checkpoints are placed, for this project is
~/data/project-data/pointcept/exp/,
soft-linked to [exp/](../exp).

# cetus

- Do not run experiments on the login node `hpc-login01`.
- Submit jobs via PBS system.
See ~/uts-cetus-pbs.md for statements.
You can submit multiple jobs for parallel experiments.
    - May need to add `#PBS` commands in those driver running scripts if not added yet.
    But do not directly add to the scripts under [scripts/](../scripts/).
    - Prefer `large_gpuq` by default.
    - Set longer `walltime` for full training.
    If no ETA is available for reference, try 8h.

- The code folder should already sit on a large disk (/shared/homes/).
The root log path can be placed under it directly.
- Use singularity, not conda env.
    - Image is `~/pointcept.sif`, also soft-linked at `/share/$(whoami)/pointcept.sif`.
    - Refer to [singh.sh](../singu.sh).

# cbai

- Use all gpus that are NOT occupied by other users for distributed experiment.
- The code folder should already sit on a large disk (/straxdata/).
The root log path can be placed under it directly.
- Use docker, not conda env.
    - Image: `pointcept:local`.
    - Refer to [dock.sh](../dock.sh).

# General Guidance

- `tmux` and `screen` are available.
Agents should use one of them to manage long tasks,
e.g. experiments and data processing,
instead of directly executing in foreground to avoid getting stuck.
- Env: use the container image defined by [cu128_pt271.Dockerfile](../containers/cu128_pt271.Dockerfile) or [pointcept-cu128_pt271.def](../containers/pointcept-cu128_pt271.def),
not conda env.
- For any helper scripts (python and bash),
agents should follow the file name patterns: `tmp-ai*.sh` and `tmp-ai*.py`,
without number limit.
They are ignored by git.
- Before editting shell scripts (*.sh),
check whether it is in use/running first.
For safety,
you can create a new inode of the shell script to be editted,
edit on the new inode,
and `mv` it back.
This trick avoids interrupting the running progress if the script is running.
Unless you want the modification affect in a future period of the very current run
and you confirm that it does not interrupt the running command,
e.g. the modification happens after the current running command.
- Use `~/mail.sh` to send e-mail to me.
- Data root path: ~/data/.

#!/bin/bash
# launch an interactive docker environment (equivalent of singu.sh) on Curvebeam AI's server.
#
# Usage:
#   ./dock.sh                        # interactive shell
#   ./dock.sh bash run_ribseg.sh     # run an experiment script
#   ./dock.sh /opt/conda/bin/python script.py
#
# Note: this host has no nvidia-container-toolkit, so `--gpus all` is not
# available; GPUs are passed through manually via --device + driver lib mounts.

IMAGE=pointcept:local
PROJ=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

# -it only when stdin is a terminal (so non-interactive/background runs work)
TTY=()
if [ -t 0 ]; then TTY=(-it); fi

docker run --rm "${TTY[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HOME="$HOME" \
  --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3 \
  --device /dev/nvidiactl --device /dev/nvidia-modeset \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05:/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05 \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.95.05:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.95.05 \
  -v /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia \
  -v "$HOME:$HOME" \
  -v "$PROJ:$PROJ" \
  -v /straxdata:/straxdata \
  --workdir "$PROJ" \
  --shm-size=16g \
  "$IMAGE" "$@"

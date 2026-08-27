# Pointcept Docker Image
# FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel
FROM tyloeng/pointcept:py3.11-torch2.7.1-cu12.8-cudnn9

# Set labels (equivalent to Singularity %labels)
LABEL python.version="3.11" \
    cuda.version="12.8" \
    cudnn.version="9" \
    pytorch.version="2.7.1+cu128" \
    torchvision.version="0.22.1+cu128"

# ENV DEBIAN_FRONTEND=noninteractive
# RUN mkdir -p /data /projects /scratch

# Update pip and install additional packages (base image already has pyyaml, open3d, etc.)
# Use conda for better version compatibility, pip for packages not in conda
RUN conda install -y -c conda-forge \
    scikit-learn scikit-image opencv \
    simpleitk nibabel medpy itk jupyterlab ipykernel seaborn \
&& pip install --no-cache-dir \
    opencv-python-headless \
&& python -m ipykernel install --name "pointcept" --display-name "Pointcept (docker)" \
&& ln -s /usr/local/bin/python /usr/local/bin/py \
&& ln -s /usr/local/bin/pip /usr/local/bin/pi

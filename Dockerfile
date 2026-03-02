# Pointcept Docker Image
FROM pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

# Set labels (equivalent to Singularity %labels)
LABEL python.version="3.11.10" \
    cuda.version="12.4" \
    cudnn.version="9" \
    pytorch.version="2.5.0+cu124" \
    torchvision.version="0.20.0+cu124"

# ENV DEBIAN_FRONTEND=noninteractive
# RUN mkdir -p /data /projects /scratch

# Update pip and remove conflicting package
RUN /opt/conda/bin/pip install --upgrade pip setuptools wheel \
&& /opt/conda/bin/pip uninstall -y open3d-cpu \
&& /opt/conda/bin/pip install --no-cache-dir \
    scikit-learn scikit-image opencv-python-headless pyyaml open3d \
    SimpleITK nibabel medpy itk jupyterlab ipykernel seaborn \
&& /opt/conda/bin/python -m ipykernel install --name "pointcept" --display-name "Pointcept (docker)" \
&& ln -s /opt/conda/bin/python /usr/local/bin/py \
&& ln -s /opt/conda/bin/pip /usr/local/bin/pi

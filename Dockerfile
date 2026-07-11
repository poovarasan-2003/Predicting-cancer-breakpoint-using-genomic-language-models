# Use a base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    libncurses5-dev \
    libncursesw5-dev \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install core scientific and genomic packages
# NOTE: transformers is pinned to 4.39.3 for Caduceus compatibility
RUN pip install pandas pybedtools scipy statsmodels pyfaidx intervaltree biopython tqdm scikit-learn matplotlib seaborn transformers==4.39.3 datasets psutil umap-learn packaging

# Install optimized Mamba kernels (Building from source inside the container)
# We use --no-build-isolation to ensure they use the already installed Torch
RUN pip install causal-conv1d==1.6.1 --no-build-isolation
RUN pip install mamba-ssm==2.3.1 --no-build-isolation

# Set the working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["bash"]

# PyTorch 2.3.1 with CUDA 12.1 — compatible with Tufts HPC's CUDA 12.2 driver
# NOTE: Do NOT upgrade to cuda12.4+ or it will fail on the cluster
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System dependencies for OpenCV, MuJoCo, and rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo 2.1 binary — required by mujoco-py
RUN mkdir -p /opt/mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz && \
    tar -xzf /tmp/mujoco210.tar.gz -C /opt/mujoco && \
    rm /tmp/mujoco210.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV NUMBA_CACHE_DIR=/tmp/numba-cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

# conda ships an older libstdc++ that lacks GLIBCXX_3.4.30, which libLLVM-15
# (pulled in by mujoco-py) requires. Symlink the system copy (Ubuntu 22.04 / gcc-12)
# over the conda one so the right version is found at import time.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

WORKDIR /workspace

# Install Python dependencies (torch is already in base image — do not add it here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Restore CUDA-compatible torch and torchvision. requirements.txt may downgrade torch
# (some package pulls an older version from PyPI) and pull in an older nvidia-nccl-cu12
# that lacks ncclCommRegister. Reinstalling without --no-deps lets pip also restore the
# correct nvidia-* libs (nccl, cublas, etc.) that torch 2.3.1+cu121 was built against.
RUN pip install --force-reinstall \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pre-compile mujoco-py so it doesn't try to build at runtime
RUN python -c "import mujoco_py"

# Copy repo and install HPT package
COPY . .
RUN pip install -e . --no-deps

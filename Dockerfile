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
    libglfw3 \
    libglew-dev \
    patchelf \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (torch is already in base image — do not add it here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo and install HPT package
COPY . .
RUN pip install -e . --no-deps

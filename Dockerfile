# Base image with NVIDIA CUDA 12.2 and cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & OpenGL headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace/medical-science

# Copy requirements and install
COPY requirements-train.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-train.txt

# Copy codebase
COPY . .

# Set environment variables for JAX GPU allocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV JAX_PLATFORMS=cuda,cpu

CMD ["python", "training/kaggle_train.py"]

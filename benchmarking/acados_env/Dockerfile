# CUDA 12.8 + cuDNN + compiler tools for acados
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for acados
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git cmake wget \
    libgl1 libglew2.2 libosmesa6 libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# CUDA paths
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Create venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Clone and build acados (matching local version)
WORKDIR /opt
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git checkout 17625e2cd && \
    git submodule update --init --recursive

# Build acados with OpenMP enabled
WORKDIR /opt/acados
RUN mkdir build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON .. && \
    make install -j4

# Set acados environment variables
ENV ACADOS_SOURCE_DIR="/opt/acados"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib"

# Install acados Python interface
WORKDIR /opt/acados/interfaces/acados_template
RUN pip install -e .

# Clone differentiable_nmpc
RUN git clone https://github.com/FreyJo/differentiable_nmpc.git /opt/differentiable_nmpc

# Set working directory
WORKDIR /workspace

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Multi-stage build for NVIDIA GPU NIM Toolkit
# Base: NVIDIA CUDA runtime image with Ubuntu 22.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create app directory and user
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .
COPY LICENSE .
COPY README.md .
COPY QUICK_START.md .

# Install the application
RUN pip install -e .

# Create directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (if needed for web interface in future)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-toolkit system-info || exit 1

# Default command
CMD ["nvidia-toolkit", "--help"]

# Labels for metadata
LABEL maintainer="NVIDIA GPU NIM Toolkit Team"
LABEL version="1.0.0"
LABEL description="NVIDIA GPU monitoring and NIM inference toolkit with CUDA support"
LABEL org.opencontainers.image.source="https://github.com/uragupathy-trend/nvidia-gpu-nim-toolkit"
LABEL org.opencontainers.image.title="nvidia-gpu-nim-toolkit"
LABEL org.opencontainers.image.description="GPU monitoring and NIM inference toolkit"
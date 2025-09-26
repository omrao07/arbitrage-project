# base.dockerfile
# -------------------------------------------------------------------
# Base image for hedge fund platform microservices
# Includes Python, build tools, scientific stack, and minimal security hardening
# -------------------------------------------------------------------

FROM python:3.11-slim AS base

# Environment
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      git \
      curl \
      wget \
      vim \
      ca-certificates \
      libssl-dev \
      libffi-dev \
      libblas-dev \
      liblapack-dev \
      && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Install pip tools
RUN pip install --upgrade pip wheel setuptools

# Copy common requirements
COPY --chown=appuser:appuser requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Default command: run REPL (overridden in service Dockerfiles)
CMD ["python"]
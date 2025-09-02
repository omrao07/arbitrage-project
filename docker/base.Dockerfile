# docker/base.Dockerfile

# Use slim Python 3.11 as base
FROM python:3.11-slim AS base

# Avoid Python writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv for dependencies
RUN python -m venv /opt/venv

# Install pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set workdir
WORKDIR /app

# Copy pyproject + lock (if you have one)
COPY pyproject.toml ./

# Install dependencies (no dev deps by default)
RUN pip install --no-cache-dir -e . 

# Default CMD is overridden in child Dockerfiles
CMD ["python", "--version"]
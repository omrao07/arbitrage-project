# docker/worker.Dockerfile

# You can build on your base image (recommended) or fall back to python:3.11-slim.
# Override with: --build-arg BASE_IMAGE=hf-platform-base:latest
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# If BASE_IMAGE is plain python, create a venv and basic build deps.
RUN if [ ! -d /opt/venv ]; then python -m venv /opt/venv; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc libffi-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
# Install deps now to leverage layer caching
RUN pip install --no-cache-dir -e .

# Copy the rest of the source
COPY . .

# (Re)install in editable mode to pick up local modules
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app && \
    chown -R app:app /app
USER app

# Common envs (override at runtime as needed)
ENV OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4318" \
    REDIS_HOST="redis" \
    REDIS_PORT="6379"

# Choose which worker to run:
# - Build-time:  --build-arg WORKER_MODULE=workers.scenario_worker
# - Run-time:    docker run -e WORKER_MODULE=workers.sentiment_worker ...
ARG WORKER_MODULE=workers.analyst_worker
ENV WORKER_MODULE=${WORKER_MODULE}

# Expose common ports (adjust per worker if needed)
# 9090/9094/9095: Prometheus metrics; 8000: optional HTTP/health
EXPOSE 8000 9090 9094 9095

# Healthcheck: verify Redis reachability
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import os,redis; r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'), port=int(os.getenv('REDIS_PORT','6379'))); r.ping()" || exit 1

# Use shell form so $WORKER_MODULE env expands at runtime
ENTRYPOINT ["sh","-c","python -m ${WORKER_MODULE}"]
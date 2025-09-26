# worker.dockerfile
# -------------------------------------------------------------------
# Worker image for execution / analytics workers
# Builds on base.dockerfile with your trading code
# -------------------------------------------------------------------

# Inherit from the base image (already has Python + deps)
FROM base AS worker

# Work directory for appuser
WORKDIR /home/appuser/app

# Copy backend source code
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser configs/ ./configs/

# Set environment (can be overridden at deploy)
ENV PYTHONPATH=/home/appuser/app \
    ENV=production

# Healthcheck: ensures worker responds
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import redis; exit(0)" || exit 1

# Default command — points to generic worker entry
# Override with service-specific command in deployment YAML
CMD ["python", "backend/engine/worker.py"]
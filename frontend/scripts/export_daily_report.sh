#!/bin/bash
# export_daily_report.sh
# Generate and export daily PnL / risk report
# -------------------------------------------------------------------

set -euo pipefail

# --- Config (override via env) ---
DATE=$(date +"%Y-%m-%d")
OUTDIR=${REPORTS_DIR:-/var/reports}
FILENAME="daily_report_${DATE}.pdf"
LOGFILE=${LOG_FILE:-/var/log/export_daily_report.log}

# Paths to Python scripts or notebooks
PYTHON_BIN=${PYTHON_BIN:-python3}
REPORT_SCRIPT=${REPORT_SCRIPT:-backend/reports/daily_report.py}

# --- Logging helper ---
log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOGFILE"
}

# --- Ensure output dir ---
mkdir -p "$OUTDIR"

log "Starting daily report export for $DATE"

# --- Run Python report generator ---
if [ -f "$REPORT_SCRIPT" ]; then
  $PYTHON_BIN "$REPORT_SCRIPT" --date "$DATE" --out "$OUTDIR/$FILENAME"
  log "Report generated at $OUTDIR/$FILENAME"
else
  log "ERROR: Report script not found at $REPORT_SCRIPT"
  exit 1
fi

# --- Optional: upload to S3 / GCS ---
if [ "${UPLOAD_S3:-false}" = "true" ]; then
  if ! command -v aws >/dev/null 2>&1; then
    log "ERROR: aws cli not found"
    exit 1
  fi
  BUCKET=${S3_BUCKET:-my-daily-reports}
  log "Uploading report to s3://$BUCKET/$FILENAME"
  aws s3 cp "$OUTDIR/$FILENAME" "s3://$BUCKET/$FILENAME"
fi

log "Daily report export complete."

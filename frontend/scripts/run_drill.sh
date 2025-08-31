#!/bin/bash
# run_drill.sh
# Disaster recovery drill script:
# - Simulates controlled failures
# - Tests backup workers, failover, and recovery procedures

set -euo pipefail

NAMESPACE=${NAMESPACE:-default}
LOGFILE=${LOG_FILE:-/var/log/run_drill.log}
DATE=$(date +"%Y-%m-%d_%H-%M-%S")

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOGFILE"
}

log "=== Starting disaster recovery drill ($DATE) in namespace=$NAMESPACE ==="

# Step 1: Kill one analyst worker pod
ANALYST_POD=$(kubectl get pods -n "$NAMESPACE" -l app=analyst-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$ANALYST_POD" ]; then
  log "Simulating failure: deleting analyst worker pod $ANALYST_POD"
  kubectl delete pod "$ANALYST_POD" -n "$NAMESPACE" --grace-period=0 --force
else
  log "No analyst worker pod found."
fi

# Step 2: Block Redis briefly (simulate outage)
REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$REDIS_POD" ]; then
  log "Simulating Redis outage on $REDIS_POD"
  kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- sh -c "iptables -A INPUT -p tcp --dport 6379 -j DROP" || true
  sleep 15
  log "Restoring Redis connectivity"
  kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- sh -c "iptables -D INPUT -p tcp --dport 6379 -j DROP" || true
else
  log "No Redis pod found."
fi

# Step 3: Verify that scenario/sentiment workers are still healthy
for APP in scenario-worker sentiment-worker; do
  POD=$(kubectl get pods -n "$NAMESPACE" -l app=$APP -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "$POD" ]; then
    log "Checking readiness of $APP pod $POD"
    kubectl get pod "$POD" -n "$NAMESPACE" -o json | jq '.status.conditions[] | select(.type=="Ready")'
  else
    log "No $APP pod found."
  fi
done

# Step 4: Generate a test signal (simulate user request during outage)
log "Publishing test signal to STREAM_SIGNALS"
kubectl exec -n "$NAMESPACE" deploy/analyst-api -- \
  redis-cli -h redis XADD STREAM_SIGNALS * topic "risk_request" payload '{"reason":"drill_test"}' || true

log "=== Drill complete. Check logs, alerts, and failover behavior ==="

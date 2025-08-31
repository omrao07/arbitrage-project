#!/bin/bash
# run_chaos.sh
# Inject chaos into services (kill pods, add latency, drop connections)
# Useful for resilience testing of analyst/sentiment/scenario workers.

set -euo pipefail

NAMESPACE=${NAMESPACE:-default}
LOGFILE=${LOG_FILE:-/var/log/run_chaos.log}
SLEEP=${SLEEP:-60}

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" | tee -a "$LOGFILE"
}

log "Starting chaos monkey in namespace=$NAMESPACE (interval ${SLEEP}s)"

while true; do
  # Pick a random pod from namespace
  POD=$(kubectl get pods -n "$NAMESPACE" --no-headers | awk '{print $1}' | shuf -n 1)

  if [ -z "$POD" ]; then
    log "No pods found in namespace=$NAMESPACE"
    sleep "$SLEEP"
    continue
  fi

  log "Selected pod: $POD"

  ACTION=$((RANDOM % 3))

  case $ACTION in
    0)
      log "Deleting pod $POD"
      kubectl delete pod "$POD" -n "$NAMESPACE" --grace-period=0 --force
      ;;
    1)
      log "Injecting 5s network delay on $POD"
      kubectl exec -n "$NAMESPACE" "$POD" -- tc qdisc add dev eth0 root netem delay 5000ms 2>/dev/null || true
      ;;
    2)
      log "Killing random process in $POD"
      kubectl exec -n "$NAMESPACE" "$POD" -- pkill -9 -f python || true
      ;;
  esac

  sleep "$SLEEP"
done

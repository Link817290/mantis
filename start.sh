#!/bin/bash
# Mantis bot launcher - persistent background run
cd /workspace/mantis

# Load env vars
export MANTIS_PRIVATE_KEY=$(grep MANTIS_PRIVATE_KEY .env | cut -d= -f2)
export BROWSER_ADDRESS=$(grep BROWSER_ADDRESS .env | cut -d= -f2)

LOG="/workspace/mantis/data/mantis.log"
PID_FILE="/workspace/mantis/data/mantis.pid"
mkdir -p /workspace/mantis/data

case "${1:-start}" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
      echo "Mantis already running (PID $(cat $PID_FILE))"
      exit 1
    fi
    echo "$(date) Starting Mantis LIVE..." | tee -a "$LOG"
    nohup python3 -m mantis.main >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Mantis started (PID $!). Log: $LOG"
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      echo "Stopping Mantis (PID $PID)..."
      kill "$PID" 2>/dev/null
      rm -f "$PID_FILE"
      echo "$(date) Mantis stopped." | tee -a "$LOG"
    else
      echo "Mantis not running."
    fi
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
      echo "Mantis running (PID $(cat $PID_FILE))"
      echo "--- Last 10 log lines ---"
      tail -10 "$LOG"
    else
      echo "Mantis not running."
      rm -f "$PID_FILE" 2>/dev/null
    fi
    ;;
  log)
    tail -${2:-30} "$LOG"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|log [N]}"
    ;;
esac

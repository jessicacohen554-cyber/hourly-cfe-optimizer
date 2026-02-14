#!/bin/bash
# Watch for next CAISO checkpoint, then kill old optimizer and restart with new code
LOG="/home/user/hourly-cfe-optimizer/optimizer_run.log"
PID=25364

echo "[watcher] Monitoring for next checkpoint (PID $PID)..."
# Wait for a new checkpoint line to appear
while true; do
    # Count checkpoint lines
    CKPTS=$(grep -c '\[checkpoint\] Saved CAISO (threshold-' "$LOG" 2>/dev/null)
    if [ "$CKPTS" -ge 2 ]; then
        echo "[watcher] Detected checkpoint #$CKPTS — killing PID $PID"
        kill $PID 2>/dev/null
        sleep 2
        # Verify killed
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "[watcher] Old optimizer stopped. Restarting with new code..."
            cd /home/user/hourly-cfe-optimizer
            nohup python3 -u optimize_overprocure.py >> optimizer_run.log 2>&1 &
            NEW_PID=$!
            echo "[watcher] New optimizer started: PID $NEW_PID"
            echo "[watcher] New optimizer PID: $NEW_PID" >> "$LOG"
            exit 0
        else
            echo "[watcher] Failed to kill PID $PID — retrying"
            kill -9 $PID 2>/dev/null
            sleep 2
            cd /home/user/hourly-cfe-optimizer
            nohup python3 -u optimize_overprocure.py >> optimizer_run.log 2>&1 &
            NEW_PID=$!
            echo "[watcher] New optimizer started: PID $NEW_PID"
            echo "[watcher] New optimizer PID: $NEW_PID" >> "$LOG"
            exit 0
        fi
    fi
    sleep 10
done

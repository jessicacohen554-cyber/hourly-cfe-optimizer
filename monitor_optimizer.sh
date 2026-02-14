#!/bin/bash
# Monitor optimizer and notify when complete
while true; do
    if ! ps -p 5444 > /dev/null 2>&1; then
        echo "OPTIMIZER COMPLETED at $(date)"
        if [ -f /home/user/hourly-cfe-optimizer/dashboard/overprocure_results.json ]; then
            FSIZE=$(stat -c %s /home/user/hourly-cfe-optimizer/dashboard/overprocure_results.json)
            FTIME=$(stat -c %Y /home/user/hourly-cfe-optimizer/dashboard/overprocure_results.json)
            echo "Results file: $FSIZE bytes, modified $(date -d @$FTIME)"
        fi
        if [ -f /home/user/hourly-cfe-optimizer/data/optimizer_cache.json ]; then
            CSIZE=$(stat -c %s /home/user/hourly-cfe-optimizer/data/optimizer_cache.json)
            echo "Cache file: $CSIZE bytes"
        fi
        break
    fi
    sleep 30
done

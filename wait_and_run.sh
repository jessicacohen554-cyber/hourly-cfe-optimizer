#!/bin/bash
# Wait for optimizer to complete, then run post-processing pipeline
cd "$(dirname "$0")"

echo "Watching optimizer process (PID 5444)..."
while ps -p 5444 > /dev/null 2>&1; do
    sleep 30
done

echo ""
echo "OPTIMIZER COMPLETED at $(date)"
echo ""

# Check if results were written
RESULTS="dashboard/overprocure_results.json"
FSIZE=$(stat -c %s "$RESULTS" 2>/dev/null || echo "0")
FTIME=$(stat -c %Y "$RESULTS" 2>/dev/null || echo "0")
echo "Results: $FSIZE bytes, modified $(date -d @$FTIME 2>/dev/null || echo 'unknown')"

# Check cache
CACHE="data/optimizer_cache.json"
if [ -f "$CACHE" ]; then
    CSIZE=$(stat -c %s "$CACHE")
    echo "Cache: $CSIZE bytes"
fi

# Run post-processing
echo ""
echo "Starting post-processing pipeline..."
bash run_post_optimizer.sh 2>&1

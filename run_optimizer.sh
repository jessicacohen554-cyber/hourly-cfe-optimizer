#!/bin/bash
# Auto-restart optimizer wrapper with periodic checkpoint commits
# - Restarts from last checkpoint if process dies
# - Commits checkpoint files every 5 minutes
# - Runs until optimizer completes successfully

REPO_DIR="/home/user/hourly-cfe-optimizer"
CHECKPOINT_DIR="$REPO_DIR/data/checkpoints"
RESULTS_FILE="$REPO_DIR/dashboard/overprocure_results.json"
CACHE_FILE="$REPO_DIR/data/optimizer_cache.json"
BRANCH="claude/restart-optimizer-checkpoint-RdHBJ"
MAX_RETRIES=20
COMMIT_INTERVAL=300  # 5 minutes

commit_checkpoints() {
    cd "$REPO_DIR"
    if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
        git add data/checkpoints/ dashboard/overprocure_results.json data/optimizer_cache.json 2>/dev/null
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "$(cat <<'COMMITEOF'
Optimizer checkpoint auto-save: progress preserved

Periodic commit of optimizer checkpoint files and any incremental results
to prevent data loss on interruption. Optimizer resumes from checkpoint.

https://claude.ai/code/session_01QPh3svMBi29na4GHiijmzv
COMMITEOF
)" 2>/dev/null
            git push -u origin "$BRANCH" 2>/dev/null
            echo "[commit] Checkpoint files committed and pushed at $(date)"
        fi
    fi
}

# Background committer
checkpoint_committer() {
    while true; do
        sleep $COMMIT_INTERVAL
        commit_checkpoints
    done
}

# Start background committer
checkpoint_committer &
COMMITTER_PID=$!
trap "kill $COMMITTER_PID 2>/dev/null; commit_checkpoints; exit" EXIT INT TERM

attempt=0
while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "=========================================="
    echo "  Optimizer run attempt $attempt/$MAX_RETRIES"
    echo "  Started: $(date)"
    echo "=========================================="

    cd "$REPO_DIR"
    PYTHONUNBUFFERED=1 python3 optimize_overprocure.py
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "  Optimizer completed successfully!"
        echo "  Finished: $(date)"
        echo "=========================================="
        # Final commit
        commit_checkpoints
        break
    else
        echo ""
        echo "=========================================="
        echo "  Optimizer exited with code $EXIT_CODE"
        echo "  Committing checkpoints before retry..."
        echo "=========================================="
        commit_checkpoints
        echo "  Waiting 5s before restart..."
        sleep 5
    fi
done

if [ $attempt -ge $MAX_RETRIES ]; then
    echo "Max retries ($MAX_RETRIES) reached. Exiting."
fi

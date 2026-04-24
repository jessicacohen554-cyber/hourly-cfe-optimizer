#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
[ "${CLAUDE_CODE_REMOTE:-false}" = "true" ] || exit 0

# Install Python dependencies only if any are missing — saves 15–30s on warm sessions
python -c "import numpy, pyarrow, numba" 2>/dev/null || pip install --quiet numpy pyarrow numba

# Set PYTHONPATH to project root for imports
echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR\"" >> "$CLAUDE_ENV_FILE"

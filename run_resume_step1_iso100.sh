#!/usr/bin/env bash
set -euo pipefail

# Resume Step 1 PFS generation from threshold checkpoints for 100% CFE targets.
# Usage:
#   ./run_resume_step1_iso100.sh

python - <<'PY'
import importlib.util, sys
missing = [m for m in ('numba', 'pyarrow') if importlib.util.find_spec(m) is None]
if missing:
    print('Missing dependencies:', ', '.join(missing))
    print('Install with: python -m pip install numba pyarrow')
    sys.exit(1)
print('Dependencies OK: numba, pyarrow')
PY

echo "Resuming NYISO 100% threshold from checkpoint..."
python "scripts/step 1 PFS - individual ISO threshold selector.py" NYISO 100

echo "Resuming CAISO 100% threshold from checkpoint..."
python "scripts/step 1 PFS - individual ISO threshold selector.py" CAISO 100

echo "Done. NYISO and CAISO 100% threshold completion attempted."

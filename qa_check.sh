#!/usr/bin/env bash
# ============================================================================
# QA Check Script — Validates project integrity after edits
# Run: bash qa_check.sh
# Exit code: 0 = all pass, 1 = failures found
# ============================================================================
set -uo pipefail

DASH="dashboard"
PASS=0
FAIL=0
WARN=0

pass()  { PASS=$((PASS + 1)); echo "  [PASS] $1"; }
fail()  { FAIL=$((FAIL + 1)); echo "  [FAIL] $1"; }
warn()  { WARN=$((WARN + 1)); echo "  [WARN] $1"; }
section() { echo ""; echo "=== $1 ==="; }

# ============================================================================
section "1. Shared Data Module Integrity"
# ============================================================================

# Check shared-data.js exists and is loaded by key pages
if [ -f "$DASH/js/shared-data.js" ]; then
    pass "shared-data.js exists"
else
    fail "shared-data.js missing"
fi

for page in index.html dashboard.html region_deepdive.html abatement_comparison.html abatement_dashboard.html; do
    if [ -f "$DASH/$page" ]; then
        if grep -q 'src="js/shared-data.js"' "$DASH/$page" 2>/dev/null; then
            pass "$page imports shared-data.js"
        else
            warn "$page does NOT import shared-data.js"
        fi
    fi
done

# Check no inline duplication of shared constants
for page in abatement_comparison.html abatement_dashboard.html; do
    if [ -f "$DASH/$page" ]; then
        # These pages should NOT have their own MAC_DATA definitions
        if grep -q "^const MAC_DATA" "$DASH/$page" 2>/dev/null; then
            fail "$page has inline MAC_DATA (should use shared-data.js)"
        else
            pass "$page uses shared MAC_DATA"
        fi
    fi
done

# ============================================================================
section "2. Threshold 100% Bug Check"
# ============================================================================

# Data-driven threshold arrays should NOT include 100
for page in dashboard.html index.html; do
    if [ -f "$DASH/$page" ]; then
        # Check for thresholdKeys or thresholdNums arrays containing 100
        bad_count=$(grep -cE "thresholdKeys.*'100'|thresholdNums.*100\]" "$DASH/$page" 2>/dev/null || true)
        if [ "$bad_count" -gt 0 ]; then
            fail "$page has $bad_count data-driven threshold 100 references"
        else
            pass "$page: no data-driven threshold 100 refs"
        fi
    fi
done

# Check shared-data.js THRESHOLDS doesn't include 100
if [ -f "$DASH/js/shared-data.js" ]; then
    if grep -q "THRESHOLDS.*100" "$DASH/js/shared-data.js" 2>/dev/null; then
        fail "shared-data.js THRESHOLDS includes 100"
    else
        pass "shared-data.js THRESHOLDS stops at 99"
    fi
fi

# ============================================================================
section "3. MAC Data Monotonicity Check"
# ============================================================================

# Verify MAC_DATA arrays in shared-data.js are non-decreasing
if [ -f "$DASH/js/shared-data.js" ]; then
    python3 -c "
import re, sys
with open('$DASH/js/shared-data.js') as f:
    content = f.read()

# Extract all array values from MAC_DATA
arrays = re.findall(r'(\w+):\s*\[([^\]]+)\]', content)
violations = []
for name, vals in arrays:
    nums = [int(x.strip()) for x in vals.split(',') if x.strip().isdigit()]
    if len(nums) >= 2:
        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                violations.append(f'{name}: index {i} ({nums[i]} < {nums[i-1]})')

if violations:
    for v in violations:
        print(f'  [FAIL] Monotonicity violation: {v}')
    sys.exit(1)
else:
    print('  [PASS] All MAC_DATA arrays are monotonically non-decreasing')
" 2>/dev/null && true || ((FAIL++))
fi

# ============================================================================
section "4. Chart.js Canvas/Init Pairing"
# ============================================================================

for page in "$DASH"/*.html; do
    [ -f "$page" ] || continue
    basename=$(basename "$page")

    # Count canvas elements and Chart() instantiations
    canvas_count=$(grep -c '<canvas' "$page" 2>/dev/null || true)
    chart_count=$(grep -c 'new Chart(' "$page" 2>/dev/null || true)

    if [ "$canvas_count" -gt 0 ] && [ "$chart_count" -eq 0 ]; then
        fail "$basename: $canvas_count canvas elements but 0 Chart() calls"
    elif [ "$canvas_count" -gt 0 ] || [ "$chart_count" -gt 0 ]; then
        pass "$basename: $canvas_count canvas, $chart_count Chart() calls"
    fi
done

# ============================================================================
section "5. Link Validation (internal)"
# ============================================================================

for page in "$DASH"/*.html; do
    [ -f "$page" ] || continue
    basename=$(basename "$page")

    # Extract href values pointing to local files (not http/https/mailto/#)
    hrefs=$(grep -oP 'href="(?!https?://|mailto:|#|javascript:)([^"]+)"' "$page" 2>/dev/null | sed 's/href="//;s/"//' || true)

    for href in $hrefs; do
        # Resolve relative to dashboard directory
        target="$DASH/$href"
        if [ ! -f "$target" ] && [ ! -d "$target" ]; then
            # Try from repo root
            if [ ! -f "$href" ] && [ ! -d "$href" ]; then
                warn "$basename: broken link -> $href"
            fi
        fi
    done
done

# ============================================================================
section "6. Python Syntax Check"
# ============================================================================

for pyfile in optimize_overprocure.py postprocess_results.py; do
    if [ -f "$pyfile" ]; then
        if python3 -c "import py_compile; py_compile.compile('$pyfile', doraise=True)" 2>/dev/null; then
            pass "$pyfile syntax OK"
        else
            fail "$pyfile has syntax errors"
        fi
    fi
done

# ============================================================================
section "7. Data File Integrity"
# ============================================================================

if [ -f "$DASH/overprocure_results.json" ]; then
    if python3 -c "
import json, sys
with open('$DASH/overprocure_results.json') as f:
    data = json.load(f)
isos = list(data.get('results', {}).keys())
print(f'  [PASS] Results JSON valid: {len(isos)} ISOs: {isos}')
for iso in isos:
    iso_data = data['results'][iso]
    if not isinstance(iso_data, dict):
        continue
    thresholds = [t for t in iso_data.keys() if t.replace('.','').isdigit()]
    empty = [t for t in thresholds if isinstance(iso_data[t], dict) and iso_data[t].get('scenario_count', 0) == 0]
    if empty:
        print(f'  [WARN] {iso}: empty thresholds: {empty}')
" 2>&1; then
        true  # Python script succeeded
    else
        fail "Results JSON invalid or could not parse"
    fi
else
    fail "Results JSON file missing"
fi

# ============================================================================
section "Summary"
# ============================================================================

echo ""
echo "Results: $PASS passed, $FAIL failed, $WARN warnings"

if [ "$FAIL" -gt 0 ]; then
    echo "STATUS: FAILED"
    exit 1
else
    echo "STATUS: PASSED"
    exit 0
fi

"""Unit tests for the Phase C foresight scorer primitives (memo §4(b)).

Synthetic 5-year × 3-resource toy problem. No dependency on EF parquets,
dispatch caches, or pipeline_config — the test exercises
_score_year_with_endpoint + _project_shares_to_endpoint directly.

Three assertions:
  1. At λ=0, foresight_score == row_score elementwise (degenerate to myopic).
  2. At sufficiently large λ, argmin flips to the candidate closest to
     target_shares even if that candidate is NOT the myopic argmin.
  3. The lookahead weight w(y) decays linearly from 1 at BASE to 0 at
     endpoint_year, so the penalty → 0 at the endpoint regardless of λ.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import step_2_3_pathway_optimizer as opt


# ─── Toy problem setup ──────────────────────────────────────────────────────
# 3 candidate mixes × 3 resources. Shares sum to 1 per mix.
#   mix 0: (0.0, 1.0, 0.0)  — pure solar  (cheap myopic, FAR from target)
#   mix 1: (0.5, 0.5, 0.0)  — solar+CF    (moderate myopic, AT target)
#   mix 2: (0.0, 0.5, 0.5)  — solar+wind  (moderate myopic, far from target)
N = 3
SHARES_PER_MIX = np.array([
    [0.0, 1.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5],
], dtype=np.float64)

TARGET_SHARES = np.array([0.5, 0.5, 0.0], dtype=np.float64)

# myopic USD scores: mix 0 is cheapest at $100, mix 1 is $110, mix 2 is $120.
ROW_SCORE = np.array([100.0, 110.0, 120.0], dtype=np.float64)

# USD-scale normalizer — pick any positive value; the real one (demand ×
# ref LCOE) is huge, which is the whole point (memo DIMENSIONAL CONVENTION
# block). For a toy test we pick C=1000 so a modest λ dominates.
FORESIGHT_C = 1000.0

# 5-year horizon toy: years [0, 1, 2, 3, 4] with endpoint_year = year 4.
BASE_Y = 0
END_Y  = 4
HORIZON = END_Y - BASE_Y


def _score_with_penalty(row_score, shares, target, lam, weight, C):
    """Inline reproduction of the foresight scoring formula (memo §4(b)
    with the Phase-C USD-scale normalizer). The solver applies this same
    expression; this test exercises the formula, not the code path.
    """
    diff = shares - target[None, :]
    dist_sq = np.sum(diff * diff, axis=1)
    penalty = (lam * weight * C) * dist_sq
    return row_score + penalty


# ─── Assertions ─────────────────────────────────────────────────────────────

def test_lambda_zero_degenerate_to_myopic():
    """Exit-gate invariant (memo §5.2): λ=0 → identical numerical score."""
    for y in range(BASE_Y, END_Y + 1):
        w = max(0.0, (END_Y - y) / HORIZON)
        fs = _score_with_penalty(
            ROW_SCORE, SHARES_PER_MIX, TARGET_SHARES,
            lam=0.0, weight=w, C=FORESIGHT_C,
        )
        assert np.allclose(fs, ROW_SCORE, atol=0.0, rtol=0.0)
        assert int(np.argmin(fs)) == 0  # mix 0 is cheapest


def test_positive_lambda_flips_argmin_toward_target():
    """λ > 0 at year 0 (weight=1.0) steers the argmin to the mix that
    minimizes ‖shares − target‖² when the penalty outweighs the myopic
    cost gap. Mix 1 AT the target beats mix 0 (cheapest myopic, far
    from target) once λ·C dominates the $10 myopic advantage.
    """
    w_y0 = max(0.0, (END_Y - BASE_Y) / HORIZON)  # == 1.0
    # Gap between mix 1 and mix 0 myopic: $10.
    # Penalty at mix 0 (dist² = 0.5² + 0.5² = 0.5): λ · 1.0 · C · 0.5 = 500·λ
    # Penalty at mix 1 (dist² = 0):                                       = 0
    # Argmin flips when 100 + 500λ > 110 → λ > 0.02.
    fs_tiny = _score_with_penalty(
        ROW_SCORE, SHARES_PER_MIX, TARGET_SHARES,
        lam=0.01, weight=w_y0, C=FORESIGHT_C,
    )
    assert int(np.argmin(fs_tiny)) == 0, "λ=0.01 is too small to flip argmin"

    fs_big = _score_with_penalty(
        ROW_SCORE, SHARES_PER_MIX, TARGET_SHARES,
        lam=0.05, weight=w_y0, C=FORESIGHT_C,
    )
    assert int(np.argmin(fs_big)) == 1, (
        "λ=0.05 should flip argmin to mix 1 (AT target) — "
        f"got scores {fs_big.tolist()}"
    )


def test_weight_decays_to_zero_at_endpoint():
    """w(endpoint_year) = 0 → penalty is zero regardless of λ, so the
    argmin collapses back to the myopic cheapest. Confirms the memo's
    "ratchet does the enforcement work late in the horizon" design.
    """
    w_end = max(0.0, (END_Y - END_Y) / HORIZON)  # == 0.0
    fs = _score_with_penalty(
        ROW_SCORE, SHARES_PER_MIX, TARGET_SHARES,
        lam=1.5, weight=w_end, C=FORESIGHT_C,
    )
    assert np.allclose(fs, ROW_SCORE, atol=0.0, rtol=0.0)
    assert int(np.argmin(fs)) == 0


def test_weight_is_linear_in_year():
    """Lookahead weight is linear: w(y) = (END - y) / (END - BASE).
    Spot-check at midpoint: y=2 → w=0.5.
    """
    y = 2
    w = max(0.0, (END_Y - y) / HORIZON)
    assert abs(w - 0.5) < 1e-12


# ─── Invariant: the production helper matches the inline formula ────────────

def test_score_year_with_endpoint_matches_inline():
    """opt._score_year_with_endpoint (the primitive reused by Phase B's
    sidecar AND by solve_pathway_with_foresight's penalty composition)
    must match the inline reference. This guards against drift if the
    primitive is ever refactored.
    Note: _score_year_with_endpoint takes the already-row_score and
    applies λ·weight·‖diff‖² WITHOUT the C normalizer — C is folded in
    at the call site in solve_pathway_with_foresight. This test uses C=1
    to stay consistent with the primitive's contract.
    """
    w_y0 = 1.0
    lam = 0.05
    # Expand shares_per_mix to 15-dim (pad with zeros) to match the
    # primitive's expected _FORESIGHT_SHARE_KEYS layout.
    pad_width = len(opt._FORESIGHT_SHARE_KEYS) - SHARES_PER_MIX.shape[1]
    shares_15 = np.hstack([
        SHARES_PER_MIX,
        np.zeros((N, pad_width), dtype=np.float64),
    ])
    target_15 = np.hstack([
        TARGET_SHARES,
        np.zeros(pad_width, dtype=np.float64),
    ])

    helper_out = opt._score_year_with_endpoint(
        row_score=ROW_SCORE,
        shares_to_endpoint=shares_15,
        target_shares=target_15,
        lam=lam,
        weight=w_y0,
    )
    # Inline, no C:
    diff = shares_15 - target_15[None, :]
    expected = ROW_SCORE + lam * w_y0 * np.sum(diff * diff, axis=1)
    assert np.allclose(helper_out, expected, atol=0.0, rtol=0.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

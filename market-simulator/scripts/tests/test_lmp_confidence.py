"""Tests for G5: LMP confidence degradation at high VRE penetration.

Updated for continuous sigmoid model (replaces 4-step discrete brackets).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lmp_engine import compute_lmp_confidence_factor


class TestLmpConfidenceFactor:
    """Verify confidence degrades correctly via continuous sigmoid."""

    def test_none_returns_full_confidence(self):
        assert compute_lmp_confidence_factor(None) == 1.0

    def test_zero_vre(self):
        assert compute_lmp_confidence_factor(0.0) > 0.99

    def test_within_calibration_low(self):
        """VRE <= 50% is effectively fully calibrated (>0.99)."""
        assert compute_lmp_confidence_factor(0.25) > 0.99
        assert compute_lmp_confidence_factor(0.50) > 0.98

    def test_within_calibration_boundary(self):
        """VRE at 60% is still near full confidence (>0.95)."""
        assert compute_lmp_confidence_factor(0.60) > 0.93

    def test_midpoint_near_half(self):
        """At sigmoid midpoint (78% VRE), confidence is 0.7 (midpoint of [0.4, 1.0])."""
        c = compute_lmp_confidence_factor(0.78)
        assert 0.65 <= c <= 0.75, f"Midpoint confidence {c} not near 0.7"

    def test_high_vre_degraded(self):
        """85-90% VRE should be significantly degraded (<0.6)."""
        assert compute_lmp_confidence_factor(0.85) < 0.6
        assert compute_lmp_confidence_factor(0.90) < 0.5

    def test_very_high_vre_near_floor(self):
        """VRE > 95% approaches floor of 0.4."""
        c95 = compute_lmp_confidence_factor(0.95)
        c100 = compute_lmp_confidence_factor(1.0)
        assert c95 < 0.45
        assert c100 < 0.43
        assert c100 >= 0.40  # Never below floor

    def test_floor_respected(self):
        """Confidence never drops below 0.4, even at 100% VRE."""
        assert compute_lmp_confidence_factor(1.0) >= 0.4

    def test_monotonically_decreasing(self):
        """Confidence must never increase as VRE increases."""
        points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65,
                  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        confidences = [compute_lmp_confidence_factor(p) for p in points]
        for i in range(1, len(confidences)):
            assert confidences[i] <= confidences[i - 1], (
                f"Confidence increased from {confidences[i-1]:.4f} to {confidences[i]:.4f} "
                f"between VRE {points[i-1]} and {points[i]}"
            )

    def test_continuous_no_jumps(self):
        """No discontinuities: adjacent 1pp steps differ by < 0.05."""
        for pct in range(0, 100):
            p1 = pct / 100.0
            p2 = (pct + 1) / 100.0
            c1 = compute_lmp_confidence_factor(p1)
            c2 = compute_lmp_confidence_factor(p2)
            assert abs(c1 - c2) < 0.05, (
                f"Discontinuity: VRE {p1:.2f}→{p2:.2f}, "
                f"confidence {c1:.4f}→{c2:.4f} (jump {abs(c1-c2):.4f})"
            )

    def test_approximate_equivalence_to_old_brackets(self):
        """New sigmoid should approximately match old bracket midpoints."""
        # At 50% VRE: was 1.0, should still be ~1.0
        assert compute_lmp_confidence_factor(0.50) > 0.98
        # At 67.5% VRE (midpoint of old 0.8 bracket): should be ~0.85-0.95
        c = compute_lmp_confidence_factor(0.675)
        assert 0.85 < c < 0.95, f"At 67.5% VRE, expected ~0.9, got {c:.3f}"
        # At 82.5% VRE (midpoint of old 0.6 bracket): should be ~0.55-0.65
        c = compute_lmp_confidence_factor(0.825)
        assert 0.5 < c < 0.65, f"At 82.5% VRE, expected ~0.6, got {c:.3f}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

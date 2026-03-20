"""Tests for G5: LMP confidence degradation at high VRE penetration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lmp_engine import compute_lmp_confidence_factor


class TestLmpConfidenceFactor:
    """Verify confidence degrades correctly at each VRE bracket."""

    def test_none_returns_full_confidence(self):
        assert compute_lmp_confidence_factor(None) == 1.0

    def test_zero_vre(self):
        assert compute_lmp_confidence_factor(0.0) == 1.0

    def test_within_calibration_low(self):
        """VRE <= 50% is fully calibrated."""
        assert compute_lmp_confidence_factor(0.25) == 1.0
        assert compute_lmp_confidence_factor(0.50) == 1.0

    def test_within_calibration_boundary(self):
        """VRE at exactly 60% is still fully calibrated."""
        assert compute_lmp_confidence_factor(0.60) == 1.0

    def test_moderate_extrapolation(self):
        """60% < VRE <= 75% -> confidence 0.8."""
        assert compute_lmp_confidence_factor(0.61) == 0.8
        assert compute_lmp_confidence_factor(0.70) == 0.8
        assert compute_lmp_confidence_factor(0.75) == 0.8

    def test_significant_extrapolation(self):
        """75% < VRE <= 90% -> confidence 0.6."""
        assert compute_lmp_confidence_factor(0.76) == 0.6
        assert compute_lmp_confidence_factor(0.85) == 0.6
        assert compute_lmp_confidence_factor(0.90) == 0.6

    def test_beyond_model_validity(self):
        """VRE > 90% -> confidence 0.4."""
        assert compute_lmp_confidence_factor(0.91) == 0.4
        assert compute_lmp_confidence_factor(0.95) == 0.4
        assert compute_lmp_confidence_factor(1.0) == 0.4

    def test_monotonically_decreasing(self):
        """Confidence must never increase as VRE increases."""
        points = [0.0, 0.3, 0.5, 0.6, 0.65, 0.75, 0.8, 0.9, 0.95, 1.0]
        confidences = [compute_lmp_confidence_factor(p) for p in points]
        for i in range(1, len(confidences)):
            assert confidences[i] <= confidences[i - 1], (
                f"Confidence increased from {confidences[i-1]} to {confidences[i]} "
                f"between VRE {points[i-1]} and {points[i]}"
            )

    def test_bracket_boundaries_exact(self):
        """Verify the exact bracket boundaries per spec."""
        # Boundary values — at the boundary, use the lower (better) confidence
        assert compute_lmp_confidence_factor(0.50) == 1.0
        assert compute_lmp_confidence_factor(0.60) == 1.0
        assert compute_lmp_confidence_factor(0.75) == 0.8
        assert compute_lmp_confidence_factor(0.90) == 0.6
        # Just above boundary — use the higher (worse) bracket
        assert compute_lmp_confidence_factor(0.600001) == 0.8
        assert compute_lmp_confidence_factor(0.750001) == 0.6
        assert compute_lmp_confidence_factor(0.900001) == 0.4


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

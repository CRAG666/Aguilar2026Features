"""Tests for the ``mwf.security`` template key-sensitivity metric."""

from __future__ import annotations

import numpy as np
import pytest

from mwf.feature_transform import transform_multimodal
from mwf.security import key_sensitivity


@pytest.fixture
def feature_vector() -> np.ndarray:
    # Even length so the multimodal split (ECG ‖ PPG) is balanced.
    rng = np.random.default_rng(0)
    return rng.normal(size=512)


def test_key_sensitivity_avalanche_around_50_percent(feature_vector):
    """A 1-bit token edit should flip ~half the quantised template bits and
    leave the templates essentially uncorrelated (ISO/IEC 30136 diversity)."""
    report = key_sensitivity(
        transform_fn=lambda tok: transform_multimodal(feature_vector, tok),
        base_password="KEY_SENS_BASE",
        n_trials=8,
    )
    assert 0.4 < report.bit_error_rate_mean < 0.6
    # correlation_mean is mean |r|, so it is non-negative by construction.
    assert 0.0 <= report.correlation_mean < 0.2

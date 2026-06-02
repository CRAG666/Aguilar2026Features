"""Validate the multimodal (ECG+PPG) wavelet feature extractor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.constants import FEATURE_WAVELET
from mwf.features import (
    N_MODALITIES,
    STATS_PER_BAND,
    extract_features,
    extract_features_batch,
    feature_dimension,
    feature_names,
    max_feature_level,
)

SEGMENT_LENGTH = 750  # MIMIC-100, 6 s × 125 Hz
RNG = np.random.default_rng(seed=20260520)


def _random_pair(n: int = SEGMENT_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    return RNG.normal(size=n), RNG.normal(size=n)


def test_feature_dimension_matches_extracted_vector():
    ecg, ppg = _random_pair()
    for level in (3, 4, 5):
        feats = extract_features(ecg, ppg, level=level)
        assert feats.shape == (feature_dimension(level),)
        assert feats.shape[0] == N_MODALITIES * STATS_PER_BAND * (level + 1)


def test_features_are_finite_for_random_input():
    ecg, ppg = _random_pair()
    feats = extract_features(ecg, ppg, level=4)
    assert np.all(np.isfinite(feats))


def test_extract_features_batch_matches_single_extract():
    ecg = np.vstack([_random_pair()[0] for _ in range(5)])
    ppg = np.vstack([_random_pair()[1] for _ in range(5)])
    out = extract_features_batch(ecg, ppg, level=4)
    for row in range(5):
        np.testing.assert_allclose(
            out[row], extract_features(ecg[row], ppg[row], level=4), atol=1e-12
        )


def test_zero_segments_yield_finite_features():
    """All-zero input is sanitised to finite values (no NaN from kurtosis/skew)."""
    zeros = np.zeros(SEGMENT_LENGTH, dtype=np.float64)
    feats = extract_features(zeros, zeros, level=4)
    assert np.all(np.isfinite(feats))


def test_extract_features_accepts_readonly_input():
    """Regression: joblib memory-maps large chunks read-only into workers, and
    ``pywt.wavedec`` needs a writable buffer — the extractor must copy first."""
    ecg, ppg = _random_pair()
    ecg.flags.writeable = False
    ppg.flags.writeable = False
    feats = extract_features(ecg, ppg, level=4)
    assert feats.shape == (feature_dimension(4),)
    assert np.all(np.isfinite(feats))


def test_extract_features_rejects_length_mismatch():
    with pytest.raises(ValueError):
        extract_features(np.zeros(100), np.zeros(101))


def test_max_feature_level_is_extractable_and_bounds_the_sweep():
    """Every level in 1..max_feature_level extracts; max + 1 must fail."""
    max_level = max_feature_level(SEGMENT_LENGTH, FEATURE_WAVELET)
    assert max_level >= 1
    ecg, ppg = _random_pair()
    for level in range(1, max_level + 1):
        feats = extract_features(ecg, ppg, wavelet=FEATURE_WAVELET, level=level)
        assert feats.shape == (feature_dimension(level),)
    with pytest.raises(ValueError):
        extract_features(ecg, ppg, wavelet=FEATURE_WAVELET, level=max_level + 1)


def test_max_feature_level_matches_bior33_on_mimic_segments():
    # bior3.3 (dec_len 8) on 750-sample MIMIC-100 segments supports 6 levels.
    assert max_feature_level(750, "bior3.3") == 6


def test_max_feature_level_rejects_empty_segment():
    with pytest.raises(ValueError):
        max_feature_level(0)


def test_feature_names_split_ecg_then_ppg():
    names = feature_names(SEGMENT_LENGTH, level=4)
    assert len(names) == feature_dimension(4)
    half = len(names) // 2
    assert all(n.startswith("ECG_") for n in names[:half])
    assert all(n.startswith("PPG_") for n in names[half:])

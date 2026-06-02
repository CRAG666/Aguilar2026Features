"""Validate the NeuroKit ECG/PPG cleaning front-end (mwf.clean)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.clean import (
    DEFAULT_ECG_METHOD,
    DEFAULT_PPG_METHOD,
    clean_ecg,
    clean_ecg_batch,
    clean_ppg,
    clean_ppg_batch,
)

SAMPLING_RATE = 125  # MIMIC-100
SEGMENT_LENGTH = 750  # 6 s × 125 Hz
RNG = np.random.default_rng(seed=20260527)


def _synthetic(freq_hz: float, n: int = SEGMENT_LENGTH) -> np.ndarray:
    """A noisy sinusoid with baseline wander, standing in for a pulse wave."""
    t = np.arange(n) / SAMPLING_RATE
    baseline = 0.5 * np.sin(2 * np.pi * 0.05 * t)  # <0.5 Hz drift the cleaners remove
    return np.sin(2 * np.pi * freq_hz * t) + baseline + 0.1 * RNG.standard_normal(n)


def test_defaults_are_neurokit_and_elgendi():
    assert DEFAULT_ECG_METHOD == "neurokit"
    assert DEFAULT_PPG_METHOD == "elgendi"


def test_clean_preserves_length_and_finiteness():
    ecg = _synthetic(1.2)
    ppg = _synthetic(1.1)
    out_ecg = clean_ecg(ecg, sampling_rate=SAMPLING_RATE)
    out_ppg = clean_ppg(ppg, sampling_rate=SAMPLING_RATE)
    assert out_ecg.shape == ecg.shape
    assert out_ppg.shape == ppg.shape
    assert out_ecg.dtype == np.float64 and out_ppg.dtype == np.float64
    assert np.isfinite(out_ecg).all() and np.isfinite(out_ppg).all()


def test_cleaning_removes_baseline_wander():
    # The 0.05 Hz drift sits below both passbands, so the cleaned mean drifts
    # far less than the raw one.
    ecg = _synthetic(1.2)
    cleaned = clean_ecg(ecg, sampling_rate=SAMPLING_RATE)
    assert abs(cleaned.mean()) < abs(ecg.mean()) + 1e-9
    assert np.std(cleaned) > 0


def test_batch_matches_per_row_and_keeps_order():
    batch = np.stack([_synthetic(1.0 + 0.1 * i) for i in range(6)])
    out = clean_ppg_batch(batch, sampling_rate=SAMPLING_RATE, n_jobs=1)
    assert out.shape == batch.shape
    for i in range(batch.shape[0]):
        expected = clean_ppg(batch[i], sampling_rate=SAMPLING_RATE)
        np.testing.assert_allclose(out[i], expected, rtol=1e-9, atol=1e-9)


def test_ecg_batch_shape():
    batch = np.stack([_synthetic(1.2) for _ in range(4)])
    out = clean_ecg_batch(batch, sampling_rate=SAMPLING_RATE, n_jobs=1)
    assert out.shape == batch.shape
    assert np.isfinite(out).all()


def test_rejects_non_1d_signal():
    with pytest.raises(ValueError):
        clean_ecg(np.zeros((2, SEGMENT_LENGTH)), sampling_rate=SAMPLING_RATE)


def test_rejects_non_positive_sampling_rate():
    with pytest.raises(ValueError):
        clean_ppg(_synthetic(1.1), sampling_rate=0)

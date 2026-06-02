"""Tests for the recognition-leakage ratio sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.dataset import BiometricSegments
from mwf.ratio_sweep import RatioSweepPoint, ratio_sweep

SEGMENT_LENGTH = 256


def _synthetic_cohort(
    n_subjects: int = 6, segs_per_subject: int = 5, jitter: float = 0.05,
) -> BiometricSegments:
    rng = np.random.default_rng(3)
    t = np.linspace(0, 4 * np.pi, SEGMENT_LENGTH)
    ecg_rows, ppg_rows, labels = [], [], []
    for subject in range(1, n_subjects + 1):
        phase = rng.uniform(0, 2 * np.pi)
        freq = 1.0 + 0.3 * subject
        ecg_base = np.sin(freq * t + phase)
        ppg_base = np.cos(0.5 * freq * t + phase)
        for _ in range(segs_per_subject):
            ecg_rows.append(ecg_base + jitter * rng.normal(size=SEGMENT_LENGTH))
            ppg_rows.append(ppg_base + jitter * rng.normal(size=SEGMENT_LENGTH))
            labels.append(subject)
    return BiometricSegments(
        ecg=np.asarray(ecg_rows),
        ppg=np.asarray(ppg_rows),
        labels=np.asarray(labels, dtype=np.int64),
        sampling_rate=125,
    )


def test_returns_one_point_per_ratio():
    ratios = (0.25, 0.5, 0.75)
    points = ratio_sweep(
        _synthetic_cohort(), feature_level=3, denoise=False, ratios=ratios,
        n_inversion_segments=4, max_victims=3,
    )
    assert len(points) == len(ratios)
    assert [p.ratio for p in points] == list(ratios)
    assert all(isinstance(p, RatioSweepPoint) for p in points)


def test_metrics_lie_in_legal_ranges():
    points = ratio_sweep(
        _synthetic_cohort(), feature_level=3, denoise=False,
        ratios=(0.25, 0.5),
        n_inversion_segments=4, max_victims=3,
    )
    for p in points:
        for eer in (p.single_key_eer, p.stolen_token_eer):
            assert np.isnan(eer) or 0.0 <= eer <= 1.0
        assert np.isnan(p.inversion_mean_correlation) or 0.0 <= p.inversion_mean_correlation <= 1.0
        assert p.n_inversion_segments >= 0


def test_inversion_correlation_grows_with_ratio():
    """Larger m/d ratio = larger row space = more recoverable feature energy."""
    points = ratio_sweep(
        _synthetic_cohort(), feature_level=3, denoise=False,
        ratios=(0.25, 0.75),
        n_inversion_segments=6, max_victims=3,
    )
    low, high = points[0], points[1]
    assert low.inversion_mean_correlation <= high.inversion_mean_correlation + 0.05


def test_results_are_deterministic_under_fixed_seed():
    cohort = _synthetic_cohort()
    p1 = ratio_sweep(
        cohort, feature_level=3, denoise=False, ratios=(0.5,),
        n_inversion_segments=4, max_victims=3, seed=11,
    )
    p2 = ratio_sweep(
        cohort, feature_level=3, denoise=False, ratios=(0.5,),
        n_inversion_segments=4, max_victims=3, seed=11,
    )
    assert p1 == p2

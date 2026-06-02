"""Tests for the cancelability protocol (renewability baseline, ratios)."""

from __future__ import annotations

import numpy as np
import pytest

from mwf.cancelability import (
    _d_sys_curve,
    _same_key_genuine_mean,
    _standardize_columns,
    evaluate_cancelability,
)
from mwf.dataset import BiometricSegments


def test_same_key_baseline_excludes_self_matches():
    # Subject 1 segments are orthogonal (cosine 0); subject 2 segments are
    # identical (cosine 1). Off-diagonal same-subject mean = (0+0+1+1)/4 = 0.5.
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    labels = np.array([1, 1, 2, 2], dtype=np.int64)
    assert _same_key_genuine_mean(features, labels) == pytest.approx(0.5)


def test_same_key_baseline_requires_repeat_segments():
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1, 2], dtype=np.int64)  # one segment each → no genuine pair
    with pytest.raises(ValueError):
        _same_key_genuine_mean(features, labels)


def test_dsys_near_zero_for_unlinkable_overlapping_distributions():
    rng = np.random.default_rng(3)
    mated = rng.normal(0.0, 0.12, 20000)
    non_mated = rng.normal(0.0, 0.12, 20000)
    curve = _d_sys_curve(mated, non_mated)
    assert curve.d_sys < 0.1
    assert 0.0 <= curve.d_sys <= 1.0
    assert np.all((curve.d_local >= 0.0) & (curve.d_local <= 1.0))


def test_dsys_grows_with_separation():
    rng = np.random.default_rng(4)
    non_mated = rng.normal(0.0, 0.12, 20000)
    overlap = _d_sys_curve(rng.normal(0.05, 0.12, 20000), non_mated).d_sys
    separated = _d_sys_curve(rng.normal(0.60, 0.12, 20000), non_mated).d_sys
    assert separated > overlap
    assert separated > 0.5  # clearly linkable


def _toy_segments(n_subjects=4, per_subject=5, length=256, seed=0) -> BiometricSegments:
    rng = np.random.default_rng(seed)
    ecg, ppg, labels = [], [], []
    for subj in range(1, n_subjects + 1):
        base_ecg = rng.normal(size=length)
        base_ppg = rng.normal(size=length)
        for _ in range(per_subject):
            ecg.append(base_ecg + rng.normal(scale=0.1, size=length))
            ppg.append(base_ppg + rng.normal(scale=0.1, size=length))
            labels.append(subj)
    return BiometricSegments(
        ecg=np.asarray(ecg), ppg=np.asarray(ppg),
        labels=np.asarray(labels, dtype=np.int64), sampling_rate=125,
    )


def test_dsys_undefined_for_empty_nonmated_pool():
    curve = _d_sys_curve(np.array([0.4, 0.6, 0.5]), np.empty(0, dtype=np.float64))
    assert np.isnan(curve.d_sys)
    assert curve.d_local.size == 0


def test_evaluate_cancelability_single_subject_no_crash():
    report = evaluate_cancelability(
        _toy_segments(n_subjects=1, per_subject=6),
        feature_level=3, n_keys=4, seed=0,
    )
    # Renewability/diversity remain defined (same-subject pairs exist);
    # only unlinkability is undefined for a one-subject cohort.
    assert np.isnan(report.unlinkability_d_sys)
    assert np.isfinite(report.renewability_ratio)


def test_evaluate_cancelability_smoke():
    report = evaluate_cancelability(
        _toy_segments(), feature_level=3, n_keys=4, seed=0,
    )
    assert report.n_keys == 4
    assert np.isfinite(report.renewability_ratio)
    assert report.renewability_baseline_mean <= 1.0
    assert 0.0 <= report.unlinkability_d_sys <= 1.0
    assert np.isfinite(report.diversity_mean_abs_corr)


def test_standardize_columns_centres_and_scales():
    x = np.array([[1.0, 10.0, 5.0], [3.0, 30.0, 5.0], [5.0, 50.0, 5.0]])
    z = _standardize_columns(x)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z[:, :2].std(axis=0), 1.0, atol=1e-12)
    assert np.allclose(z[:, 2], 0.0)

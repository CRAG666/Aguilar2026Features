"""Validate the inversion / leakage analysis of the multimodal hybrid transform."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.inversion import InversionReport, multimodal_leakage_metrics

FEATURE_DIM = 130  # e.g. multimodal bior3.3 level 4: 2 * 13 * 5
RNG = np.random.default_rng(seed=20260520)


def test_multimodal_leakage_metrics_returns_report_in_range():
    ecg = RNG.normal(size=FEATURE_DIM // 2)
    ppg = RNG.normal(size=FEATURE_DIM // 2)
    report = multimodal_leakage_metrics(ecg, ppg, "tok")
    assert isinstance(report, InversionReport)
    for r in (report.ecg_correlation, report.ppg_correlation, report.max_feature_correlation):
        assert -1.0 <= r <= 1.0
    assert 0.0 <= report.subspace_ratio <= 1.0


def test_multimodal_leakage_rejects_2d():
    ecg = RNG.normal(size=(2, FEATURE_DIM // 2))
    ppg = RNG.normal(size=FEATURE_DIM // 2)
    with pytest.raises(ValueError):
        multimodal_leakage_metrics(ecg, ppg, "tok")


def test_multimodal_reports_linear_ppg_baseline():
    """The report carries a controlled linear-PPG leak for the IoM comparison."""
    ecg = RNG.normal(size=FEATURE_DIM // 2)
    ppg = RNG.normal(size=FEATURE_DIM // 2)
    report = multimodal_leakage_metrics(ecg, ppg, "tok")
    assert np.isfinite(report.ppg_linear_correlation)
    assert -1.0 <= report.ppg_linear_correlation <= 1.0

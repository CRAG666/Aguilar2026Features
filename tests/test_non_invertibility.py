"""Tests for the Wu-style non-invertibility analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.dataset import BiometricSegments
from mwf.non_invertibility import (
    NonInvertibilityReport,
    non_invertibility_analysis,
)

SEGMENT_LENGTH = 256


def _synthetic_cohort(
    n_subjects: int = 6, segs_per_subject: int = 6, jitter: float = 0.05,
) -> BiometricSegments:
    """Per-subject pattern + jitter — same recipe as the stolen-token tests."""
    rng = np.random.default_rng(42)
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


def test_returns_report_and_three_populated_pools():
    report, pools = non_invertibility_analysis(
        _synthetic_cohort(), feature_level=3, denoise=False,
    )
    assert isinstance(report, NonInvertibilityReport)
    assert set(pools.keys()) == {"mated", "non_mated", "genuine_ref"}
    for population in pools.values():
        assert population.size > 0
        # absolute correlations live in [0, 1]
        assert np.all((population >= 0.0) & (population <= 1.0 + 1e-9))


def test_report_counts_match_pool_sizes():
    report, pools = non_invertibility_analysis(
        _synthetic_cohort(), feature_level=3, denoise=False,
    )
    assert report.n_mated == int(pools["mated"].size)
    assert report.n_non_mated == int(pools["non_mated"].size)
    assert report.n_genuine_ref == int(pools["genuine_ref"].size)


def test_genuine_reference_dominates_non_mated():
    """Intra-subject samples must look more alike than cross-subject pairs."""
    report, _ = non_invertibility_analysis(
        _synthetic_cohort(jitter=0.05), feature_level=3, denoise=False,
    )
    assert report.genuine_ref_mean > report.non_mated_mean


def test_sar_values_lie_in_unit_interval():
    """SAR is a fraction of probes that pass; it must land in [0, 1]."""
    report, _ = non_invertibility_analysis(
        _synthetic_cohort(), feature_level=3, denoise=False,
    )
    for sar in (report.sar_type1, report.sar_type2):
        assert np.isnan(sar) or 0.0 <= sar <= 1.0


def test_results_are_deterministic_under_fixed_seed():
    cohort = _synthetic_cohort()
    r1, _ = non_invertibility_analysis(cohort, feature_level=3, denoise=False, seed=11)
    r2, _ = non_invertibility_analysis(cohort, feature_level=3, denoise=False, seed=11)
    assert r1 == r2

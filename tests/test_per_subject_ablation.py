"""Tests for the per_subject key-vs-biometric ablation sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.dataset import BiometricSegments
from mwf.per_subject_ablation import AblationPoint, per_subject_ablation

SEGMENT_LENGTH = 256


def _synthetic_cohort(
    n_subjects: int = 8, segs_per_subject: int = 6, jitter: float = 0.05,
) -> BiometricSegments:
    rng = np.random.default_rng(0)
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


def test_returns_one_point_per_group_size():
    cohort = _synthetic_cohort()
    sizes = (1, 2, 4, 8)
    points = per_subject_ablation(
        cohort, feature_level=3, denoise=False, group_sizes=sizes,
    )
    assert len(points) == len(sizes)
    assert [p.group_size for p in points] == list(sizes)
    assert all(isinstance(p, AblationPoint) for p in points)


def test_group_size_one_has_no_within_group_impostor_pairs():
    """group_size=1 = per_subject: nobody shares a token, so no within-group impostors."""
    points = per_subject_ablation(
        _synthetic_cohort(), feature_level=3, denoise=False, group_sizes=(1,),
    )
    assert points[0].n_within_group_impostor == 0


def test_full_group_collapses_to_single_key():
    """group_size = n_subjects = single_key: every impostor pair is within the group."""
    cohort = _synthetic_cohort()
    n_subjects = int(np.unique(cohort.labels).size)
    points = per_subject_ablation(
        cohort, feature_level=3, denoise=False, group_sizes=(n_subjects,),
    )
    p = points[0]
    assert p.n_groups == 1
    assert p.n_within_group_impostor == p.n_impostor


def test_eer_monotone_or_grows_with_group_size():
    """Sharing keys can only make biometry's job harder; EER should not improve."""
    cohort = _synthetic_cohort()
    points = per_subject_ablation(
        cohort, feature_level=3, denoise=False, group_sizes=(1, 4, 8),
    )
    eers = [p.eer for p in points if np.isfinite(p.eer)]
    # tolerate small noise from the random partition
    assert eers[-1] >= eers[0] - 0.05


def test_results_are_deterministic_under_fixed_seed():
    """Same seed must give the same partition and the same EER."""
    cohort = _synthetic_cohort()
    p1 = per_subject_ablation(
        cohort, feature_level=3, denoise=False, group_sizes=(1, 4), seed=11,
    )
    p2 = per_subject_ablation(
        cohort, feature_level=3, denoise=False, group_sizes=(1, 4), seed=11,
    )
    # Dataclass equality breaks on NaN (NaN != NaN); compare the numeric
    # fields explicitly with nan-tolerant equality.
    for a, b in zip(p1, p2):
        assert a.group_size == b.group_size
        assert a.n_groups == b.n_groups
        assert a.n_genuine == b.n_genuine
        assert a.n_impostor == b.n_impostor
        np.testing.assert_allclose(a.eer, b.eer, equal_nan=True)
        np.testing.assert_allclose(a.genuine_mean, b.genuine_mean, equal_nan=True)
        np.testing.assert_allclose(
            a.within_group_impostor_mean, b.within_group_impostor_mean, equal_nan=True,
        )
        np.testing.assert_allclose(
            a.across_group_impostor_mean, b.across_group_impostor_mean, equal_nan=True,
        )

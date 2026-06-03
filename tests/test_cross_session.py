"""Tests for the cross-session / cross-activity verification protocol.

Synthetic: two cohorts share each subject's base ECG/PPG pattern but carry
condition-specific jitter, so identity persists across the simulated condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.cross_session import (
    CrossSessionResult,
    cross_session_score_pools,
    cross_session_verification,
)
from mwf.dataset import BiometricSegments

SEGMENT_LENGTH = 256


def _condition_cohort(
    cond_seed: int, n_subjects: int = 6, segs_per_subject: int = 6, jitter: float = 0.06,
    first_label: int = 1,
) -> BiometricSegments:
    """A cohort whose per-subject base is fixed (shared across conditions).

    ``cond_seed`` only drives the per-segment jitter, so two cohorts built with
    different ``cond_seed`` represent the *same subjects* recorded under two
    conditions; ``first_label`` shifts the id space for the disjoint-subject test.
    """
    rng = np.random.default_rng(cond_seed)
    t = np.linspace(0, 4 * np.pi, SEGMENT_LENGTH)
    ecg_rows, ppg_rows, labels = [], [], []
    for k in range(n_subjects):
        subject = first_label + k
        # Deterministic in the subject index → identical base in every condition.
        freq = 1.0 + 0.3 * (k + 1)
        phase = 0.4 * (k + 1)
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


def test_cross_condition_pools_nonempty_and_deterministic():
    enrol, probe = _condition_cohort(1), _condition_cohort(2)
    g1, i1 = cross_session_score_pools(enrol, probe, denoise=False, feature_level=3)
    g2, i2 = cross_session_score_pools(enrol, probe, denoise=False, feature_level=3)
    assert g1.size > 0 and i1.size > 0
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(i1, i2)


def test_genuine_separates_across_conditions():
    """Same subject across conditions scores higher than impostors → EER < chance."""
    enrol, probe = _condition_cohort(1), _condition_cohort(2)
    result = cross_session_verification(
        enrol, probe, enrol_label="A", probe_label="B", denoise=False, feature_level=3,
    )
    assert isinstance(result, CrossSessionResult)
    assert result.enrol_label == "A" and result.probe_label == "B"
    assert result.n_subjects == 6
    assert result.genuine_mean > result.impostor_mean
    assert result.eer < 0.45


def test_within_condition_separates_strongly():
    """Enrol and probe on the SAME cohort must score near-perfectly.

    Regression guard for the enrol/probe scaler-consistency bug: if the two
    transforms use different pre-projection standardisation, the cosine compares
    vectors in two z-scored spaces and the EER collapses to chance (~0.43) even
    here, where genuine probes are the enrolment segments themselves.
    """
    cohort = _condition_cohort(1, n_subjects=6)
    result = cross_session_verification(cohort, cohort, denoise=False, feature_level=3)
    assert result.eer < 0.3
    assert result.decidability > 1.0


def test_eer_within_bootstrap_ci():
    enrol, probe = _condition_cohort(1), _condition_cohort(2)
    result = cross_session_verification(enrol, probe, denoise=False, feature_level=3)
    assert result.eer_ci_low <= result.eer + 1e-9
    assert result.eer <= result.eer_ci_high + 1e-9


def test_only_common_subjects_are_scored():
    """Subjects missing from one condition (e.g. PTT's missing activity records)
    are dropped, not treated as impostors."""
    enrol = _condition_cohort(1, n_subjects=6)
    probe = _condition_cohort(2, n_subjects=4)  # only subjects 1..4 in the probe
    result = cross_session_verification(enrol, probe, denoise=False, feature_level=3)
    assert result.n_subjects == 4


def test_mismatched_segment_length_raises():
    enrol = _condition_cohort(1)
    probe = _condition_cohort(2)
    short = BiometricSegments(
        ecg=probe.ecg[:, :128], ppg=probe.ppg[:, :128],
        labels=probe.labels, sampling_rate=probe.sampling_rate,
    )
    with pytest.raises(ValueError, match="share a length"):
        cross_session_score_pools(enrol, short, denoise=False, feature_level=3)


def test_no_common_subjects_raises():
    enrol = _condition_cohort(1, n_subjects=4, first_label=1)
    probe = _condition_cohort(2, n_subjects=4, first_label=100)  # disjoint id space
    with pytest.raises(ValueError, match="No subject is present in both"):
        cross_session_score_pools(enrol, probe, denoise=False, feature_level=3)


def test_unknown_score_norm_raises():
    enrol, probe = _condition_cohort(1), _condition_cohort(2)
    with pytest.raises(ValueError, match="Unknown score_norm"):
        cross_session_score_pools(enrol, probe, score_norm="bogus", denoise=False, feature_level=3)


def test_znorm_runs_and_changes_scores():
    enrol, probe = _condition_cohort(1, n_subjects=8), _condition_cohort(2, n_subjects=8)
    raw_g, _ = cross_session_score_pools(
        enrol, probe, denoise=False, feature_level=3, score_norm=None,
    )
    zn_g, _ = cross_session_score_pools(
        enrol, probe, denoise=False, feature_level=3, score_norm="znorm",
    )
    assert raw_g.size == zn_g.size
    assert not np.allclose(raw_g, zn_g)

"""Tests for the stolen-token (lost-key) verification protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.dataset import BiometricSegments
from mwf.stolen_token import (
    StolenTokenResult,
    stolen_token_score_pools,
    stolen_token_verification,
)

SEGMENT_LENGTH = 256


def _synthetic_cohort(
    n_subjects: int = 6, segs_per_subject: int = 6, jitter: float = 0.05
) -> BiometricSegments:
    """Subjects with a distinct base ECG/PPG pattern plus small per-segment jitter.

    Same-subject segments are highly correlated, different subjects are not, so
    a biometric verifier should separate genuine from impostor even when the
    cancelable token is shared (stolen-token scenario).
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0, 4 * np.pi, SEGMENT_LENGTH)
    ecg_rows, ppg_rows, labels = [], [], []
    for subject in range(1, n_subjects + 1):  # labels must be ≥ 1
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


def test_score_pools_are_nonempty_and_deterministic():
    cohort = _synthetic_cohort()
    g1, i1 = stolen_token_score_pools(cohort, denoise=False, feature_level=3)
    g2, i2 = stolen_token_score_pools(cohort, denoise=False, feature_level=3)
    assert g1.size > 0 and i1.size > 0
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(i1, i2)


def test_genuine_separates_from_impostor_under_shared_token():
    """With real per-subject structure, genuine > impostor even when the
    attacker holds the victim's token — the system still recognises by biometry."""
    cohort = _synthetic_cohort(jitter=0.05)
    result = stolen_token_verification(cohort, denoise=False, feature_level=3)
    assert isinstance(result, StolenTokenResult)
    assert result.genuine_mean > result.impostor_mean
    assert result.eer < 0.45  # clearly better than chance


def test_znorm_option_runs_and_changes_scores():
    cohort = _synthetic_cohort()
    raw_g, _ = stolen_token_score_pools(
        cohort, denoise=False, feature_level=3, score_norm=None
    )
    zn_g, _ = stolen_token_score_pools(
        cohort, denoise=False, feature_level=3, score_norm="znorm"
    )
    assert raw_g.size == zn_g.size
    assert not np.allclose(raw_g, zn_g)


def test_unknown_score_norm_raises():
    with pytest.raises(ValueError):
        stolen_token_score_pools(_synthetic_cohort(), score_norm="bogus")

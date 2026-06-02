"""Tests for the sealed held-out test-set utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mwf.dataset import BiometricSegments
from mwf.holdout import (
    subject_holdout,
    subject_holdout_multiseed,
    temporal_holdout_per_subject,
)


def _toy_segments(n_subjects=6, per_subject=10, length=64) -> BiometricSegments:
    rng = np.random.default_rng(0)
    ecg, ppg, labels = [], [], []
    for subj in range(1, n_subjects + 1):
        for _ in range(per_subject):
            ecg.append(rng.normal(size=length))
            ppg.append(rng.normal(size=length))
            labels.append(subj)
    return BiometricSegments(
        ecg=np.asarray(ecg), ppg=np.asarray(ppg),
        labels=np.asarray(labels, dtype=np.int64), sampling_rate=125,
    )


def test_temporal_holdout_keeps_all_subjects_in_both_halves():
    seg = _toy_segments(n_subjects=4, per_subject=10)
    split = temporal_holdout_per_subject(seg, test_fraction=0.2)
    assert split.n_train + split.n_test == seg.num_segments
    all_subj = set(np.unique(seg.labels).tolist())
    assert set(np.unique(split.train.labels).tolist()) == all_subj
    assert set(np.unique(split.test.labels).tolist()) == all_subj


def test_temporal_holdout_test_indices_select_the_test_rows():
    seg = _toy_segments(n_subjects=3, per_subject=6)
    split = temporal_holdout_per_subject(seg, test_fraction=0.5)
    np.testing.assert_array_equal(seg.ecg[split.test_indices], split.test.ecg)


def test_subject_holdout_halves_share_no_subjects():
    seg = _toy_segments(n_subjects=10, per_subject=4)
    split = subject_holdout(seg, test_fraction=0.3, seed=42)
    train_subj = set(np.unique(split.train.labels).tolist())
    test_subj = set(np.unique(split.test.labels).tolist())
    assert train_subj.isdisjoint(test_subj)
    assert split.n_train + split.n_test == seg.num_segments


def test_subject_holdout_multiseed_yields_one_split_per_seed():
    seg = _toy_segments(n_subjects=10, per_subject=4)
    splits = subject_holdout_multiseed(seg, test_fraction=0.3, seeds=(1, 2, 3))
    assert len(splits) == 3
    # Different seeds should not all hold out the identical subject set.
    held_out = {frozenset(np.unique(s.test.labels).tolist()) for s in splits}
    assert len(held_out) > 1


def test_holdout_rejects_out_of_range_fraction():
    seg = _toy_segments()
    with pytest.raises(ValueError):
        temporal_holdout_per_subject(seg, test_fraction=0.0)
    with pytest.raises(ValueError):
        subject_holdout(seg, test_fraction=1.0)

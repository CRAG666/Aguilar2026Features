"""Tests for the paired model-comparison + multiple-comparison correction."""

from __future__ import annotations

import numpy as np
import pytest

from mwf.classifiers import build_classifier
from mwf.dataset import BiometricSegments
from mwf.pipeline import (
    KeyMode,
    build_templates,
    cross_validate_classifier_multiseed,
)
from mwf.significance import ComparisonRow, compare_classifiers


def _toy_segments(n_subjects=5, per_subject=8, length=256, seed=0) -> BiometricSegments:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    ecg, ppg, labels = [], [], []
    for subj in range(1, n_subjects + 1):
        phase = rng.uniform(0, 2 * np.pi)
        freq = 1.0 + 0.3 * subj
        ecg_base = np.sin(freq * t + phase)
        ppg_base = np.cos(0.5 * freq * t + phase)
        for _ in range(per_subject):
            ecg.append(ecg_base + 0.05 * rng.normal(size=length))
            ppg.append(ppg_base + 0.05 * rng.normal(size=length))
            labels.append(subj)
    return BiometricSegments(
        ecg=np.asarray(ecg), ppg=np.asarray(ppg),
        labels=np.asarray(labels, dtype=np.int64), sampling_rate=125,
    )


def _results(seg, names=("LR", "DT"), n_folds=3):
    bundle = build_templates(
        seg, feature_level=3, key_mode=KeyMode.PER_SUBJECT, denoise=False,
    )
    return {
        "per_subject@L3": {
            name: cross_validate_classifier_multiseed(
                bundle, name, build_classifier(name),
                n_folds=n_folds, split_seeds=(42, 43, 44), n_jobs=1,
            )
            for name in names
        }
    }


def test_compare_classifiers_returns_rows_with_corrected_pvalues():
    results = _results(_toy_segments())
    rows = compare_classifiers(results, ("accuracy", "f1"), n_folds=3)
    assert rows and all(isinstance(r, ComparisonRow) for r in rows)
    # One pair (LR vs DT) per metric → two comparisons.
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r.p_value <= 1.0
        # BH-adjusted p is never smaller than the raw p.
        assert r.p_value_corrected + 1e-9 >= r.p_value
        assert isinstance(r.significant, bool)


def test_bh_correction_raises_pvalues_above_raw():
    # Three classifiers → three pairs per metric → a family worth correcting.
    results = _results(_toy_segments(), names=("LR", "DT", "RF"))
    rows = compare_classifiers(results, ("accuracy",), n_folds=3)
    assert len(rows) == 3
    assert all(r.p_value_corrected + 1e-9 >= r.p_value for r in rows)


def test_empty_results_yield_no_rows():
    assert compare_classifiers({}, ("accuracy",), n_folds=3) == []

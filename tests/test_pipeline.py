"""End-to-end smoke tests for template building and identification CV."""

from __future__ import annotations

import numpy as np

from mwf.classifiers import build_classifier
from mwf.dataset import BiometricSegments
from mwf.feature_transform import multimodal_dims
from mwf.features import feature_dimension
from mwf.pipeline import (
    KeyMode,
    build_templates,
    cross_validate_classifier_multiseed,
)


def _toy_segments(n_subjects=4, per_subject=8, length=256, seed=0) -> BiometricSegments:
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


def test_identity_templates_keep_full_feature_dimension():
    seg = _toy_segments()
    bundle = build_templates(
        seg, feature_level=3, key_mode=KeyMode.IDENTITY, denoise=False,
    )
    assert bundle.key_mode is KeyMode.IDENTITY
    assert bundle.features.shape == (seg.num_segments, feature_dimension(3))


def test_per_subject_templates_are_hybrid_dim():
    """Protected template = ECG BioHash block ‖ PPG IoM one-hot block."""
    seg = _toy_segments()
    bundle = build_templates(
        seg, feature_level=3, projection_ratio=0.5,
        key_mode=KeyMode.PER_SUBJECT, denoise=False,
    )
    ecg_dim, ppg_dim = multimodal_dims(feature_dimension(3), 0.5)
    assert bundle.features.shape == (seg.num_segments, ecg_dim + ppg_dim)
    # The protected template is non-expanding overall is no longer required
    # (IoM one-hot expands the PPG block), but the ECG block stays ≤ its input.
    assert ecg_dim <= feature_dimension(3) // 2


def test_single_and_per_subject_templates_differ():
    seg = _toy_segments()
    shared = build_templates(
        seg, feature_level=3, key_mode=KeyMode.SINGLE_KEY, denoise=False,
    )
    per_subj = build_templates(
        seg, feature_level=3, key_mode=KeyMode.PER_SUBJECT, denoise=False,
    )
    assert not np.allclose(shared.features, per_subj.features)


def test_cross_validate_classifier_multiseed_runs():
    seg = _toy_segments(n_subjects=4, per_subject=8)
    bundle = build_templates(
        seg, feature_level=3, key_mode=KeyMode.PER_SUBJECT, denoise=False,
    )
    result = cross_validate_classifier_multiseed(
        bundle, "LR", build_classifier("LR"),
        n_folds=3, split_seeds=(42, 43), n_jobs=1,
    )
    assert result.n_folds == 6  # 2 seeds × 3 folds
    assert np.all(np.isfinite(result.per_metric_values("accuracy")))

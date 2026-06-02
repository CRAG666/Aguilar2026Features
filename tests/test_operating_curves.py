"""Smoke tests for DET / CMC / operating-point helpers."""

from __future__ import annotations

import numpy as np

from mwf.operating_curves import (
    cmc_curve,
    det_curve_from_scores,
    operating_points,
    pr_curve_from_scores,
    rank_k_accuracies,
    roc_curve_from_scores,
)


def _separable_scores(rng):
    return rng.normal(0.8, 0.1, 500), rng.normal(0.2, 0.1, 5000)


def test_det_curve_returns_sorted_fmr():
    rng = np.random.default_rng(0)
    g, i = _separable_scores(rng)
    det = det_curve_from_scores(g, i)
    assert det.fmr.size == det.fnmr.size == det.thresholds.size
    assert np.all(np.diff(det.fmr) <= 1e-12)  # decreasing


def test_roc_curve_auc_and_shape():
    rng = np.random.default_rng(0)
    g, i = _separable_scores(rng)
    roc = roc_curve_from_scores(g, i)
    assert roc.fpr.size == roc.tpr.size == roc.thresholds.size
    assert np.all(np.diff(roc.fpr) >= -1e-12)  # non-decreasing FPR
    assert 0.0 <= roc.auc <= 1.0
    assert roc.auc > 0.9  # separable pools


def test_roc_curve_perfect_separation():
    roc = roc_curve_from_scores(np.ones(100), np.zeros(100))
    assert roc.auc == 1.0


def test_pr_curve_ap_and_shape():
    rng = np.random.default_rng(0)
    g, i = _separable_scores(rng)
    pr = pr_curve_from_scores(g, i)
    assert pr.precision.size == pr.recall.size
    assert pr.thresholds.size == pr.precision.size - 1
    assert 0.0 <= pr.average_precision <= 1.0
    assert pr.average_precision > 0.9


def test_operating_points_keys_present():
    rng = np.random.default_rng(0)
    g, i = _separable_scores(rng)
    ops = operating_points(g, i)
    assert any(k.startswith("fnmr_at_fmr_") for k in ops)
    assert any(k.startswith("fmr_at_fnmr_") for k in ops)
    assert all(0.0 <= v <= 1.0 for v in ops.values())


def test_cmc_curve_monotonically_non_decreasing():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 200)
    classes = np.arange(10)
    y_score = rng.random((200, 10))
    y_score[np.arange(200), y_true] += 0.5

    curve = cmc_curve(y_true, y_score, classes, max_rank=5)
    assert curve.ranks.size == 5
    assert np.all(np.diff(curve.accuracies) >= -1e-12)
    assert curve.rank_k(1) <= curve.rank_k(5) <= 1.0


def test_rank_k_accuracies_default_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 200)
    classes = np.arange(10)
    y_score = rng.random((200, 10))
    y_score[np.arange(200), y_true] += 0.5

    accs = rank_k_accuracies(y_true, y_score, classes)
    assert {k for k in accs} == {
        "rank_1_accuracy", "rank_5_accuracy",
        "rank_10_accuracy", "rank_20_accuracy",
    }
    assert accs["rank_1_accuracy"] <= accs["rank_5_accuracy"]

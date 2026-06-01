"""Paired model-comparison tests with multiple-comparison correction.

Reporting means ± CI for a grid of classifiers × regimes × metrics is not enough
for a Q1 claim that one model *beats* another: the difference has to clear a
significance test, and the test has to account for two confounds at once —

  1. **Correlated CV folds.** The per-fold scores of two models compared on the
     same (repeated) k-fold partition reuse overlapping training data, so the
     naive paired t / Wilcoxon is anti-conservative. We use the Nadeau & Bengio
     (2003) corrected resampled paired t-test (:func:`mwf.stats_helpers.
     nadeau_bengio_paired_t`).
  2. **Many comparisons.** Sweeping all classifier pairs across regimes and
     metrics runs hundreds of tests; at α = 0.05 a handful look significant by
     chance alone. We control the false-discovery rate over the whole family with
     Benjamini-Hochberg (``statsmodels.stats.multitest.multipletests``).

The output is one row per (regime, metric, classifier-pair) with the raw and
corrected p-values and an FDR-controlled ``significant`` flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
from statsmodels.stats.multitest import multipletests

from .pipeline import CrossValidationResult
from .stats_helpers import nadeau_bengio_paired_t

DEFAULT_ALPHA: Final[float] = 0.05
DEFAULT_CORRECTION: Final[str] = "fdr_bh"


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One paired classifier comparison on a single (regime, metric).

    Attributes:
        regime: Key regime the two classifiers were evaluated under.
        metric: Metric compared (e.g. ``"f1"``).
        classifier_a: First classifier name.
        classifier_b: Second classifier name.
        mean_a: Mean of ``classifier_a`` over the shared folds.
        mean_b: Mean of ``classifier_b`` over the shared folds.
        mean_diff: ``mean_a − mean_b``.
        t_stat: Nadeau-Bengio corrected paired t statistic.
        p_value: Raw two-sided p-value.
        p_value_corrected: Benjamini-Hochberg adjusted p-value over the family.
        significant: ``True`` when the FDR-controlled null is rejected.
    """

    regime: str
    metric: str
    classifier_a: str
    classifier_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    t_stat: float
    p_value: float
    p_value_corrected: float
    significant: bool


def _paired_rows(
    results_by_regime: Mapping[str, Mapping[str, CrossValidationResult]],
    metrics: Sequence[str],
    n_folds: int,
) -> list[dict]:
    """Build the un-corrected comparison rows for every (regime, metric, pair).

    Classifiers are compared on the *same* folds (the CV uses a fixed,
    deterministic seed order), so the per-fold values are paired position by
    position — the precondition for the paired test.
    """
    rows: list[dict] = []
    for regime, by_clf in results_by_regime.items():
        names = sorted(by_clf)
        for metric in metrics:
            for name_a, name_b in combinations(names, 2):
                vals_a = by_clf[name_a].per_metric_values(metric)
                vals_b = by_clf[name_b].per_metric_values(metric)
                mask = np.isfinite(vals_a) & np.isfinite(vals_b)
                if mask.sum() < 2:
                    continue
                a, b = vals_a[mask], vals_b[mask]
                t_stat, p_value = nadeau_bengio_paired_t(a, b, n_folds=n_folds)
                rows.append({
                    "regime": regime,
                    "metric": metric,
                    "classifier_a": name_a,
                    "classifier_b": name_b,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_diff": float(a.mean() - b.mean()),
                    "t_stat": t_stat,
                    "p_value": p_value,
                })
    return rows


def compare_classifiers(
    results_by_regime: Mapping[str, Mapping[str, CrossValidationResult]],
    metrics: Sequence[str],
    n_folds: int,
    alpha: float = DEFAULT_ALPHA,
    correction: str = DEFAULT_CORRECTION,
) -> list[ComparisonRow]:
    """Run every pairwise classifier comparison with FDR-controlled p-values.

    Args:
        results_by_regime: ``{regime: {classifier: CrossValidationResult}}``. All
            results must share the same fold ordering (guaranteed when produced by
            :func:`mwf.cross_validate_classifier_multiseed` with identical
            ``split_seeds`` and ``n_folds``).
        metrics: Metric names to compare (each must be a field of the per-fold
            :class:`~mwf.metrics.ClassificationMetrics`).
        n_folds: Folds per CV repetition (the ``k`` of k-fold), for the
            Nadeau-Bengio variance correction.
        alpha: Family-wise FDR level.
        correction: ``statsmodels`` method, default Benjamini-Hochberg
            (``"fdr_bh"``); ``"bonferroni"`` and ``"holm"`` are also valid.

    Returns:
        One :class:`ComparisonRow` per (regime, metric, classifier-pair), with
        Benjamini-Hochberg adjusted p-values computed jointly over the whole
        family. Empty when fewer than one comparison is computable.
    """
    rows = _paired_rows(results_by_regime, metrics, n_folds)
    if not rows:
        return []
    raw_p = np.array([r["p_value"] for r in rows], dtype=np.float64)
    finite = np.isfinite(raw_p)
    corrected = np.full_like(raw_p, np.nan)
    reject = np.zeros(raw_p.shape, dtype=bool)
    if finite.any():
        rej, p_adj, _, _ = multipletests(raw_p[finite], alpha=alpha, method=correction)
        corrected[finite] = p_adj
        reject[finite] = rej
    return [
        ComparisonRow(
            regime=r["regime"],
            metric=r["metric"],
            classifier_a=r["classifier_a"],
            classifier_b=r["classifier_b"],
            mean_a=r["mean_a"],
            mean_b=r["mean_b"],
            mean_diff=r["mean_diff"],
            t_stat=r["t_stat"],
            p_value=r["p_value"],
            p_value_corrected=float(corrected[i]),
            significant=bool(reject[i]),
        )
        for i, r in enumerate(rows)
    ]


__all__ = ["ComparisonRow", "DEFAULT_ALPHA", "DEFAULT_CORRECTION", "compare_classifiers"]

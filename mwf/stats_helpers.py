"""Shared statistical micro-helpers used across the package."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import bootstrap
from scipy.stats import t as _student_t

from .constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_RESAMPLES_EVAL


def std_or_zero(samples: NDArray[np.float64]) -> float:
    """Return the unbiased sample standard deviation, or 0.0 for tiny inputs.

    Args:
        samples: 1-D array of finite floats.

    Returns:
        ``samples.std(ddof=1)`` if ``samples.size > 1``, else ``0.0``.
    """
    return float(samples.std(ddof=1)) if samples.size > 1 else 0.0


def bootstrap_ci_mean(
    samples: NDArray[np.float64],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES_EVAL,
    level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the mean of ``samples`` (BCa with percentile fallback).

    Args:
        samples: 1-D array of independent observations.
        n_resamples: Number of bootstrap resamples.
        level: Confidence level in ``(0, 1)``.
        seed: Random state forwarded to :func:`scipy.stats.bootstrap`.

    Returns:
        ``(low, high)`` confidence bounds; ``(NaN, NaN)`` if ``samples.size < 2``.
    """
    if samples.size < 2:
        return float("nan"), float("nan")
    kwargs = dict(
        statistic=np.mean,
        confidence_level=level,
        n_resamples=n_resamples,
        random_state=seed,
    )
    try:
        res = bootstrap((samples,), method="BCa", **kwargs)
    except (ValueError, RuntimeError):
        # BCa is undefined for degenerate samples (e.g. near-constant
        # observations make the acceleration term blow up); scipy raises
        # ValueError there. Fall back to the always-defined percentile CI.
        res = bootstrap((samples,), method="percentile", **kwargs)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def nadeau_bengio_ci_mean(
    samples: NDArray[np.float64],
    *,
    n_folds: int,
    level: float = BOOTSTRAP_CI_LEVEL,
) -> tuple[float, float]:
    """Corrected resampled t-CI for the mean of k-fold CV scores.

    The naive percentile/bootstrap CI treats the ``J`` fold scores as
    i.i.d. and understates uncertainty, because cross-validation folds
    reuse overlapping training data. Nadeau & Bengio (2003) inflate the
    variance of the mean by ``(1/J + n_test/n_train)``, with
    ``n_test/n_train = 1/(n_folds - 1)`` for k-fold CV, and use a Student-t
    quantile with ``J - 1`` degrees of freedom.

    Args:
        samples: 1-D array of per-fold metric values (NaNs already dropped).
        n_folds: Folds per CV repetition (the ``k`` in k-fold). Must be ≥ 2.
        level: Confidence level in ``(0, 1)``.

    Returns:
        ``(low, high)`` bounds; ``(NaN, NaN)`` if ``samples.size < 2`` or
        ``n_folds < 2``.
    """
    j = int(samples.size)
    if j < 2 or n_folds < 2:
        return float("nan"), float("nan")
    mean = float(samples.mean())
    var = float(samples.var(ddof=1))
    # n_test/n_train = 1/(n_folds - 1) for k-fold CV (Nadeau & Bengio, 2003).
    corrected_se = math.sqrt(max(var, 0.0) * (1.0 / j + 1.0 / (n_folds - 1)))
    half = float(_student_t.ppf(0.5 + level / 2.0, df=j - 1)) * corrected_se
    return mean - half, mean + half


def nadeau_bengio_paired_t(
    values_a: NDArray[np.float64],
    values_b: NDArray[np.float64],
    *,
    n_folds: int,
) -> tuple[float, float]:
    """Corrected resampled paired t-test comparing two models on CV folds.

    Tests ``H0: E[a - b] = 0`` with the Nadeau & Bengio (2003) variance
    correction on the paired per-fold differences — the recommended test
    for comparing two algorithms across (repeated) cross-validation,
    because the uncorrected paired t/Wilcoxon over reused folds is
    anti-conservative (its effective sample size is inflated).

    Args:
        values_a: 1-D per-fold scores of model A.
        values_b: 1-D per-fold scores of model B, paired with ``values_a``.
        n_folds: Folds per CV repetition. Must be ≥ 2.

    Returns:
        ``(t_statistic, two_sided_p_value)``; ``(NaN, NaN)`` when undefined
        (fewer than two pairs, ``n_folds < 2``, or zero-variance diffs).
    """
    d = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    j = int(d.size)
    if j < 2 or n_folds < 2:
        return float("nan"), float("nan")
    var_d = float(d.var(ddof=1))
    if var_d == 0.0:
        return float("nan"), float("nan")
    # n_test/n_train = 1/(n_folds - 1) for k-fold CV (Nadeau & Bengio, 2003).
    corrected_se = math.sqrt(var_d * (1.0 / j + 1.0 / (n_folds - 1)))
    t_stat = float(d.mean()) / corrected_se
    p_value = float(2.0 * _student_t.sf(abs(t_stat), df=j - 1))
    return float(t_stat), p_value


__all__ = [
    "bootstrap_ci_mean",
    "nadeau_bengio_ci_mean",
    "nadeau_bengio_paired_t",
    "std_or_zero",
]

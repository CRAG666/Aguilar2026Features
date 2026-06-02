"""Tests for the Nadeau-Bengio corrected CV statistics."""

from __future__ import annotations

import math

import numpy as np

from mwf.stats_helpers import (
    bootstrap_ci_mean,
    nadeau_bengio_ci_mean,
    nadeau_bengio_paired_t,
)


def test_nb_ci_is_wider_than_naive_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.normal(0.9, 0.02, size=25)
    lo_b, hi_b = bootstrap_ci_mean(x, seed=0)
    lo_n, hi_n = nadeau_bengio_ci_mean(x, n_folds=5)
    # The correction inflates variance, so the CI must be strictly wider.
    assert (hi_n - lo_n) > (hi_b - lo_b)
    # Centred on the sample mean.
    assert math.isclose((lo_n + hi_n) / 2.0, float(x.mean()), rel_tol=1e-9)


def test_nb_ci_correction_factor_matches_formula():
    x = np.array([0.80, 0.82, 0.78, 0.81, 0.79, 0.83, 0.77, 0.80])
    n = x.size
    var = x.var(ddof=1)
    n_folds = 4
    # Corrected SE = sqrt(var * (1/n + 1/(n_folds-1))).
    se = math.sqrt(var * (1.0 / n + 1.0 / (n_folds - 1)))
    from scipy.stats import t

    half = t.ppf(0.975, df=n - 1) * se
    lo, hi = nadeau_bengio_ci_mean(x, n_folds=n_folds, level=0.95)
    assert math.isclose(hi - lo, 2.0 * half, rel_tol=1e-9)


def test_nb_paired_t_sign_and_degenerate_guards():
    rng = np.random.default_rng(1)
    a = rng.normal(0.95, 0.02, 25)
    b = rng.normal(0.90, 0.02, 25)
    t_stat, p = nadeau_bengio_paired_t(a, b, n_folds=5)
    assert t_stat > 0  # a > b on average
    assert 0.0 <= p <= 1.0
    # Degenerate inputs return NaNs rather than raising.
    assert all(math.isnan(v) for v in nadeau_bengio_ci_mean(np.array([1.0]), n_folds=5))
    assert all(
        math.isnan(v)
        for v in nadeau_bengio_paired_t(np.ones(5), np.ones(5), n_folds=5)
    )


def test_nb_paired_t_is_more_conservative_than_uncorrected():
    rng = np.random.default_rng(2)
    a = rng.normal(0.90, 0.03, 25)
    b = rng.normal(0.89, 0.03, 25)
    d = a - b
    n = d.size
    # Uncorrected paired-t p-value.
    from scipy.stats import t

    se_naive = d.std(ddof=1) / math.sqrt(n)
    t_naive = d.mean() / se_naive
    p_naive = 2.0 * t.sf(abs(t_naive), df=n - 1)
    _, p_nb = nadeau_bengio_paired_t(a, b, n_folds=5)
    assert p_nb > p_naive  # correction widens the null, raising the p-value

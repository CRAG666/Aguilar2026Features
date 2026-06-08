"""Aggregation and bootstrap/Nadeau-Bengio CIs for cross-validation runs."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)

from .constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_RESAMPLES_EVAL,
    DEFAULT_SEED,
)
from .pipeline import CrossValidationResult
from .stats_helpers import bootstrap_ci_mean, nadeau_bengio_ci_mean, std_or_zero

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_RESAMPLES: Final[int] = BOOTSTRAP_RESAMPLES_EVAL
DEFAULT_CI_LEVEL: Final[float] = BOOTSTRAP_CI_LEVEL
DEFAULT_GLOBAL_SEED: Final[int] = DEFAULT_SEED

METRIC_NAMES: Final[tuple[str, ...]] = (
    "accuracy", "balanced_accuracy", "auc", "eer", "precision", "recall", "f1", "ap",
)

# Only group-aware splitters are offered. The biometric cohort has several
# temporally-correlated segments per subject, so any non-group strategy would
# place segments from the same subject in both train and test — identity
# (group) leakage that inflates every metric. Grouping is therefore not
# optional here, and the leak-prone strategies are deliberately absent.
CV_STRATEGIES: Final[tuple[str, ...]] = (
    "stratified_group", "group_kfold", "loso",
)


_CV_STRATEGIES: Final = MappingProxyType({
    "stratified_group": lambda n_splits, random_state: (
        StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state,
        )
    ),
    "group_kfold": lambda n_splits, _: GroupKFold(n_splits=n_splits),
    "loso": lambda *_: LeaveOneGroupOut(),
})


def make_cv_splitter(
    strategy: str = "stratified_group",
    n_splits: int = 5,
    random_state: int = DEFAULT_GLOBAL_SEED,
):
    """Build a group-aware scikit-learn CV splitter by name.

    Args:
        strategy: One of :data:`CV_STRATEGIES`.
        n_splits: Number of folds (or groups for ``loso``).
        random_state: Seed for shuffling splitters (ignored by
            ``group_kfold`` / ``loso``, which are deterministic).

    Returns:
        A scikit-learn splitter ready for ``.split(X, y, groups)``.

    Raises:
        ValueError: If ``strategy`` is not in :data:`CV_STRATEGIES`.
    """
    try:
        return _CV_STRATEGIES[strategy](n_splits, random_state)
    except KeyError as exc:
        raise ValueError(
            f"Unknown CV strategy {strategy!r}; expected one of {CV_STRATEGIES}."
        ) from exc


def set_global_seeds(seed: int = DEFAULT_GLOBAL_SEED) -> None:
    """Seed ``random`` and the legacy ``numpy.random`` globals.

    Pins fall-back RNGs used by third-party libraries (e.g. ``pyeer``).
    Does not reseed existing :class:`numpy.random.Generator` instances
    and cannot set ``PYTHONHASHSEED`` (consumed at interpreter start-up).

    Args:
        seed: Integer seed broadcast to ``random`` and ``numpy.random``.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — legacy fallback for pyeer/etc.
    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED is unset — string hashing will be non-deterministic "
            "across runs. Re-launch with `PYTHONHASHSEED=%d python …` to pin it.",
            seed,
        )
    logger.info("Global RNGs seeded with %d.", seed)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Mean, std and bootstrap CI for one metric across CV folds.

    Attributes:
        name: Metric label.
        values: Finite per-fold observations (NaNs already dropped).
        mean: Arithmetic mean of ``values``.
        std: Unbiased sample standard deviation, or ``0.0`` when ``n <= 1``.
        ci_low: Lower bootstrap-CI bound for the mean.
        ci_high: Upper bootstrap-CI bound for the mean.
    """

    name: str
    values: tuple[float, ...]
    mean: float
    std: float
    ci_low: float
    ci_high: float

    @property
    def n_obs(self) -> int:
        """Number of finite observations in the summary."""
        return len(self.values)


def summarise(
    name: str, values: Sequence[float] | NDArray[np.float64],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_GLOBAL_SEED,
    n_folds: int | None = None,
) -> MetricSummary:
    """Aggregate per-fold metric values into a :class:`MetricSummary`.

    NaNs in ``values`` are dropped before aggregation. All-NaN input yields an
    empty summary with NaN mean/std/CI. When ``n_folds`` is given, the CI uses
    the Nadeau-Bengio (2003) corrected resampled t-interval, which accounts for
    the train-set overlap that makes CV-fold scores correlated.

    Args:
        name: Metric label.
        values: Per-fold observations.
        n_resamples: Bootstrap resamples for the CI (``n_folds=None`` path).
        ci_level: Confidence level (e.g. ``0.95``).
        seed: Random state for the bootstrap (``n_folds=None`` path).
        n_folds: Folds per CV repetition; enables the corrected CI.

    Returns:
        A :class:`MetricSummary` ready for serialisation.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return MetricSummary(name, tuple(), float("nan"), float("nan"),
                             float("nan"), float("nan"))
    if n_folds is not None and n_folds >= 2:
        low, high = nadeau_bengio_ci_mean(arr, n_folds=n_folds, level=ci_level)
    else:
        low, high = bootstrap_ci_mean(
            arr, n_resamples=n_resamples, level=ci_level, seed=seed,
        )
    return MetricSummary(
        name=name, values=tuple(float(v) for v in arr),
        mean=float(arr.mean()),
        std=std_or_zero(arr),
        ci_low=low, ci_high=high,
    )


def summarise_run(
    result: CrossValidationResult,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_GLOBAL_SEED,
) -> dict[str, MetricSummary]:
    """Summarise every metric of a CV run.

    Args:
        result: One CV run.
        n_resamples: Bootstrap resamples per metric.
        ci_level: Confidence level forwarded to :func:`summarise`.
        seed: Random state for the bootstrap.

    Returns:
        Mapping ``{metric: MetricSummary}`` over :data:`METRIC_NAMES`.
    """
    n_folds = result.n_folds_per_seed or None
    return {
        m: summarise(
            m, result.per_metric_values(m), n_resamples, ci_level, seed,
            n_folds=n_folds,
        )
        for m in METRIC_NAMES
    }


__all__ = [
    "CV_STRATEGIES",
    "METRIC_NAMES",
    "MetricSummary",
    "make_cv_splitter",
    "set_global_seeds",
    "summarise",
    "summarise_run",
]

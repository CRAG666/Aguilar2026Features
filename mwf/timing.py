"""Computational-cost benchmarks for the BioHashing cancelable pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import singledispatch
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from .constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_RESAMPLES_TIMING, DEFAULT_SEED
from .stats_helpers import bootstrap_ci_mean, std_or_zero

DEFAULT_REPEATS: Final[int] = 5
DEFAULT_INNER_LOOPS: Final[int] = 1
DEFAULT_CI_LEVEL: Final[float] = BOOTSTRAP_CI_LEVEL
DEFAULT_BOOTSTRAP_RESAMPLES: Final[int] = BOOTSTRAP_RESAMPLES_TIMING


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Single benchmark observation produced by :func:`benchmark`.

    Attributes:
        label: Benchmark label.
        repeats: Outer ``perf_counter`` repetitions.
        inner_loops: Calls per repetition.
        mean_s: Mean elapsed time per repetition (seconds).
        std_s: Unbiased sample standard deviation (``0.0`` for one repeat).
        median_s: Median elapsed time per repetition (seconds).
        ci_low: Lower bound of the bootstrap CI for ``mean_s``.
        ci_high: Upper bound of the bootstrap CI for ``mean_s``.
        bytes_output: Best-effort byte size of the last produced output.
    """

    label: str
    repeats: int
    inner_loops: int
    mean_s: float
    std_s: float
    median_s: float
    ci_low: float
    ci_high: float
    bytes_output: int

    def per_call_ms(self) -> float:
        """Return the mean time per single call in milliseconds."""
        return 1000.0 * self.mean_s / max(1, self.inner_loops)


def benchmark(
    label: str, fn: Callable[[], object],
    repeats: int = DEFAULT_REPEATS,
    inner_loops: int = DEFAULT_INNER_LOOPS,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_SEED,
) -> TimingResult:
    """Benchmark ``fn`` and return a :class:`TimingResult`.

    Args:
        label: Label attached to the result.
        fn: Zero-argument callable; only its last return value is kept.
        repeats: Outer measurements used for the bootstrap CI.
        inner_loops: Calls per repetition to amortise timer noise.
        ci_level: Confidence level for the bootstrap CI.
        seed: Random state for the bootstrap.

    Returns:
        A :class:`TimingResult` with mean/std/median/CI and output size.
    """
    samples = np.empty(repeats, dtype=np.float64)
    last_output: object = None
    for r in range(repeats):
        start = time.perf_counter()
        for _ in range(inner_loops):
            last_output = fn()
        samples[r] = time.perf_counter() - start

    ci_low, ci_high = bootstrap_ci_mean(
        samples,
        n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        level=ci_level,
        seed=seed,
    )
    return TimingResult(
        label=label, repeats=repeats, inner_loops=inner_loops,
        mean_s=float(samples.mean()),
        std_s=std_or_zero(samples),
        median_s=float(np.median(samples)),
        ci_low=ci_low, ci_high=ci_high,
        bytes_output=_nbytes(last_output),
    )


@singledispatch
def _nbytes(obj: object) -> int:
    """Best-effort byte count for a benchmarked output.

    Args:
        obj: Object returned by the benchmarked function.

    Returns:
        Byte count of ``obj``, or ``0`` if it cannot be determined.
    """
    if obj is None:
        return 0
    nbytes_attr = getattr(obj, "nbytes", None)
    return int(nbytes_attr) if isinstance(nbytes_attr, (int, np.integer)) else 0


@_nbytes.register
def _(obj: np.ndarray) -> int:
    """Return the byte size of a NumPy array."""
    return int(obj.nbytes)


@_nbytes.register(list)
@_nbytes.register(tuple)
def _(obj: Sequence) -> int:
    """Sum the byte size of every element in a list or tuple."""
    return sum(_nbytes(o) for o in obj)


__all__ = ["TimingResult", "benchmark"]

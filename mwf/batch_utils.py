"""Row-wise parallel-map helper for batched signal/feature APIs."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Final, TypeVar

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray

from .progress import track, tqdm_joblib

T = TypeVar("T")
R = TypeVar("R")

# Single environment variable controls per-row batch parallelism for every
# consumer in the package. ``-1`` means "all logical cores"; ``1`` recovers
# the sequential path; positive integers cap the worker count.
BATCH_N_JOBS_ENV_VAR: Final[str] = "AGUILAR_FEATURES_N_JOBS"
DEFAULT_BATCH_N_JOBS: Final[int] = int(os.environ.get(BATCH_N_JOBS_ENV_VAR, "-1"))
DEFAULT_PARALLEL_THRESHOLD: Final[int] = 64


def parallel_row_map(
    signals: NDArray[np.float64],
    chunk_worker: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    n_jobs: int | None = None,
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
) -> NDArray[np.float64]:
    """Apply ``chunk_worker`` to row-chunks of ``signals`` in parallel.

    Row order is preserved across the parallel split and no worker sees
    rows from a different chunk. Small batches or ``n_jobs == 1`` skip
    the joblib pool entirely.

    Args:
        signals: ``(B, N)`` batch.
        chunk_worker: Callable mapping a ``(k, N)`` slab to a ``(k, …)``
            output slab; must preserve row order within the chunk.
        n_jobs: Worker count; ``None`` reads :data:`DEFAULT_BATCH_N_JOBS`.
        parallel_threshold: Minimum batch size that triggers parallelism.

    Returns:
        Concatenated worker outputs, in input row order.
    """
    b = signals.shape[0]
    effective_n_jobs = DEFAULT_BATCH_N_JOBS if n_jobs is None else n_jobs
    if effective_n_jobs == 1 or b < parallel_threshold:
        return chunk_worker(signals)
    n_workers = effective_n_jobs if effective_n_jobs > 0 else os.cpu_count() or 1
    chunks = np.array_split(signals, min(n_workers, b), axis=0)
    results = Parallel(n_jobs=effective_n_jobs, prefer="processes")(
        delayed(chunk_worker)(c) for c in chunks
    )
    return np.concatenate(results, axis=0)


def parallel_map(
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    n_jobs: int | None = None,
    parallel_threshold: int = 2,
    desc: str = "tasks",
) -> list[R]:
    """Apply ``worker`` to each item of ``items``, preserving input order.

    A thin order-preserving wrapper over :class:`joblib.Parallel` for the
    embarrassingly-parallel per-item loops of the security/leakage suite (one
    independent victim / segment / probe per item). joblib returns results in
    submission order regardless of completion order, so the output is identical
    — element for element — to the sequential ``[worker(x) for x in items]``;
    parallelism is therefore a pure wall-clock optimisation with no effect on
    any reported number. Only use it for workers whose result depends solely on
    their own item (no shared RNG state mutated across iterations).

    Small inputs or ``n_jobs == 1`` run sequentially with no joblib overhead.

    Args:
        items: Independent work items, in the order their results are wanted.
        worker: Pure callable mapping one item to its result.
        n_jobs: Worker count; ``None`` reads :data:`DEFAULT_BATCH_N_JOBS`.
        parallel_threshold: Minimum item count that triggers the joblib pool.
        desc: Progress-bar label naming the current stage.

    Returns:
        ``[worker(x) for x in items]``, computed in parallel when worthwhile.
    """
    items = list(items)
    effective_n_jobs = DEFAULT_BATCH_N_JOBS if n_jobs is None else n_jobs
    if effective_n_jobs == 1 or len(items) < parallel_threshold:
        return [worker(x) for x in track(items, desc=desc)]
    # inner_max_num_threads=1 pins each worker's BLAS to one thread: it stops the
    # workers from collectively oversubscribing the cores (dozens of processes ×
    # 64 BLAS threads) and keeps reduction orders fixed, so a worker's result is
    # bit-identical to the sequential path regardless of the pool size.
    with parallel_config(backend="loky", inner_max_num_threads=1), \
            tqdm_joblib(len(items), desc=desc):
        return list(
            Parallel(n_jobs=effective_n_jobs)(
                delayed(worker)(x) for x in items
            )
        )


__all__ = [
    "BATCH_N_JOBS_ENV_VAR",
    "DEFAULT_BATCH_N_JOBS",
    "DEFAULT_PARALLEL_THRESHOLD",
    "parallel_map",
    "parallel_row_map",
]

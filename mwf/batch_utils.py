"""Row-wise parallel-map helper for batched signal/feature APIs."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

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


__all__ = [
    "BATCH_N_JOBS_ENV_VAR",
    "DEFAULT_BATCH_N_JOBS",
    "DEFAULT_PARALLEL_THRESHOLD",
    "parallel_row_map",
]

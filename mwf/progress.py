"""Progress-bar helpers: a tqdm bridge for joblib pools and a plain wrapper.

The progress bars write to stderr and are independent of the logging level, so
a run always shows what stage it is on and how far along it is — even without
``--verbose``. Set ``AGUILAR_FEATURES_NO_PROGRESS=1`` to silence every bar (e.g.
in non-interactive batch jobs or test runs).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator
from typing import Final, TypeVar

import joblib
from tqdm.auto import tqdm

T = TypeVar("T")

# Opt-out switch so SLURM/CI logs are not flooded with carriage-return bars.
_NO_PROGRESS_ENV_VAR: Final[str] = "AGUILAR_FEATURES_NO_PROGRESS"


def progress_disabled() -> bool:
    """Return ``True`` when bars are suppressed via the opt-out env var."""
    return os.environ.get(_NO_PROGRESS_ENV_VAR, "") not in ("", "0", "false")


def track(iterable: Iterable[T], desc: str, *, total: int | None = None) -> Iterable[T]:
    """Wrap ``iterable`` in a tqdm bar labelled ``desc`` (no-op when disabled)."""
    return tqdm(iterable, desc=desc, total=total, disable=progress_disabled())


@contextlib.contextmanager
def tqdm_joblib(total: int | None, desc: str) -> Iterator[tqdm]:
    """Route a :class:`joblib.Parallel` run's completions into a tqdm bar.

    joblib exposes no native progress hook, so this temporarily replaces its
    batch-completion callback for the duration of the ``with`` block; each
    finished task advances the bar. The original callback is always restored on
    exit, so nested or subsequent ``Parallel`` calls are unaffected.

    Args:
        total: Number of tasks dispatched; ``None`` leaves the bar unbounded.
        desc: Bar label, shown as the current stage.

    Yields:
        The active tqdm bar.
    """
    bar = tqdm(total=total, desc=desc, disable=progress_disabled())
    original = joblib.parallel.BatchCompletionCallBack

    class _TqdmCallback(original):
        def __call__(self, *args: object, **kwargs: object) -> object:
            bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _TqdmCallback
    try:
        yield bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original
        bar.close()


__all__ = ["progress_disabled", "track", "tqdm_joblib"]

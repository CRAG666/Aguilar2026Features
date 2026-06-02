"""NeuroKit physiological cleaning of ECG/PPG before feature extraction.

This is the *Preprocesado* stage of the cancelable pipeline. It uses
modality-specific NeuroKit cleaners, which exploit the known passbands of each
waveform instead of a generic wavelet shrinkage:

* ECG (``method="neurokit"``): 0.5 Hz high-pass Butterworth (baseline-wander
  removal) followed by powerline filtering.
* PPG (``method="elgendi"``): 0.5-8 Hz band-pass Butterworth.

Both ``method`` strings are forwarded verbatim to ``neurokit2.ecg_clean`` /
``neurokit2.ppg_clean`` together with the cohort sampling rate, so the cleaning
is reproducible and citable. Cleaning preserves the segment length, so the
``(B, N)`` batch shape flows unchanged into :mod:`mwf.features`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Final

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray

from .batch_utils import parallel_row_map
from .constants import DEFAULT_ECG_CLEAN_METHOD, DEFAULT_PPG_CLEAN_METHOD
from .validation import ensure_2d_batch

DEFAULT_ECG_METHOD: Final[str] = DEFAULT_ECG_CLEAN_METHOD
DEFAULT_PPG_METHOD: Final[str] = DEFAULT_PPG_CLEAN_METHOD


def _validate(signal: NDArray[np.float64], sampling_rate: int) -> None:
    """Validate a single-segment cleaning request.

    Args:
        signal: Candidate 1-D segment.
        sampling_rate: Candidate sampling frequency in Hz.

    Raises:
        ValueError: If ``signal`` is not 1-D or ``sampling_rate <= 0``.
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-D.")
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be > 0; got {sampling_rate}.")


# NeuroKit cleaners share one signature ``(signal, sampling_rate=, method=)`` and
# one calling convention (suppress their chatty warnings, coerce to float64).
_NkCleaner = Callable[..., NDArray[np.float64]]


def _clean_one(
    signal: NDArray[np.float64],
    sampling_rate: int,
    method: str,
    cleaner: _NkCleaner,
) -> NDArray[np.float64]:
    """Validate, then run one NeuroKit cleaner with warnings suppressed."""
    _validate(signal, sampling_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = cleaner(
            np.asarray(signal, dtype=np.float64),
            sampling_rate=sampling_rate,
            method=method,
        )
    return np.asarray(cleaned, dtype=np.float64)


def clean_ecg(
    signal: NDArray[np.float64],
    sampling_rate: int,
    method: str = DEFAULT_ECG_METHOD,
) -> NDArray[np.float64]:
    """Clean a 1-D ECG segment with ``neurokit2.ecg_clean``.

    Args:
        signal: 1-D float64 ECG segment.
        sampling_rate: Sampling frequency in Hz.
        method: Cleaning method forwarded to ``neurokit2.ecg_clean``.

    Returns:
        Cleaned 1-D float64 array of the same length as ``signal``.

    Raises:
        ValueError: If ``signal`` is not 1-D or ``sampling_rate <= 0``.
    """
    return _clean_one(signal, sampling_rate, method, nk.ecg_clean)


def clean_ppg(
    signal: NDArray[np.float64],
    sampling_rate: int,
    method: str = DEFAULT_PPG_METHOD,
) -> NDArray[np.float64]:
    """Clean a 1-D PPG segment with ``neurokit2.ppg_clean``.

    Args:
        signal: 1-D float64 PPG segment.
        sampling_rate: Sampling frequency in Hz.
        method: Cleaning method forwarded to ``neurokit2.ppg_clean``.

    Returns:
        Cleaned 1-D float64 array of the same length as ``signal``.

    Raises:
        ValueError: If ``signal`` is not 1-D or ``sampling_rate <= 0``.
    """
    return _clean_one(signal, sampling_rate, method, nk.ppg_clean)


def _clean_chunk(
    chunk: NDArray[np.float64],
    clean_one,
    sampling_rate: int,
    method: str,
) -> NDArray[np.float64]:
    """Apply a single-segment cleaner to every row of a slab.

    Args:
        chunk: ``(k, N)`` slab.
        clean_one: :func:`clean_ecg` or :func:`clean_ppg`.
        sampling_rate: Forwarded to ``clean_one``.
        method: Forwarded to ``clean_one``.

    Returns:
        ``(k, N)`` cleaned array in input row order.
    """
    return np.stack([
        clean_one(row, sampling_rate, method) for row in chunk
    ])


def _clean_batch(
    signals: NDArray[np.float64],
    clean_one: Callable[..., NDArray[np.float64]],
    sampling_rate: int,
    method: str,
    n_jobs: int | None,
) -> NDArray[np.float64]:
    """Row-parallel cleaning scaffold shared by the ECG/PPG batch entry points."""
    ensure_2d_batch(signals, name="signals")

    def _worker(chunk: NDArray[np.float64]) -> NDArray[np.float64]:
        return _clean_chunk(chunk, clean_one, sampling_rate, method)

    return parallel_row_map(signals, _worker, n_jobs=n_jobs)


def clean_ecg_batch(
    signals: NDArray[np.float64],
    sampling_rate: int,
    method: str = DEFAULT_ECG_METHOD,
    n_jobs: int | None = None,
) -> NDArray[np.float64]:
    """Clean every row of a ``(B, N)`` ECG batch in parallel.

    Args:
        signals: ``(B, N)`` float64 ECG batch.
        sampling_rate: Sampling frequency in Hz.
        method: Cleaning method forwarded to ``neurokit2.ecg_clean``.
        n_jobs: Worker count forwarded to :func:`parallel_row_map`.

    Returns:
        ``(B, N)`` cleaned array in input row order.
    """
    return _clean_batch(signals, clean_ecg, sampling_rate, method, n_jobs)


def clean_ppg_batch(
    signals: NDArray[np.float64],
    sampling_rate: int,
    method: str = DEFAULT_PPG_METHOD,
    n_jobs: int | None = None,
) -> NDArray[np.float64]:
    """Clean every row of a ``(B, N)`` PPG batch in parallel.

    Args:
        signals: ``(B, N)`` float64 PPG batch.
        sampling_rate: Sampling frequency in Hz.
        method: Cleaning method forwarded to ``neurokit2.ppg_clean``.
        n_jobs: Worker count forwarded to :func:`parallel_row_map`.

    Returns:
        ``(B, N)`` cleaned array in input row order.
    """
    return _clean_batch(signals, clean_ppg, sampling_rate, method, n_jobs)


__all__ = [
    "DEFAULT_ECG_METHOD",
    "DEFAULT_PPG_METHOD",
    "clean_ecg",
    "clean_ecg_batch",
    "clean_ppg",
    "clean_ppg_batch",
]

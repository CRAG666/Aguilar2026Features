"""Additive white Gaussian noise utilities for robustness experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .rng import make_rng
from .validation import ensure_2d_batch

RngLike = np.random.Generator | int | None


def _coerce_rng(rng: RngLike) -> np.random.Generator:
    """Return ``rng`` as a ``Generator``, building one from a seed if needed.

    Args:
        rng: Existing ``Generator``, integer seed, or ``None``.

    Returns:
        A ``numpy.random.Generator`` instance.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return make_rng(rng)


def add_awgn_batch(
    signals: NDArray[np.float64],
    snr_db: float,
    seed: RngLike = None,
) -> NDArray[np.float64]:
    """Add row-wise AWGN at a fixed SNR to a batch of signals.

    Noise variance is computed per row from its empirical power, so the
    requested SNR is honoured even when row energies differ widely. Rows
    with zero power receive zero noise.

    Args:
        signals: ``(B, N)`` float64 batch; each row is processed independently.
        snr_db: Target signal-to-noise ratio in decibels.
        seed: ``Generator``, integer seed, or ``None``.

    Returns:
        A new ``(B, N)`` array equal to ``signals`` plus noise.
    """
    ensure_2d_batch(signals, name="signals")
    rng = _coerce_rng(seed)
    power = np.mean(signals**2, axis=1, keepdims=True)
    noise_power = np.where(power > 0, power / (10.0 ** (snr_db / 10.0)), 0.0)
    noise = rng.normal(size=signals.shape) * np.sqrt(noise_power)
    return signals + noise


__all__ = ["add_awgn_batch"]

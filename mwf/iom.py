"""Index-of-Max (IoM) hashing — the non-invertible cancelable transform for PPG.

Implements the Gaussian-Random-Projection variant of IoM hashing (Jin, Hwang,
Lai, Kim & Teoh, "Ranking-Based Locality Sensitive Hashing-Enabled Cancelable
Biometrics: Index-of-Max Hashing", IEEE TIFS 13(2), 2018).

For a feature vector ``v ∈ ℝ^d`` and a token, the token seeds ``m`` independent
Gaussian matrices ``W_i ∈ ℝ^{q×d}``. Each hash keeps only the **index of the
maximum** response, ``h_i = argmax(W_i v) ∈ {0,…,q−1}``; the protected code is
``h = (h_1,…,h_m)``. Magnitudes are discarded.

Why this is the right tool for the two PPG goals (see :mod:`mwf.constants`):

* **Non-invertible.** Only the winning index of each hash survives, so the
  magnitude information a linear (min-norm ``Rᵀy``) attack needs is gone.
  Recovering ``v`` from the indices is a hard combinatorial feasibility problem,
  not a matrix transpose — the ~``√(m/d)`` correlation leak of a plain
  orthonormal projection no longer applies.
* **Similarity-preserving (LSH), key-independent.** The probability that two
  codes agree on a hash depends only on the angle between the two vectors, *not*
  on the token. So genuine pairs collide more than impostor pairs even under a
  shared (stolen) token — the biometric, not the key, carries discriminability.
* **Revocable / unlinkable.** A new token reseeds every ``W_i`` into an
  unrelated code.

The code is returned **one-hot** (each index → a length-``q`` indicator block),
because the inner product of two one-hot codes equals their collision count.
That keeps the existing cosine matcher and classifier pipeline applicable with
no special distance: cosine on the one-hot code *is* the IoM collision rate.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .constants import IOM_HASHES_RATIO, IOM_WINDOW
from .keystream import keystream_rng

DEFAULT_WINDOW: Final[int] = IOM_WINDOW
DEFAULT_HASHES_RATIO: Final[float] = IOM_HASHES_RATIO
_MIN_HASHES: Final[int] = 1
_MIN_WINDOW: Final[int] = 2  # argmax over a single projection is degenerate


def hash_count(ppg_dim: int, ratio: float = DEFAULT_HASHES_RATIO) -> int:
    """Resolve the IoM code length ``m`` from a ratio of the PPG block size.

    Args:
        ppg_dim: PPG feature-block dimension ``d`` (must be ≥ 1).
        ratio: Target ``m / d`` (must be > 0).

    Returns:
        ``m = max(1, round(ratio * ppg_dim))``.

    Raises:
        ValueError: If ``ppg_dim < 1`` or ``ratio <= 0``.
    """
    if ppg_dim < 1:
        raise ValueError(f"ppg_dim must be ≥ 1; got {ppg_dim}.")
    if ratio <= 0.0:
        raise ValueError(f"ratio must be > 0; got {ratio}.")
    return max(_MIN_HASHES, round(ratio * ppg_dim))


def iom_dim(n_hashes: int, window: int = DEFAULT_WINDOW) -> int:
    """Return the one-hot code length ``m · q``.

    Args:
        n_hashes: Number of hashes ``m``.
        window: Projections per hash ``q``.

    Returns:
        ``n_hashes * window``.
    """
    return int(n_hashes * window)


def derive_iom(
    token: str,
    in_dim: int,
    n_hashes: int,
    window: int = DEFAULT_WINDOW,
) -> NDArray[np.float64]:
    """Build the token-seeded Gaussian projection stack for IoM hashing.

    Args:
        token: Cancelable-template token (key ``k``).
        in_dim: PPG feature-block dimension ``d``.
        n_hashes: Number of hashes ``m`` (must be ≥ 1).
        window: Projections per hash ``q`` (must be ≥ 2).

    Returns:
        Array ``W`` of shape ``(n_hashes, window, in_dim)``; ``W[i]`` is the
        ``q × d`` Gaussian whose row-projections are ranked by :func:`iom_indices`.

    Raises:
        ValueError: If ``n_hashes < 1`` or ``window < 2``.
    """
    if n_hashes < _MIN_HASHES:
        raise ValueError(f"n_hashes must be ≥ {_MIN_HASHES}; got {n_hashes}.")
    if window < _MIN_WINDOW:
        raise ValueError(f"window must be ≥ {_MIN_WINDOW}; got {window}.")
    rng = keystream_rng(token)
    return rng.standard_normal((n_hashes, window, in_dim))


def iom_indices(
    features: NDArray[np.float64],
    token: str,
    n_hashes: int,
    window: int = DEFAULT_WINDOW,
) -> NDArray[np.int64]:
    """Compute the integer IoM code for a ``(B, d)`` batch (or one ``(d,)`` row).

    Args:
        features: ``(B, d)`` PPG feature block, or a single ``(d,)`` vector.
        token: Token seeding the Gaussian stack.
        n_hashes: Number of hashes ``m``.
        window: Projections per hash ``q``.

    Returns:
        ``(B, m)`` integer indices in ``[0, q)`` (or ``(m,)`` for a 1-D input).
    """
    x = np.asarray(features, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]
    in_dim = x.shape[1]
    w = derive_iom(token, in_dim, n_hashes, window)             # (m, q, d)
    flat = w.reshape(n_hashes * window, in_dim)                 # (m*q, d)
    proj = (x @ flat.T).reshape(x.shape[0], n_hashes, window)   # (B, m, q)
    idx = np.argmax(proj, axis=2).astype(np.int64)              # (B, m)
    return idx[0] if single else idx


def iom_onehot(
    features: NDArray[np.float64],
    token: str,
    n_hashes: int,
    window: int = DEFAULT_WINDOW,
    normalise: bool = True,
) -> NDArray[np.float64]:
    """Return the one-hot IoM code for a ``(B, d)`` batch (or one ``(d,)`` row).

    Args:
        features: ``(B, d)`` PPG feature block, or a single ``(d,)`` vector.
        token: Token seeding the Gaussian stack.
        n_hashes: Number of hashes ``m``.
        window: Projections per hash ``q``.
        normalise: If ``True``, unit-L2-normalise each row so cosine equals the
            collision rate.

    Returns:
        ``(B, m*window)`` one-hot code (or ``(m*window,)`` for a 1-D input).
    """
    x = np.asarray(features, dtype=np.float64)
    single = x.ndim == 1
    idx = iom_indices(x, token, n_hashes, window)
    if single:
        idx = idx[np.newaxis, :]
    b, m = idx.shape
    code = np.zeros((b, m, window), dtype=np.float64)
    code[np.arange(b)[:, None], np.arange(m)[None, :], idx] = 1.0
    code = code.reshape(b, m * window)
    if normalise:  # unit L2 per row → row inner product equals collision rate
        code /= np.sqrt(float(m))
    return code[0] if single else code


__all__ = [
    "DEFAULT_HASHES_RATIO",
    "DEFAULT_WINDOW",
    "derive_iom",
    "hash_count",
    "iom_dim",
    "iom_indices",
    "iom_onehot",
]

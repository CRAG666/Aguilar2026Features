"""Token-keyed BioHashing transform for multimodal wavelet-feature templates.

This is the *Transformación cancelable* ``T(x, k)`` of the architecture: it maps
the extracted multimodal feature vector ``x ∈ ℝ^d`` (see :mod:`mwf.features`) to
a protected template ``T(x, k) ∈ ℝ^m`` under a user token ``k``.

The transform is a **random orthonormal projection** (BioHashing, Teoh et al.
2004): the token seeds — via SHA-256 (:mod:`mwf.keystream`) — a Gaussian matrix
whose rows are orthonormalised (QR) into a projection ``R ∈ ℝ^{m×d}`` with
``R Rᵀ = I_m``; the template is ``y = R x`` (optionally sign-binarised into a
``±1`` BioCode).

Properties this buys, all of which the design relies on:

* **Renewability / diversity** (ISO/IEC 30136): a one-bit token edit yields a
  fully decorrelated SHA-256 digest, hence an independent ``R`` and an unrelated
  template — re-issue by handing out a fresh token.
* **Non-invertibility**: when ``m < d`` the projection is many-to-one. Its
  ``(d − m)``-dimensional null space is destroyed, so even an adversary who
  knows both ``R`` and ``y`` recovers only ``x``'s component in the row space of
  ``R`` (a min-norm pre-image ``Rᵀ y``), never ``x`` itself. The
  :data:`projection_ratio` ``m/d`` is the irreversibility budget — smaller
  ratios leak less but discard more discriminative information.
* **Determinism**: ``R`` is a pure function of ``(token, d, m)``, so enrolment
  and verification reproduce the same projection bit-for-bit.

Multimodality enters upstream: ``x`` already concatenates the ECG and PPG
descriptor blocks, so ``R`` mixes both modalities into every output coordinate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from . import iom
from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_STANDARDIZE,
    IOM_HASHES_RATIO,
    IOM_WINDOW,
)
from .keystream import keystream_rng
from .scoring import l2_normalise

DEFAULT_RATIO: Final[float] = DEFAULT_PROJECTION_RATIO
MIN_OUT_DIM: Final[int] = 1
# Per-modality token salts: the same user token seeds two *independent* sub-
# transforms (ECG BioHashing vs PPG IoM), so the salts keep their random
# material decorrelated while a single token still drives the whole template.
ECG_SALT: Final[str] = "::ECG"
PPG_SALT: Final[str] = "::PPG"


@dataclass(frozen=True, slots=True)
class ProjectionKey:
    """Token-derived random-projection material.

    Attributes:
        matrix: The projection ``R ∈ ℝ^{m×d}`` with orthonormal rows
            (``R Rᵀ = I_m``). Rows are drawn from a SHA-256-seeded Gaussian and
            orthonormalised by QR, so ``R`` is decorrelated across tokens.
    """

    matrix: NDArray[np.float64]

    @property
    def out_dim(self) -> int:
        """Protected-template length ``m`` (number of projection rows)."""
        return int(self.matrix.shape[0])

    @property
    def in_dim(self) -> int:
        """Input feature dimension ``d`` (number of projection columns)."""
        return int(self.matrix.shape[1])


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    """Per-feature z-score statistics for the pre-projection standardisation.

    Standardising ``x`` before the BioHashing projection equalises every
    descriptor's contribution. Without it the high-variance descriptors (e.g.
    sub-band energy) dominate the random projection and bury the lower-variance
    but discriminative ones — which is what collapses accuracy under a shared
    token, where all subjects are mapped by the same ``R`` and the projection
    cannot adapt. These are non-secret enrolment-time normalisation parameters
    (the same status as the wavelet basis), *not* part of the token ``k``.

    Attributes:
        mean: Per-feature mean ``μ ∈ ℝ^d`` estimated on the enrolment cohort.
        scale: Per-feature standard deviation ``σ ∈ ℝ^d`` with zeros replaced by
            ``1.0``, so constant features pass through unscaled (no ÷0).
    """

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fit(cls, features: NDArray[np.float64]) -> "FeatureScaler":
        """Estimate per-feature ``(μ, σ)`` from an enrolment feature matrix.

        Args:
            features: ``(B, d)`` feature matrix.

        Returns:
            A :class:`FeatureScaler` whose ``scale`` guards zero-variance dims.

        Raises:
            ValueError: If ``features`` is not 2-D.
        """
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("features must be 2-D (B, d) to fit a FeatureScaler.")
        scale = feats.std(axis=0)
        return cls(mean=feats.mean(axis=0), scale=np.where(scale > 0.0, scale, 1.0))

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score ``x`` with the stored statistics (broadcasts over rows).

        Args:
            x: ``(d,)`` vector or ``(B, d)`` matrix to standardise.

        Returns:
            ``(x - μ) / σ`` as float64.
        """
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale


def projection_dim(in_dim: int, ratio: float = DEFAULT_RATIO) -> int:
    """Resolve the protected-template length ``m`` from a ratio ``m/d``.

    Args:
        in_dim: Input feature dimension ``d`` (must be ≥ 1).
        ratio: Target ``m / d`` in ``(0, 1]``.

    Returns:
        ``m = clip(round(ratio * d), 1, d)``.

    Raises:
        ValueError: If ``in_dim < 1`` or ``ratio`` is outside ``(0, 1]``.
    """
    if in_dim < MIN_OUT_DIM:
        raise ValueError(f"in_dim must be ≥ {MIN_OUT_DIM}; got {in_dim}.")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must lie in (0, 1]; got {ratio}.")
    return int(min(in_dim, max(MIN_OUT_DIM, round(ratio * in_dim))))


def derive_projection(token: str, in_dim: int, out_dim: int) -> ProjectionKey:
    """Build the orthonormal projection for one token.

    A SHA-256-seeded generator draws a ``d × m`` Gaussian; its (thin) QR
    factor supplies ``m`` orthonormal columns, transposed into a ``m × d``
    projection with orthonormal rows. The seeding gives the avalanche
    (token sensitivity); the Gaussian-then-QR step gives a Haar-distributed
    orthonormal frame.

    Args:
        token: User token (key ``k``).
        in_dim: Input feature dimension ``d``.
        out_dim: Protected-template length ``m`` (``1 ≤ m ≤ d``).

    Returns:
        A :class:`ProjectionKey` wrapping ``R ∈ ℝ^{m×d}``.

    Raises:
        ValueError: If ``out_dim`` is outside ``[1, in_dim]``.
    """
    if not MIN_OUT_DIM <= out_dim <= in_dim:
        raise ValueError(
            f"out_dim must lie in [{MIN_OUT_DIM}, {in_dim}]; got {out_dim}."
        )
    rng = keystream_rng(token)
    gaussian = rng.standard_normal((in_dim, out_dim))
    q, _ = np.linalg.qr(gaussian)  # q: (d, m), orthonormal columns
    return ProjectionKey(matrix=np.ascontiguousarray(q.T))


def _apply(matrix: NDArray[np.float64], x: NDArray[np.float64], binarise: bool) -> NDArray[np.float64]:
    """Project ``x`` through ``matrix`` and optionally sign-binarise.

    Args:
        matrix: Projection ``R ∈ ℝ^{m×d}``.
        x: Feature vector(s); ``(d,)`` or ``(d, B)``.
        binarise: If ``True``, map the projection to ``±1`` by its sign
            (zeros map to ``+1``).

    Returns:
        ``R x`` (real-valued) or its sign (binarised BioCode).
    """
    y = matrix @ x
    return np.where(y >= 0.0, 1.0, -1.0) if binarise else y


# ---------------------------------------------------------------------------
# Multimodal hybrid transform: ECG → BioHashing, PPG → IoM hashing
# ---------------------------------------------------------------------------
# The operational template protects the two modalities differently. The ECG
# block keeps the orthonormal BioHashing projection (real-valued, ~√ratio
# recoverable). The PPG block — the one a linear inversion attack reconstructs
# at r≈0.78 — is replaced by Index-of-Max hashing (:mod:`mwf.iom`), which is
# strongly non-invertible and key-independent in its similarity structure. Each
# block is unit-L2-normalised so that cosine on the concatenation fuses the two
# modalities with equal weight (and so the IoM half contributes its collision
# rate directly).


def _split_modalities(d: int) -> int:
    """Return the ECG/PPG split point for a length-``d`` multimodal vector.

    Features concatenate equal-length ECG then PPG blocks (see
    :mod:`mwf.features`), so the split is the midpoint.

    Raises:
        ValueError: If ``d`` is not even (not a balanced ECG‖PPG vector).
    """
    if d % 2 != 0:
        raise ValueError(f"multimodal feature dim must be even (ECG‖PPG); got {d}.")
    return d // 2


def multimodal_dims(
    in_dim: int,
    projection_ratio: float = DEFAULT_RATIO,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
) -> tuple[int, int]:
    """Return ``(ecg_dim, ppg_dim)`` of the hybrid template for a given input.

    Args:
        in_dim: Multimodal feature dimension ``d`` (even; ECG‖PPG).
        projection_ratio: BioHashing ``m/d_ecg`` for the ECG block.
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.

    Returns:
        ``(ecg_template_dim, ppg_onehot_dim)``; the full template length is
        their sum.
    """
    half = _split_modalities(in_dim)
    ecg_dim = projection_dim(half, projection_ratio)
    ppg_dim = iom.iom_dim(iom.hash_count(half, n_hashes_ratio), window)
    return ecg_dim, ppg_dim


def transform_multimodal(
    x: NDArray[np.float64],
    token: str,
    projection_ratio: float = DEFAULT_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
    scaler: FeatureScaler | None = None,
) -> NDArray[np.float64]:
    """Protect one multimodal feature vector with the hybrid ECG/PPG transform.

    The ECG half is BioHashed (orthonormal projection); the PPG half is
    Index-of-Max hashed. Each protected block is unit-L2-normalised, then they
    are concatenated.

    Args:
        x: 1-D multimodal feature vector ``∈ ℝ^d`` (even ``d``; ECG‖PPG).
        token: Cancelable-template token ``k`` (salted per modality internally).
        projection_ratio: BioHashing ``m/d_ecg`` for the ECG block.
        binarise: If ``True``, sign-binarise the ECG BioHashing block.
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.
        scaler: Optional pre-fitted :class:`FeatureScaler` applied to ``x``
            before splitting (leakage-free standardisation).

    Returns:
        1-D hybrid template ``[ECG BioHash | PPG IoM one-hot]``.

    Raises:
        ValueError: If ``x`` is not 1-D or has odd length.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1-D feature vector.")
    x_std = scaler.apply(x) if scaler is not None else x.astype(np.float64, copy=False)
    half = _split_modalities(int(x_std.shape[0]))
    ecg, ppg = x_std[:half], x_std[half:]

    m_ecg = projection_dim(half, projection_ratio)
    ecg_key = derive_projection(token + ECG_SALT, half, m_ecg)
    ecg_t = _apply(ecg_key.matrix, ecg, binarise)
    ecg_t = l2_normalise(ecg_t[np.newaxis, :])[0]

    n_hashes = iom.hash_count(half, n_hashes_ratio)
    ppg_t = iom.iom_onehot(ppg, token + PPG_SALT, n_hashes, window, normalise=True)
    return np.concatenate([ecg_t, ppg_t])


def transform_multimodal_batch(
    features: NDArray[np.float64],
    tokens: Sequence[str],
    projection_ratio: float = DEFAULT_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
    standardize: bool = DEFAULT_STANDARDIZE,
    scaler: FeatureScaler | None = None,
) -> NDArray[np.float64]:
    """Hybrid ECG-BioHash / PPG-IoM transform over a ``(B, d)`` batch.

    Standardisation mirrors :func:`transform_batch` (fit on the cohort unless a
    ``scaler`` is supplied), after which each row's ECG half is BioHashed and
    PPG half is IoM-hashed. Rows are grouped by token so each distinct token's
    projection material is derived once.

    Args:
        features: ``(B, d)`` multimodal feature matrix (even ``d``; ECG‖PPG).
        tokens: One token per row.
        projection_ratio: BioHashing ``m/d_ecg`` for the ECG block.
        binarise: If ``True``, sign-binarise the ECG BioHashing block.
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.
        standardize: If ``True`` (and no ``scaler``), fit a :class:`FeatureScaler`
            on ``features`` and z-score before transforming.
        scaler: Pre-fitted scaler to apply instead (leakage-free CV); overrides
            ``standardize`` when given.

    Returns:
        ``(B, ecg_dim + ppg_dim)`` hybrid template matrix in input row order.

    Raises:
        ValueError: If ``features`` is not 2-D, ``len(tokens)`` mismatches, or
            ``d`` is odd.
    """
    if features.ndim != 2:
        raise ValueError("Batch features must be 2-D (B, d).")
    if len(tokens) != features.shape[0]:
        raise ValueError("`tokens` must have one entry per row of `features`.")

    if scaler is None and standardize:
        scaler = FeatureScaler.fit(features)
    feats = scaler.apply(features) if scaler is not None else np.asarray(features, dtype=np.float64)

    d = int(feats.shape[1])
    half = _split_modalities(d)
    m_ecg = projection_dim(half, projection_ratio)
    n_hashes = iom.hash_count(half, n_hashes_ratio)
    out_dim = m_ecg + iom.iom_dim(n_hashes, window)
    out = np.empty((feats.shape[0], out_dim), dtype=np.float64)

    tokens_arr = np.asarray(tokens, dtype=object)
    for token in dict.fromkeys(tokens):  # unique tokens, first-seen order
        rows = np.flatnonzero(tokens_arr == token)
        ecg_block = feats[rows, :half]
        ppg_block = feats[rows, half:]

        ecg_key = derive_projection(token + ECG_SALT, half, m_ecg)
        ecg_t = ecg_block @ ecg_key.matrix.T
        if binarise:
            ecg_t = np.where(ecg_t >= 0.0, 1.0, -1.0)
        ecg_t = l2_normalise(ecg_t)

        ppg_t = iom.iom_onehot(ppg_block, token + PPG_SALT, n_hashes, window, normalise=True)
        out[rows] = np.concatenate([ecg_t, ppg_t], axis=1)
    return out


__all__ = [
    "ECG_SALT",
    "FeatureScaler",
    "PPG_SALT",
    "ProjectionKey",
    "derive_projection",
    "multimodal_dims",
    "projection_dim",
    "transform_multimodal",
    "transform_multimodal_batch",
]

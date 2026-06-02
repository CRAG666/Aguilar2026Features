"""Inversion analysis for the BioHashing feature transform.

The cancelable transform applies, to the multimodal feature vector ``x ∈ ℝ^d``:
  1. a token-keyed orthonormal projection ``y = R x`` with ``R ∈ ℝ^{m×d}``,
     ``R Rᵀ = I_m`` and ``m ≤ d``;
  2. (optionally) a sign binarisation.

Given the token, ``R`` is reproducible, so an adversary can compute the
min-norm pre-image ``x̂ = Rᵀ y = Rᵀ R x`` — the orthogonal projection of ``x``
onto the ``m``-dimensional row space of ``R``. The complementary ``(d − m)``-
dimensional null space is annihilated by the forward transform and cannot be
recovered. This module quantifies the residual leakage: how much of the
original ECG/PPG descriptors survives in that best-case reconstruction.

The binarisation case is *strictly harder* to invert (it additionally discards
every magnitude), so the real-valued pre-image analysed here is a conservative
(adversary-favouring) upper bound on what is recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from . import iom
from .constants import IOM_HASHES_RATIO, IOM_WINDOW
from .feature_transform import (
    ECG_SALT,
    PPG_SALT,
    DEFAULT_BINARISE,
    DEFAULT_RATIO,
    _apply,
    derive_projection,
    projection_dim,
)


@dataclass(frozen=True, slots=True)
class InversionReport:
    """Quantified information leakage after inverting the BioHashing transform.

    Attributes:
        ecg_correlation: Pearson ``r`` between the recovered ECG descriptor
            block and the original ECG descriptors.
        ppg_correlation: Pearson ``r`` between the recovered PPG descriptor
            block and the original PPG descriptors.
        max_feature_correlation: Pearson ``r`` over the full recovered vs.
            original feature vector — the overall leakage.
        feature_recovery_energy_ratio: ``‖x̂‖² / ‖x‖²`` — the fraction of the
            feature energy captured by the projection's row space.
        subspace_ratio: ``m / d`` — the theoretical upper bound on the
            recoverable fraction (the irreversibility budget).
        ppg_linear_correlation: Controlled baseline — the PPG-block leakage *if*
            it used plain BioHashing (a min-norm linear pre-image) at the ECG
            projection ratio, instead of IoM. The IoM win is
            ``ppg_correlation`` vs this. ``NaN`` for the all-linear analysis.
    """

    ecg_correlation: float
    ppg_correlation: float
    max_feature_correlation: float
    feature_recovery_energy_ratio: float
    subspace_ratio: float
    ppg_linear_correlation: float = float("nan")


def _safe_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return ``pearsonr(a, b)``, mapping the degenerate (constant) case to 0.

    Args:
        a: First 1-D vector.
        b: Second 1-D vector of equal length.

    Returns:
        Pearson correlation, or ``0.0`` when either vector is constant.
    """
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(0.0 if np.isnan(r) else r)


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``v`` scaled to unit L2 norm (or unchanged if it is all zero)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def recover_ppg_iom(
    ppg_features: NDArray[np.float64],
    token: str,
    n_hashes: int,
    window: int = IOM_WINDOW,
) -> NDArray[np.float64]:
    """Best-effort adversarial reconstruction of the PPG block from its IoM code.

    The adversary holds the token, so it can reproduce every Gaussian matrix and
    read off which projection won each hash. Each winning row is the direction
    along which ``ppg_features`` had the largest response in that hash's window,
    so the maximally informative *linear* estimate available is the sum of the
    winning direction vectors. Unlike a plain projection there is no exact
    pre-image — the argmax discards the magnitudes — so this is only a coarse
    direction guess, which is exactly what makes IoM non-invertible.

    Args:
        ppg_features: 1-D original PPG descriptor block (to read the winners).
        token: PPG sub-transform token (already salted by the caller).
        n_hashes: IoM code length ``m``.
        window: Projections per hash ``q``.

    Returns:
        1-D best-effort estimate ``x̂_ppg`` of the PPG block.
    """
    d = int(ppg_features.shape[0])
    w = iom.derive_iom(token, d, n_hashes, window)                 # (m, q, d)
    idx = iom.iom_indices(ppg_features, token, n_hashes, window)   # (m,)
    winners = w[np.arange(n_hashes), idx]                          # (m, d)
    return winners.sum(axis=0)


def multimodal_leakage_metrics(
    ecg_features: NDArray[np.float64],
    ppg_features: NDArray[np.float64],
    token: str,
    projection_ratio: float = DEFAULT_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
) -> InversionReport:
    """Leakage of the hybrid template: linear ECG inverse vs IoM PPG best-effort.

    Mirrors :func:`mwf.feature_transform.transform_multimodal`: the ECG block is
    inverted with the min-norm pre-image ``Rᵀy`` (the same √(m/d) leakage as
    plain BioHashing), while the PPG block is attacked with
    :func:`recover_ppg_iom`. The PPG correlation is the figure of merit — it
    should collapse relative to the ~0.78 of the all-linear transform.

    Args:
        ecg_features: 1-D original ECG descriptor block.
        ppg_features: 1-D original PPG descriptor block.
        token: User token (salted per modality internally, matching the forward
            transform).
        projection_ratio: ECG BioHashing ``m/d_ecg``.
        binarise: Whether the ECG block was sign-binarised.
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.

    Returns:
        An :class:`InversionReport`. ``feature_recovery_energy_ratio`` and
        ``subspace_ratio`` describe the *linear ECG* block (where they are
        meaningful); ``ppg_correlation`` quantifies the IoM leakage.

    Raises:
        ValueError: If the inputs are not 1-D.
    """
    if ecg_features.ndim != 1 or ppg_features.ndim != 1:
        raise ValueError("ecg_features and ppg_features must be 1-D arrays.")

    ecg = ecg_features.astype(np.float64, copy=False)
    ppg = ppg_features.astype(np.float64, copy=False)
    d_ecg = int(ecg.shape[0])

    # ECG block: orthonormal BioHashing → min-norm linear pre-image.
    m_ecg = projection_dim(d_ecg, projection_ratio)
    r_ecg = derive_projection(token + ECG_SALT, d_ecg, m_ecg).matrix
    ecg_template = _apply(r_ecg, ecg, binarise)
    rec_ecg = r_ecg.T @ ecg_template

    # PPG block: IoM → best-effort direction estimate (no exact pre-image).
    d_ppg = int(ppg.shape[0])
    n_hashes = iom.hash_count(d_ppg, n_hashes_ratio)
    rec_ppg = recover_ppg_iom(ppg, token + PPG_SALT, n_hashes, window)

    # Controlled baseline: what the PPG block would leak under plain (real-valued)
    # BioHashing at the same ratio — the √(m/d) min-norm linear pre-image.
    m_ppg_lin = projection_dim(d_ppg, projection_ratio)
    r_ppg_lin = derive_projection(token + "::PPG-LINBASE", d_ppg, m_ppg_lin).matrix
    rec_ppg_lin = r_ppg_lin.T @ (r_ppg_lin @ ppg)
    ppg_linear_corr = _safe_corr(rec_ppg_lin, ppg)

    # Balanced overall leakage: unit-normalise each block so neither dominates
    # the joint Pearson (the two recovered blocks live on different scales).
    rec_all = np.concatenate([_unit(rec_ecg), _unit(rec_ppg)])
    x_all = np.concatenate([_unit(ecg), _unit(ppg)])

    ecg_energy = float(np.dot(ecg, ecg))
    rec_ecg_energy = float(np.dot(rec_ecg, rec_ecg))
    energy_ratio = rec_ecg_energy / ecg_energy if ecg_energy > 0 else float("nan")

    return InversionReport(
        ecg_correlation=_safe_corr(rec_ecg, ecg),
        ppg_correlation=_safe_corr(rec_ppg, ppg),
        max_feature_correlation=_safe_corr(rec_all, x_all),
        feature_recovery_energy_ratio=energy_ratio,
        subspace_ratio=float(m_ecg / d_ecg),
        ppg_linear_correlation=ppg_linear_corr,
    )


__all__ = [
    "InversionReport",
    "multimodal_leakage_metrics",
    "recover_ppg_iom",
]

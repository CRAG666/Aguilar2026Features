"""Cancelability evaluation: renewability, diversity, unlinkability."""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, zscore

from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
    PipelineConfig,
)
from .dataset import BiometricSegments
from .feature_transform import transform_multimodal_batch
from .features import extract_features_batch
from .rng import make_rng
from .scoring import cosine_score_matrix, genuine_impostor_scores
from .stats_helpers import std_or_zero

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_LEVEL: Final[int] = DWT_DEFAULT_LEVEL
DEFAULT_FEATURE_WAVELET: Final[str] = FEATURE_WAVELET
DEFAULT_RATIO: Final[float] = DEFAULT_PROJECTION_RATIO
DEFAULT_N_KEYS: Final[int] = 32  # wide enough for a tight D_↔^sys CI
DEFAULT_RANDOM_STATE: Final[int] = DEFAULT_SEED
_RANDOM_TOKEN_PREFIX: Final[str] = "RAND_TOK_"
_SCORE_BINS: Final[int] = 200


def _random_tokens(n_keys: int, seed: int) -> list[str]:
    """Generate deterministic SHA-256-tagged random tokens.

    Args:
        n_keys: Number of tokens to generate.
        seed: Master RNG seed.

    Returns:
        List of ``n_keys`` prefixed token strings.
    """
    rng = make_rng(seed)
    raw = rng.integers(0, 1 << 63, size=n_keys, dtype=np.uint64)
    return [
        f"{_RANDOM_TOKEN_PREFIX}{hashlib.sha256(int(x).to_bytes(8, 'big')).hexdigest()}"
        for x in raw
    ]


def _templates_for_token(
    features: NDArray[np.float64],
    token: str,
    projection_ratio: float,
    binarise: bool,
) -> NDArray[np.float64]:
    """Project every feature row under a shared token and return templates.

    Args:
        features: ``(n, d)`` extracted feature matrix (token-independent).
        token: Shared token applied to every row.
        projection_ratio: BioHashing template length ``m / d``.
        binarise: Whether to sign-binarise the projection.

    Returns:
        ``(n, m)`` template matrix.
    """
    tokens = [token] * features.shape[0]
    return transform_multimodal_batch(
        features, tokens, projection_ratio=projection_ratio, binarise=binarise,
    )


@dataclass(frozen=True, slots=True)
class UnlinkabilityCurve:
    """Score-resolved Gomez-Barrero unlinkability curve.

    Attributes:
        thresholds: Score bin centres.
        d_local: Per-bin local unlinkability in ``[0, 1]``.
        d_sys: Global system-level unlinkability (``0`` = unlinkable).
    """

    thresholds: NDArray[np.float64]
    d_local: NDArray[np.float64]
    d_sys: float


def _d_sys_curve(mated: NDArray[np.float64], non_mated: NDArray[np.float64]) -> UnlinkabilityCurve:
    """Compute the Gomez-Barrero unlinkability curve.

    Args:
        mated: Mated (same-subject, different-key) scores.
        non_mated: Non-mated (different-subject) scores.

    Returns:
        A populated :class:`UnlinkabilityCurve`; ``d_sys`` is NaN when either
        score pool is empty (unlinkability is undefined without both).
    """
    if mated.size == 0 or non_mated.size == 0:
        # Unlinkability needs both mated (same-subject, cross-key) and
        # non-mated (cross-subject) scores. A single-subject cohort has no
        # cross-subject pairs, so the measure is undefined — return NaN
        # rather than 0.0, which would falsely read as "perfectly unlinkable".
        return UnlinkabilityCurve(
            thresholds=np.empty(0, dtype=np.float64),
            d_local=np.empty(0, dtype=np.float64),
            d_sys=float("nan"),
        )
    lo = float(min(mated.min(), non_mated.min()))
    hi = float(max(mated.max(), non_mated.max()))
    if hi <= lo:
        return UnlinkabilityCurve(
            thresholds=np.array([lo]),
            d_local=np.zeros(1),
            d_sys=0.0,
        )
    edges = np.linspace(lo, hi, _SCORE_BINS + 1)
    bin_width = float(edges[1] - edges[0])
    p_mated, _ = np.histogram(mated, bins=edges, density=True)
    p_non, _ = np.histogram(non_mated, bins=edges, density=True)
    denom = p_mated + p_non
    # Gomez-Barrero et al. (2018) local measure D_↔(s): one-sided, because only
    # score regions where the mated distribution dominates the non-mated one
    # reveal linkability.
    d_local = np.divide(
        np.maximum(0.0, p_mated - p_non),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    # Global D_↔^sys = ∫ p(s | mated) · D_↔(s) ds — the local linkability
    # averaged over the mated score distribution, not the worst-case bin.
    d_sys = float(np.sum(p_mated * d_local) * bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return UnlinkabilityCurve(
        thresholds=centres,
        d_local=d_local,
        d_sys=d_sys,
    )


@dataclass(frozen=True, slots=True)
class CancelabilityReport:
    """ISO/IEC 30136 cancelability summary.

    Attributes:
        n_keys: Number of random keys exercised.
        renewability_genuine_mean: Mean cross-key genuine-pair score.
        renewability_baseline_mean: Mean same-key genuine-pair score.
        renewability_ratio: ``genuine_mean / baseline_mean``.
        diversity_mean_abs_corr: Mean cross-key per-subject correlation.
        diversity_std_abs_corr: Std of the same.
        unlinkability_d_sys: Global Gomez-Barrero ``D_↔^sys``.
    """

    n_keys: int
    renewability_genuine_mean: float
    renewability_baseline_mean: float
    renewability_ratio: float
    diversity_mean_abs_corr: float
    diversity_std_abs_corr: float
    unlinkability_d_sys: float

    @property
    def renewable(self) -> bool:
        """``True`` when ``renewability_ratio < 0.05``."""
        return self.renewability_ratio < 0.05

    @property
    def diverse(self) -> bool:
        """``True`` when ``diversity_mean_abs_corr < 0.01``."""
        return self.diversity_mean_abs_corr < 0.01

    @property
    def unlinkable(self) -> bool:
        """``True`` when ``unlinkability_d_sys < 0.05``."""
        return self.unlinkability_d_sys < 0.05


def _same_key_genuine_mean(
    templates: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    """Mean same-subject cosine score under one key, excluding self-matches.

    The renewability baseline answers "how similar are two enrolments of the
    same subject under the *same* key?". The diagonal of the all-vs-all score
    matrix is a template compared with itself (cosine ``= 1``); those trivial
    self-matches would inflate the baseline and make the renewability ratio
    look artificially small, so they are dropped.

    Args:
        templates: ``(n, d)`` template matrix under a single key.
        labels: ``(n,)`` subject identifiers.

    Returns:
        Mean off-diagonal same-subject cosine similarity.

    Raises:
        ValueError: If no subject has ≥ 2 segments (empty genuine pool).
    """
    sim = cosine_score_matrix(templates, templates)
    same_subject = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_subject, False)
    genuine = sim[same_subject]
    if genuine.size == 0:
        raise ValueError(
            "Same-key genuine baseline is empty: renewability needs ≥ 2 "
            "segments for at least one subject."
        )
    return float(genuine.mean())


def _abs_pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Return ``|pearsonr(x, y)|``, or NaN when either vector is constant.

    Args:
        x: First vector.
        y: Second vector of equal length.

    Returns:
        Absolute Pearson correlation, or NaN if either input has zero variance.
    """
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(abs(r))


def _standardize_columns(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column-wise z-score a template matrix (zero-variance columns left at 0).

    Diversity asks whether two *keys* produce decorrelated templates. Computed
    on raw features, the correlation is dominated by structure shared across
    every key, so even a perfectly key-randomising transform reads as highly
    correlated. Removing each feature's cohort mean and scale isolates the
    key-dependent variation, which is what ISO/IEC 30136 diversity captures.

    Args:
        features: ``(n, d)`` template matrix.

    Returns:
        ``(n, d)`` matrix with each column centred and scaled to unit variance;
        columns with zero variance are returned as zeros.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        standardized = zscore(features, axis=0, ddof=0)
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_cancelability(
    segments: BiometricSegments,
    feature_level: int = DEFAULT_FEATURE_LEVEL,
    feature_wavelet: str = DEFAULT_FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    n_keys: int = DEFAULT_N_KEYS,
    seed: int = DEFAULT_RANDOM_STATE,
    return_curve: bool = False,
    config: PipelineConfig | None = None,
) -> CancelabilityReport | tuple[CancelabilityReport, UnlinkabilityCurve]:
    """Run the cancelability protocol on ``segments``.

    Features are extracted once (they do not depend on the token); each random
    key then re-projects that same feature matrix, so renewability and
    diversity isolate the effect of the token alone.

    Args:
        segments: Cohort to evaluate.
        feature_level: DWT depth of the feature extractor.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing template length ``m / d``.
        binarise: Whether to sign-binarise the projection.
        n_keys: Number of random keys (must be ≥ 2).
        seed: Master RNG seed.
        return_curve: If ``True``, also return the :class:`UnlinkabilityCurve`.
        config: Optional :class:`PipelineConfig` overriding the per-knob args.

    Returns:
        A :class:`CancelabilityReport`, or ``(report, curve)`` when
        ``return_curve`` is true.

    Raises:
        ValueError: If ``n_keys < 2``.
    """
    if config is not None:
        feature_level = config.feature_level
        feature_wavelet = config.feature_wavelet
        projection_ratio = config.projection_ratio
        binarise = config.binarise

    features = extract_features_batch(
        segments.ecg, segments.ppg, wavelet=feature_wavelet, level=feature_level,
    )
    tokens = _random_tokens(n_keys, seed=seed)
    base = _templates_for_token(features, tokens[0], projection_ratio, binarise)
    baseline_mean = _same_key_genuine_mean(base, segments.labels)

    renew_means: list[float] = []
    diversity_corrs: list[float] = []
    mated_pool: list[NDArray[np.float64]] = []
    non_mated_pool: list[NDArray[np.float64]] = []

    for token in tokens[1:]:
        reissued = _templates_for_token(features, token, projection_ratio, binarise)
        mated, non_mated = genuine_impostor_scores(base, reissued, segments.labels)
        renew_means.append(float(mated.mean()))
        mated_pool.append(mated)
        non_mated_pool.append(non_mated)
        base_z = _standardize_columns(base)
        reissued_z = _standardize_columns(reissued)
        for cls in np.unique(segments.labels):
            mask = segments.labels == cls
            if mask.sum() == 0:
                continue
            corr = _abs_pearson(base_z[mask].ravel(), reissued_z[mask].ravel())
            if not np.isnan(corr):
                diversity_corrs.append(corr)

    if not renew_means:
        raise ValueError("n_keys must be ≥ 2 to evaluate cancelability.")

    diversity_arr = np.asarray(diversity_corrs, dtype=np.float64)
    curve = _d_sys_curve(np.concatenate(mated_pool), np.concatenate(non_mated_pool))
    renew_mean = float(np.mean(renew_means))
    report = CancelabilityReport(
        n_keys=n_keys,
        renewability_genuine_mean=renew_mean,
        renewability_baseline_mean=baseline_mean,
        renewability_ratio=(renew_mean / baseline_mean if baseline_mean else float("nan")),
        diversity_mean_abs_corr=float(diversity_arr.mean()),
        diversity_std_abs_corr=std_or_zero(diversity_arr),
        unlinkability_d_sys=curve.d_sys,
    )
    return (report, curve) if return_curve else report


__all__ = [
    "CancelabilityReport",
    "UnlinkabilityCurve",
    "evaluate_cancelability",
]

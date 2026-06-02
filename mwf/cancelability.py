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
_BOOTSTRAP_RESAMPLES: Final[int] = 500  # key-level resamples for the CIs
_BOOTSTRAP_LEVEL: Final[float] = 0.95
# A renewable transform should decorrelate templates down to the chance level of
# two independent vectors, not to an arbitrary absolute constant. The pass/fail
# margin below is a multiple of that analytical chance |r| (≈ √(2/(π(L−1))) for
# length-L i.i.d. vectors), so the thresholds are referenced to a null baseline
# rather than to magic numbers.
_DIVERSITY_CHANCE_MARGIN: Final[float] = 1.5
_RENEWABILITY_MAX_RATIO: Final[float] = 0.05
_UNLINKABILITY_MAX_DSYS: Final[float] = 0.05


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
    """ISO/IEC 30136 cancelability summary with key-level bootstrap CIs.

    Every figure of merit carries a 95 % percentile-bootstrap CI resampled over
    the random keys, so the report states uncertainty rather than a single point
    estimate. The pass/fail flags are referenced to statistical baselines (a
    chance |r| for diversity) instead of bare constants.

    Attributes:
        n_keys: Number of random keys exercised.
        renewability_genuine_mean: Mean cross-key genuine-pair score.
        renewability_baseline_mean: Mean same-key genuine-pair score.
        renewability_ratio: ``genuine_mean / baseline_mean``.
        renewability_ratio_ci_low: Lower CI bound on the ratio.
        renewability_ratio_ci_high: Upper CI bound on the ratio.
        diversity_mean_abs_corr: Mean cross-key per-subject correlation.
        diversity_std_abs_corr: Std of the same.
        diversity_ci_low: Lower CI bound on the mean ``|corr|``.
        diversity_ci_high: Upper CI bound on the mean ``|corr|``.
        diversity_chance_abs_corr: Analytical chance-level ``|corr|`` of two
            independent vectors of the template's length — the null baseline the
            measured diversity is compared against.
        unlinkability_d_sys: Global Gomez-Barrero ``D_↔^sys``.
        unlinkability_d_sys_ci_low: Lower CI bound on ``D_↔^sys``.
        unlinkability_d_sys_ci_high: Upper CI bound on ``D_↔^sys``.
    """

    n_keys: int
    renewability_genuine_mean: float
    renewability_baseline_mean: float
    renewability_ratio: float
    renewability_ratio_ci_low: float
    renewability_ratio_ci_high: float
    diversity_mean_abs_corr: float
    diversity_std_abs_corr: float
    diversity_ci_low: float
    diversity_ci_high: float
    diversity_chance_abs_corr: float
    unlinkability_d_sys: float
    unlinkability_d_sys_ci_low: float
    unlinkability_d_sys_ci_high: float

    @property
    def renewable(self) -> bool:
        """``True`` when the cross-key genuine score is ≤ 5 % of the baseline."""
        return self.renewability_ratio < _RENEWABILITY_MAX_RATIO

    @property
    def diverse(self) -> bool:
        """``True`` when mean ``|corr|`` stays within the chance-level margin.

        Compared to ``_DIVERSITY_CHANCE_MARGIN × diversity_chance_abs_corr`` — the
        decorrelation a perfectly key-randomising transform can be expected to
        reach — rather than to an absolute constant.
        """
        return self.diversity_mean_abs_corr < _DIVERSITY_CHANCE_MARGIN * self.diversity_chance_abs_corr

    @property
    def unlinkable(self) -> bool:
        """``True`` when ``unlinkability_d_sys < 0.05`` (Gomez-Barrero)."""
        return self.unlinkability_d_sys < _UNLINKABILITY_MAX_DSYS


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


def _diversity_chance_abs_corr(template_dim: int, segs_per_subject: float) -> float:
    """Analytical chance-level ``|corr|`` for the diversity null baseline.

    The diversity correlation is taken over a flattened ``(segs × template_dim)``
    block. Under independence the sample Pearson ``r`` has std ``1/√(L−1)`` and a
    folded-normal mean ``√(2/π)·std``, so the chance ``|r| ≈ √(2/(π(L−1)))`` for
    block length ``L``. This is the level a perfectly key-randomising transform
    would reach; the measured diversity is judged against it.
    """
    length = max(2.0, float(segs_per_subject) * float(template_dim))
    return float(np.sqrt(2.0 / (np.pi * (length - 1.0))))


def _percentile_ci(samples: NDArray[np.float64], level: float) -> tuple[float, float]:
    """Central ``level`` percentile band of ``samples`` (NaNs dropped)."""
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))


def _bootstrap_d_sys_ci(
    mated_per_key: list[NDArray[np.float64]],
    non_mated_per_key: list[NDArray[np.float64]],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI for ``D_↔^sys`` resampling the keys with replacement."""
    n = len(mated_per_key)
    if n < 2:
        return float("nan"), float("nan")
    vals = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for b in range(_BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, n, size=n)
        mated = np.concatenate([mated_per_key[k] for k in pick])
        non_mated = np.concatenate([non_mated_per_key[k] for k in pick])
        vals[b] = _d_sys_curve(mated, non_mated).d_sys
    return _percentile_ci(vals, _BOOTSTRAP_LEVEL)


def _bootstrap_key_mean_ci(
    per_key_values: list[NDArray[np.float64]],
    rng: np.random.Generator,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Percentile CI for the pooled mean of per-key value arrays.

    Resamples keys with replacement (the unit of independence), pools the chosen
    keys' values and takes their mean, optionally divided by ``scale`` (used for
    the renewability *ratio*, whose denominator is the fixed same-key baseline).
    """
    n = len(per_key_values)
    if n < 2 or scale == 0.0:
        return float("nan"), float("nan")
    vals = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for b in range(_BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, n, size=n)
        pooled = np.concatenate([per_key_values[k] for k in pick])
        pooled = pooled[np.isfinite(pooled)]
        vals[b] = float(pooled.mean()) / scale if pooled.size else np.nan
    return _percentile_ci(vals, _BOOTSTRAP_LEVEL)


def _per_class_abs_corr(
    a_z: NDArray[np.float64],
    b_z: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Per-class ``|Pearson r|`` between two column-standardised template sets.

    The diversity metric of one re-issued key: for every subject, how correlated
    its base and re-issued templates remain (lower ⇒ more diverse). Degenerate
    (NaN) correlations are dropped.
    """
    corrs: list[float] = []
    for cls in np.unique(labels):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        r = _abs_pearson(a_z[mask].ravel(), b_z[mask].ravel())
        if not np.isnan(r):
            corrs.append(r)
    return np.asarray(corrs, dtype=np.float64)


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
    base_z = _standardize_columns(base)

    renew_per_key: list[NDArray[np.float64]] = []
    diversity_per_key: list[NDArray[np.float64]] = []
    mated_pool: list[NDArray[np.float64]] = []
    non_mated_pool: list[NDArray[np.float64]] = []

    for token in tokens[1:]:
        reissued = _templates_for_token(features, token, projection_ratio, binarise)
        mated, non_mated = genuine_impostor_scores(base, reissued, segments.labels)
        renew_per_key.append(mated)
        mated_pool.append(mated)
        non_mated_pool.append(non_mated)
        reissued_z = _standardize_columns(reissued)
        diversity_per_key.append(
            _per_class_abs_corr(base_z, reissued_z, segments.labels)
        )

    if not renew_per_key:
        raise ValueError("n_keys must be ≥ 2 to evaluate cancelability.")

    boot_rng = make_rng(seed + 101)
    diversity_arr = np.concatenate(diversity_per_key) if diversity_per_key else np.empty(0)
    curve = _d_sys_curve(np.concatenate(mated_pool), np.concatenate(non_mated_pool))
    renew_mean = float(np.concatenate(renew_per_key).mean())
    renew_ratio = renew_mean / baseline_mean if baseline_mean else float("nan")

    renew_ci_lo, renew_ci_hi = _bootstrap_key_mean_ci(
        renew_per_key, boot_rng, scale=baseline_mean if baseline_mean else 1.0,
    )
    div_ci_lo, div_ci_hi = _bootstrap_key_mean_ci(diversity_per_key, boot_rng)
    dsys_ci_lo, dsys_ci_hi = _bootstrap_d_sys_ci(mated_pool, non_mated_pool, boot_rng)
    chance = _diversity_chance_abs_corr(
        template_dim=int(base.shape[1]),
        segs_per_subject=base.shape[0] / max(1, np.unique(segments.labels).size),
    )

    report = CancelabilityReport(
        n_keys=n_keys,
        renewability_genuine_mean=renew_mean,
        renewability_baseline_mean=baseline_mean,
        renewability_ratio=renew_ratio,
        renewability_ratio_ci_low=renew_ci_lo,
        renewability_ratio_ci_high=renew_ci_hi,
        diversity_mean_abs_corr=float(diversity_arr.mean()) if diversity_arr.size else float("nan"),
        diversity_std_abs_corr=std_or_zero(diversity_arr),
        diversity_ci_low=div_ci_lo,
        diversity_ci_high=div_ci_hi,
        diversity_chance_abs_corr=chance,
        unlinkability_d_sys=curve.d_sys,
        unlinkability_d_sys_ci_low=dsys_ci_lo,
        unlinkability_d_sys_ci_high=dsys_ci_hi,
    )
    return (report, curve) if return_curve else report


__all__ = [
    "CancelabilityReport",
    "UnlinkabilityCurve",
    "evaluate_cancelability",
]

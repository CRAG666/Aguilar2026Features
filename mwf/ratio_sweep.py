"""Sweep the BioHashing ratio ``m/d`` — recognition / leakage / SAR trade-off.

The projection ratio is the irreversibility budget: smaller ratios collapse
``x``'s ``(d − m)``-dimensional component into the null space (less leakage) but
also discard more discriminative information (worse EER). The operational
default of ``m/d = 0.5`` was chosen heuristically and is reported as a single
operating point in the headline tables — a reviewer asks how it behaves across
the range.

This module produces the trade-off curve: for each ratio it reports
(i) single-key (biometric-only) verification EER, (ii) stolen-token EER —
the worst-case operating figure, (iii) the mean inversion correlation of the
multimodal reconstruction (the leakage axis), and (iv) the protected verifier's
mated EER. Together they plot the recognition–leakage frontier and locate the
operating-point sweet spot.

Single-key EER (rather than per_subject) is used because single_key is the only
regime where the recognition margin is biometric-only — per_subject conflates
key uniqueness with biometric content (see :mod:`mwf.per_subject_ablation`), so
its EER cannot be read as a function of the projection ratio alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_info import get_eer_stats

from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
)
from .dataset import BiometricSegments
from .feature_transform import transform_multimodal_batch
from .features import extract_features, extract_features_batch, feature_dimension
from .inversion import multimodal_leakage_metrics
from .pipeline import SHARED_TOKEN, preprocess_signals
from .rng import make_rng
from .scoring import cosine_score_matrix, decidability
from .stolen_token import stolen_token_verification

logger = logging.getLogger(__name__)

DEFAULT_RATIOS: tuple[float, ...] = (0.25, 0.40, 0.50, 0.60, 0.75)


@dataclass(frozen=True, slots=True)
class RatioSweepPoint:
    """One point on the recognition–leakage trade-off curve.

    Attributes:
        ratio: BioHashing ratio ``m/d_ecg``.
        single_key_eer: EER under the single_key regime (shared token across
            the cohort) — the biometric-only recognition floor at this ratio.
        single_key_decidability: Daugman's ``d'`` for single_key.
        stolen_token_eer: Worst-case EER when the adversary holds each
            victim's token (:func:`mwf.stolen_token_verification`).
        stolen_token_decidability: Daugman's ``d'`` for the stolen-token
            scenario.
        inversion_mean_correlation: Mean absolute reconstruction-vs-original
            correlation of the multimodal min-norm pre-image — the leakage
            axis. Lower is better.
        inversion_std_correlation: Std of the per-segment inversion
            correlation sample.
        n_inversion_segments: Number of probed segments backing the inversion
            statistics.
    """

    ratio: float
    single_key_eer: float
    single_key_decidability: float
    stolen_token_eer: float
    stolen_token_decidability: float
    inversion_mean_correlation: float
    inversion_std_correlation: float
    n_inversion_segments: int


def _single_key_eer(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    ratio: float,
    binarise: bool,
) -> tuple[float, float]:
    """Single-key EER (biometric-only) at one projection ratio."""
    tokens = [SHARED_TOKEN] * labels.size
    templates = transform_multimodal_batch(
        features, tokens, projection_ratio=ratio, binarise=binarise,
    )
    sim = cosine_score_matrix(templates, templates)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    genuine = sim[same]
    impostor = sim[~same]
    if genuine.size < 2 or impostor.size < 2:
        return float("nan"), float("nan")
    stats = get_eer_stats(genuine, impostor)
    return float(stats.eer), float(decidability(genuine, impostor))


def _inversion_summary(
    segments: BiometricSegments,
    *,
    feature_level: int,
    ratio: float,
    n_segments: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Mean / std / count of the per-segment multimodal inversion correlation."""
    idx = rng.choice(
        segments.num_segments,
        size=min(n_segments, segments.num_segments),
        replace=False,
    )
    half = feature_dimension(feature_level) // 2
    corrs: list[float] = []
    for i in idx:
        x = extract_features(segments.ecg[i], segments.ppg[i], level=feature_level)
        report = multimodal_leakage_metrics(
            x[:half], x[half:], f"USER_RAT_{i}", projection_ratio=ratio,
        )
        corrs.append(abs(report.max_feature_correlation))
    arr = np.asarray(corrs, dtype=np.float64)
    return (
        float(arr.mean()) if arr.size else float("nan"),
        float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        int(arr.size),
    )


def ratio_sweep(
    segments: BiometricSegments,
    *,
    feature_level: int = DWT_DEFAULT_LEVEL,
    feature_wavelet: str = FEATURE_WAVELET,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    binarise: bool = DEFAULT_BINARISE,
    max_victims: int | None = 40,
    n_inversion_segments: int = 16,
    seed: int = DEFAULT_SEED,
    denoise: bool = True,
) -> list[RatioSweepPoint]:
    """Compute recognition/leakage figures across ``ratios`` at a fixed feature level.

    Args:
        segments: ECG/PPG cohort.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        ratios: BioHashing ratios to sweep (each in ``(0, 1]``).
        binarise: Whether to sign-binarise the BioHashing projection.
        max_victims: Cap on stolen-token victims per ratio (``None`` = all).
        n_inversion_segments: Segments probed per ratio for the inversion
            correlation summary.
        seed: Master RNG seed.
        denoise: Whether to run NeuroKit cleaning before feature extraction.

    Returns:
        One :class:`RatioSweepPoint` per ratio, in the input order.
    """
    ecg, ppg = preprocess_signals(
        segments.ecg, segments.ppg, sampling_rate=segments.sampling_rate,
        snr_db=None, denoise=denoise,
    )
    features = extract_features_batch(
        ecg, ppg, wavelet=feature_wavelet, level=feature_level,
    )
    rng = make_rng(seed)
    points: list[RatioSweepPoint] = []
    for ratio in ratios:
        sk_eer, sk_deci = _single_key_eer(features, segments.labels, ratio, binarise)
        stolen = stolen_token_verification(
            segments,
            feature_level=feature_level,
            feature_wavelet=feature_wavelet,
            projection_ratio=ratio,
            binarise=binarise,
            max_victims=max_victims,
            seed=seed,
            denoise=denoise,
        )
        inv_mean, inv_std, inv_n = _inversion_summary(
            segments, feature_level=feature_level, ratio=ratio,
            n_segments=n_inversion_segments, rng=rng,
        )
        point = RatioSweepPoint(
            ratio=float(ratio),
            single_key_eer=sk_eer,
            single_key_decidability=sk_deci,
            stolen_token_eer=float(stolen.eer),
            stolen_token_decidability=float(stolen.decidability),
            inversion_mean_correlation=inv_mean,
            inversion_std_correlation=inv_std,
            n_inversion_segments=inv_n,
        )
        logger.info(
            "[ratio sweep | ratio=%.2f] single_key EER=%.4f stolen EER=%.4f "
            "inversion=%.3f (n=%d)",
            point.ratio, point.single_key_eer, point.stolen_token_eer,
            point.inversion_mean_correlation, point.n_inversion_segments,
        )
        points.append(point)
    return points


__all__ = [
    "DEFAULT_RATIOS",
    "RatioSweepPoint",
    "ratio_sweep",
]

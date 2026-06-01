"""Stolen-token (lost-key) verification: the honest biometric figure of merit.

Under the standard cancelable-biometric threat model the attacker is assumed to
hold the victim's token — the cancelable transform protects the stored
*template*, not the secret. This module re-projects the whole cohort under each
victim's token, so the per-subject key is neutralised as a discriminative cue.
Any remaining separation between genuine and impostor scores is then *pure
biometric*.

This is the metric that must stay low (EER) / high (decidability) for the system
to be simultaneously secure and a strong recogniser. It deliberately strips out
the deterministic per-key signature that inflates the naive ``per_subject``
identification numbers: there, impostors carry their *own* token, so a classifier
can separate subjects by key rather than by physiology. Here every comparison
shares the claimed identity's token, so only physiology can separate them.

Because the BioHashing transform acts on the feature vector — which does not
depend on the token — features are extracted once and only the projection is
repeated per victim, unlike the signal-domain sibling that re-transforms the raw
waveforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_info import get_eer_stats

from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_ECG_CLEAN_METHOD,
    DEFAULT_PPG_CLEAN_METHOD,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
    PipelineConfig,
)
from .dataset import BiometricSegments
from .feature_transform import transform_multimodal_batch
from .features import extract_features_batch
from .operating_curves import bootstrap_eer_ci, operating_points
from .pipeline import _token_for_label, preprocess_signals
from .rng import make_rng
from .scoring import compute_subject_centroids, cosine_score_matrix, l2_normalise, znorm

logger = logging.getLogger(__name__)

DEFAULT_ENROL_FRACTION: Final[float] = 0.5
_MIN_VICTIM_SEGMENTS: Final[int] = 2  # need ≥1 enrol + ≥1 query
SCORE_NORMS: Final[tuple[str | None, ...]] = (None, "znorm")


@dataclass(frozen=True, slots=True)
class StolenTokenResult:
    """Worst-case (stolen-key) verification summary.

    Attributes:
        n_victims: Enrolled subjects actually exercised as verification targets
            (subjects with ≥ 2 segments, capped by ``max_victims``).
        n_genuine: Genuine score count (victim queries vs own enrolment).
        n_impostor: Impostor score count (other subjects under the victim token).
        eer: Equal-error rate with the key neutralised — pure-biometric EER.
        eer_ci_low: Lower 95 % percentile-bootstrap bound on the EER.
        eer_ci_high: Upper 95 % percentile-bootstrap bound on the EER.
        decidability: Daugman's ``d'`` of the pooled scores.
        genuine_mean: Mean genuine score.
        impostor_mean: Mean impostor score.
        operating_points: FNMR@FMR / FMR@FNMR dictionary.
    """

    n_victims: int
    n_genuine: int
    n_impostor: int
    eer: float
    eer_ci_low: float
    eer_ci_high: float
    decidability: float
    genuine_mean: float
    impostor_mean: float
    operating_points: dict[str, float]


def _split_impostor_cohort(
    other_idx: NDArray[np.int64],
    labels: NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split impostor segment indices into a Z-norm cohort and a scored set.

    The split is by *subject* (never by segment) so no subject straddles the two
    halves — the cohort that estimates the normalisation statistics shares no
    identity with the impostors those statistics are applied to.

    Args:
        other_idx: Segment indices of all non-victim (impostor) segments.
        labels: Full label array.
        rng: Seeded generator for the deterministic subject split.

    Returns:
        Tuple ``(cohort_idx, scored_idx)``. With only one impostor subject the
        split is impossible, so both fall back to ``other_idx`` (the degenerate
        single-subject case; immaterial on a real cohort).
    """
    other_subjects = np.unique(labels[other_idx])
    if other_subjects.size < 2:
        return other_idx, other_idx
    perm = rng.permutation(other_subjects.size)
    n_cohort = max(1, other_subjects.size // 2)
    cohort_subjects = other_subjects[perm[:n_cohort]]
    in_cohort = np.isin(labels[other_idx], cohort_subjects)
    return other_idx[in_cohort], other_idx[~in_cohort]


def stolen_token_score_pools(
    segments: BiometricSegments,
    feature_level: int = DWT_DEFAULT_LEVEL,
    feature_wavelet: str = FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    snr_db: float | None = None,
    noise_seed: int = 0,
    denoise: bool = True,
    ecg_method: str = DEFAULT_ECG_CLEAN_METHOD,
    ppg_method: str = DEFAULT_PPG_CLEAN_METHOD,
    max_victims: int | None = None,
    enrol_fraction: float = DEFAULT_ENROL_FRACTION,
    seed: int = DEFAULT_SEED,
    score_norm: str | None = None,
    config: PipelineConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool genuine/impostor scores under the stolen-key threat model.

    For each victim subject ``S`` the entire cohort's features are re-projected
    under ``S``'s token. ``S``'s own segments are split into an enrolment
    centroid and genuine queries; every other subject's segment (also under
    ``S``'s token) becomes an impostor query against that centroid.

    Args:
        segments: ECG/PPG cohort with per-subject labels.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing template length ``m / d``.
        binarise: Whether to sign-binarise the BioHashing projection.
        snr_db: AWGN SNR injected pre-clean (``None`` to skip).
        noise_seed: Seed for AWGN injection.
        denoise: Whether to run NeuroKit physiological cleaning.
        ecg_method: Method forwarded to ``neurokit2.ecg_clean``.
        ppg_method: Method forwarded to ``neurokit2.ppg_clean``.
        max_victims: Cap on the number of victim subjects (``None`` = all).
        enrol_fraction: Fraction of a victim's segments used for enrolment.
        seed: Master RNG seed for victim subsampling and enrol/query split.
        score_norm: ``None`` for raw cosine, or ``"znorm"`` to z-normalise each
            victim's genuine and impostor scores against a **held-out** impostor
            cohort (disjoint impostor subjects) before pooling. The disjoint
            cohort is what makes z-norm honest — normalising against the scored
            impostors themselves would deflate the EER by construction.
        config: Optional :class:`PipelineConfig` overriding feature/projection knobs.

    Returns:
        Tuple ``(genuine_scores, impostor_scores)`` pooled across victims.

    Raises:
        ValueError: If no subject has ≥ 2 segments (no genuine pair possible),
            or ``score_norm`` is not a recognised option.
    """
    if score_norm not in SCORE_NORMS:
        raise ValueError(f"Unknown score_norm {score_norm!r}. Expected one of {SCORE_NORMS}.")
    if config is not None:
        feature_level = config.feature_level
        feature_wavelet = config.feature_wavelet
        projection_ratio = config.projection_ratio
        binarise = config.binarise

    ecg, ppg = preprocess_signals(
        segments.ecg,
        segments.ppg,
        sampling_rate=segments.sampling_rate,
        snr_db=snr_db,
        noise_seed=noise_seed,
        denoise=denoise,
        ecg_method=ecg_method,
        ppg_method=ppg_method,
    )
    # Features are token-independent → extract once, re-project per victim.
    features = extract_features_batch(
        ecg, ppg, wavelet=feature_wavelet, level=feature_level
    )
    labels = segments.labels
    uniq = np.unique(labels)
    rng = make_rng(seed)
    if max_victims is not None and max_victims < uniq.size:
        uniq = np.sort(rng.choice(uniq, size=max_victims, replace=False))

    genuine_pool: list[NDArray[np.float64]] = []
    impostor_pool: list[NDArray[np.float64]] = []
    for victim in uniq:
        victim_mask = labels == victim
        victim_idx = np.flatnonzero(victim_mask)
        if victim_idx.size < _MIN_VICTIM_SEGMENTS:
            continue

        tokens = [_token_for_label(int(victim))] * labels.shape[0]
        templates = transform_multimodal_batch(
            features, tokens, projection_ratio=projection_ratio, binarise=binarise,
        )
        feats_n = l2_normalise(templates)

        perm = rng.permutation(victim_idx.size)
        n_enrol = int(round(enrol_fraction * victim_idx.size))
        n_enrol = min(max(n_enrol, 1), victim_idx.size - 1)
        enrol_idx = victim_idx[perm[:n_enrol]]
        query_idx = victim_idx[perm[n_enrol:]]

        centroid, _ = compute_subject_centroids(feats_n[enrol_idx], labels[enrol_idx])
        gen = cosine_score_matrix(feats_n[query_idx], centroid).ravel()
        other_idx = np.flatnonzero(~victim_mask)
        if score_norm == "znorm":
            # Z-norm statistics MUST come from an impostor cohort disjoint from
            # the scored impostors; estimating μ/σ from the very scores being
            # normalised (znorm(imp, imp)) forces the impostor distribution to
            # mean 0 / unit std by construction and optimistically deflates the
            # EER. Split the impostor *subjects* into a held-out cohort (μ/σ) and
            # a disjoint scored set, so the reported EER is honest.
            cohort_idx, scored_idx = _split_impostor_cohort(other_idx, labels, rng)
            cohort_scores = cosine_score_matrix(feats_n[cohort_idx], centroid).ravel()
            imp = cosine_score_matrix(feats_n[scored_idx], centroid).ravel()
            gen, imp = znorm(gen, cohort_scores), znorm(imp, cohort_scores)
        else:
            imp = cosine_score_matrix(feats_n[other_idx], centroid).ravel()
        genuine_pool.append(gen)
        impostor_pool.append(imp)

    if not genuine_pool:
        raise ValueError(
            "Stolen-token verification needs ≥ 2 segments for at least one subject."
        )
    return np.concatenate(genuine_pool), np.concatenate(impostor_pool)


def stolen_token_verification(
    segments: BiometricSegments,
    **kwargs: object,
) -> StolenTokenResult:
    """Summarise stolen-key verification into an EER / decidability report.

    Args:
        segments: ECG/PPG cohort with per-subject labels.
        **kwargs: Forwarded to :func:`stolen_token_score_pools`.

    Returns:
        A :class:`StolenTokenResult` over the pooled genuine/impostor scores.
    """
    genuine, impostor = stolen_token_score_pools(segments, **kwargs)  # type: ignore[arg-type]
    stats = get_eer_stats(genuine, impostor)
    # Count only subjects actually exercised (≥ 2 segments), capped by max_victims,
    # rather than every label — victims with a single segment are skipped in the
    # pooling loop and must not inflate the reported victim count.
    _, seg_counts = np.unique(segments.labels, return_counts=True)
    n_victims = int(np.count_nonzero(seg_counts >= _MIN_VICTIM_SEGMENTS))
    if kwargs.get("max_victims") is not None:
        n_victims = min(n_victims, int(kwargs["max_victims"]))  # type: ignore[arg-type]
    eer_seed = int(kwargs.get("seed", DEFAULT_SEED))  # type: ignore[arg-type]
    eer_ci_low, eer_ci_high = bootstrap_eer_ci(genuine, impostor, seed=eer_seed)
    result = StolenTokenResult(
        n_victims=n_victims,
        n_genuine=int(genuine.size),
        n_impostor=int(impostor.size),
        eer=float(stats.eer),
        eer_ci_low=eer_ci_low,
        eer_ci_high=eer_ci_high,
        decidability=float(stats.decidability),
        genuine_mean=float(stats.gmean),
        impostor_mean=float(stats.imean),
        operating_points=operating_points(genuine, impostor),
    )
    logger.info(
        "[stolen-token | %d victims | %d genuine | %d impostor] EER=%.4f d'=%.3f",
        result.n_victims,
        result.n_genuine,
        result.n_impostor,
        result.eer,
        result.decidability,
    )
    return result


__all__ = [
    "StolenTokenResult",
    "stolen_token_score_pools",
    "stolen_token_verification",
]

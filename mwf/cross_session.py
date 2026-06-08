"""Cross-session / cross-activity verification: enrol one condition, probe another.

Enrols each subject's cancelable template from one recording condition and
verifies probes from a different one, with the token neutralised as in
:mod:`mwf.stolen_token` so the residual genuine/impostor separation is pure
biometric. PTT-PPG records its activities in one visit, so this measures
cross-condition (physiological-state) robustness, not multi-day template ageing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray
from pyeer.eer_info import get_eer_stats

from .batch_utils import DEFAULT_BATCH_N_JOBS
from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_ECG_CLEAN_METHOD,
    DEFAULT_PPG_CLEAN_METHOD,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
)
from .dataset import BiometricSegments
from .feature_transform import FeatureScaler, transform_multimodal_batch
from .features import extract_features_batch
from .operating_curves import bootstrap_eer_ci, operating_points
from .pipeline import _token_for_label, preprocess_signals
from .rng import make_rng
from .scoring import compute_subject_centroids, cosine_score_matrix, l2_normalise, znorm

logger = logging.getLogger(__name__)

SCORE_NORMS: Final[tuple[str | None, ...]] = (None, "znorm")


@dataclass(frozen=True, slots=True)
class CrossSessionResult:
    """Cross-condition (enrol-A / probe-B) verification summary.

    Attributes:
        enrol_label: Name of the enrolment condition (e.g. ``"sit"``).
        probe_label: Name of the probe condition (e.g. ``"run"``).
        n_subjects: Subjects present in *both* conditions (the scored cohort).
        n_genuine: Genuine score count (probe-B vs own enrol-A centroid).
        n_impostor: Impostor score count (other subjects under the claimed token).
        eer: Cross-condition equal-error rate (key neutralised → pure biometric).
        eer_ci_low: Lower 95 % percentile-bootstrap bound on the EER.
        eer_ci_high: Upper 95 % percentile-bootstrap bound on the EER.
        decidability: Daugman's ``d'`` of the pooled scores.
        genuine_mean: Mean genuine score.
        impostor_mean: Mean impostor score.
        operating_points: FNMR@FMR / FMR@FNMR dictionary.
    """

    enrol_label: str
    probe_label: str
    n_subjects: int
    n_genuine: int
    n_impostor: int
    eer: float
    eer_ci_low: float
    eer_ci_high: float
    decidability: float
    genuine_mean: float
    impostor_mean: float
    operating_points: dict[str, float]


def _features(
    segments: BiometricSegments,
    *,
    feature_level: int,
    feature_wavelet: str,
    denoise: bool,
    ecg_method: str,
    ppg_method: str,
) -> NDArray[np.float64]:
    """Preprocess and extract the token-independent feature matrix of a bundle."""
    ecg, ppg = preprocess_signals(
        segments.ecg, segments.ppg, sampling_rate=segments.sampling_rate,
        snr_db=None, denoise=denoise, ecg_method=ecg_method, ppg_method=ppg_method,
    )
    return extract_features_batch(ecg, ppg, wavelet=feature_wavelet, level=feature_level)


def cross_session_score_pools(
    enrol_segments: BiometricSegments,
    probe_segments: BiometricSegments,
    *,
    feature_level: int = DWT_DEFAULT_LEVEL,
    feature_wavelet: str = FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    denoise: bool = True,
    ecg_method: str = DEFAULT_ECG_CLEAN_METHOD,
    ppg_method: str = DEFAULT_PPG_CLEAN_METHOD,
    score_norm: str | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool genuine/impostor scores enrolling on one condition, probing another.

    For each subject present in both bundles, the whole probe cohort is
    re-projected under that subject's token; their own enrol centroid scores
    genuine against their probes and impostor against everyone else's.

    Args:
        enrol_segments: Enrolment-condition ECG/PPG cohort (e.g. ``sit``).
        probe_segments: Probe-condition ECG/PPG cohort (e.g. ``run``), same
            subject id space as ``enrol_segments``.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing template length ``m / d``.
        binarise: Whether to sign-binarise the BioHashing projection.
        denoise: Whether to run NeuroKit physiological cleaning.
        ecg_method: Method forwarded to ``neurokit2.ecg_clean``.
        ppg_method: Method forwarded to ``neurokit2.ppg_clean``.
        score_norm: ``None`` for raw cosine, or ``"znorm"`` to z-normalise each
            claim's scores against a held-out (disjoint-subject) impostor cohort.
        seed: RNG seed for the deterministic Z-norm cohort split.

    Returns:
        Tuple ``(genuine_scores, impostor_scores)`` pooled across subjects.

    Raises:
        ValueError: If ``score_norm`` is unrecognised, the two bundles use
            different segment lengths (incompatible feature spaces), or no
            subject is common to both conditions.
    """
    if score_norm not in SCORE_NORMS:
        raise ValueError(f"Unknown score_norm {score_norm!r}. Expected one of {SCORE_NORMS}.")
    if enrol_segments.segment_length != probe_segments.segment_length:
        raise ValueError(
            "Enrol and probe segments must share a length (same rate × duration) "
            f"for a common feature space: {enrol_segments.segment_length} vs "
            f"{probe_segments.segment_length}."
        )

    feat_kwargs = dict(
        feature_level=feature_level, feature_wavelet=feature_wavelet,
        denoise=denoise, ecg_method=ecg_method, ppg_method=ppg_method,
    )
    enrol_features = _features(enrol_segments, **feat_kwargs)
    probe_features = _features(probe_segments, **feat_kwargs)
    enrol_labels = enrol_segments.labels
    probe_labels = probe_segments.labels

    common = np.intersect1d(np.unique(enrol_labels), np.unique(probe_labels))
    if common.size == 0:
        raise ValueError("No subject is present in both the enrol and probe conditions.")

    # One scaler fit on enrol only: keeps enrol and probe templates in the same
    # z-scored space (a per-batch scaler would collapse the cosine to chance) and
    # is leakage-free.
    scaler = FeatureScaler.fit(enrol_features)

    rng = make_rng(seed)

    # Filter valid victims first (skip RNG advance for skipped subjects).
    valid_common = [
        v for v in common
        if np.flatnonzero(probe_labels == v).size > 0
        and np.flatnonzero(probe_labels != v).size > 0
    ]

    # Pre-generate per-victim RNG state sequentially so parallel workers
    # receive deterministic, pre-computed values — bit-identical to the old
    # sequential loop under OMP=1.
    victim_cohort_perms: dict[int, np.ndarray | None] = {}
    for victim in valid_common:
        if score_norm == "znorm":
            other_probe = np.flatnonzero(probe_labels != victim)
            other_subjects = np.unique(probe_labels[other_probe])
            victim_cohort_perms[victim] = (
                rng.permutation(other_subjects.size)
                if other_subjects.size >= 2 else None
            )
        else:
            victim_cohort_perms[victim] = None

    def _one_victim(
        victim: int,
        cohort_perm: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        token = _token_for_label(int(victim))
        enrol_mask = enrol_labels == victim
        enrol_templates = transform_multimodal_batch(
            enrol_features[enrol_mask],
            [token] * int(np.count_nonzero(enrol_mask)),
            projection_ratio=projection_ratio, binarise=binarise, scaler=scaler,
        )
        centroid, _ = compute_subject_centroids(
            l2_normalise(enrol_templates), enrol_labels[enrol_mask],
        )
        probe_templates = transform_multimodal_batch(
            probe_features, [token] * probe_labels.shape[0],
            projection_ratio=projection_ratio, binarise=binarise, scaler=scaler,
        )
        probe_n = l2_normalise(probe_templates)

        victim_probe = np.flatnonzero(probe_labels == victim)
        other_probe = np.flatnonzero(probe_labels != victim)
        gen = cosine_score_matrix(probe_n[victim_probe], centroid).ravel()
        if score_norm == "znorm":
            other_subjects = np.unique(probe_labels[other_probe])
            if cohort_perm is not None and other_subjects.size >= 2:
                n_cohort = max(1, other_subjects.size // 2)
                cohort_subjects = other_subjects[cohort_perm[:n_cohort]]
                in_cohort = np.isin(probe_labels[other_probe], cohort_subjects)
                c_idx = other_probe[in_cohort]
                s_idx = other_probe[~in_cohort]
                cohort_scores = cosine_score_matrix(probe_n[c_idx], centroid).ravel()
                imp = cosine_score_matrix(probe_n[s_idx], centroid).ravel()
                gen, imp = znorm(gen, cohort_scores), znorm(imp, cohort_scores)
            else:
                # Only 1 non-victim subject — a disjoint z-norm cohort is impossible.
                # Self-normalisation would deflate EER by construction; use raw scores.
                imp = cosine_score_matrix(probe_n[other_probe], centroid).ravel()
        else:
            imp = cosine_score_matrix(probe_n[other_probe], centroid).ravel()
        return gen, imp

    if not valid_common:
        raise ValueError("No subject yielded both a genuine probe and an impostor probe.")

    with parallel_config(backend="loky", inner_max_num_threads=1):
        pairs = Parallel(n_jobs=DEFAULT_BATCH_N_JOBS)(
            delayed(_one_victim)(victim, victim_cohort_perms[victim])
            for victim in valid_common
        )

    genuine_pool = [g for g, _ in pairs]
    impostor_pool = [i for _, i in pairs]
    if not genuine_pool:
        raise ValueError("No subject yielded both a genuine probe and an impostor probe.")
    return np.concatenate(genuine_pool), np.concatenate(impostor_pool)


def cross_session_verification(
    enrol_segments: BiometricSegments,
    probe_segments: BiometricSegments,
    *,
    enrol_label: str = "enrol",
    probe_label: str = "probe",
    seed: int = DEFAULT_SEED,
    **kwargs: object,
) -> CrossSessionResult:
    """Summarise cross-condition verification into an EER / decidability report.

    Args:
        enrol_segments: Enrolment-condition cohort.
        probe_segments: Probe-condition cohort (same subject id space).
        enrol_label: Human-readable name of the enrol condition (for the report).
        probe_label: Human-readable name of the probe condition (for the report).
        seed: RNG seed forwarded to pooling and the EER bootstrap CI.
        **kwargs: Forwarded to :func:`cross_session_score_pools`.

    Returns:
        A :class:`CrossSessionResult` over the pooled genuine/impostor scores.
    """
    genuine, impostor = cross_session_score_pools(
        enrol_segments, probe_segments, seed=seed, **kwargs,  # type: ignore[arg-type]
    )
    stats = get_eer_stats(genuine, impostor)
    eer_ci_low, eer_ci_high = bootstrap_eer_ci(genuine, impostor, seed=seed)
    # Count only subjects that actually contributed scores (probe segment present +
    # at least one impostor), not every subject in the label intersection.
    _pl = probe_segments.labels
    _common = np.intersect1d(np.unique(enrol_segments.labels), np.unique(_pl))
    n_subjects = int(sum(
        1 for v in _common
        if np.any(_pl == v) and np.any(_pl != v)
    ))
    result = CrossSessionResult(
        enrol_label=enrol_label,
        probe_label=probe_label,
        n_subjects=n_subjects,
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
        "[cross-session %s→%s | %d subjects | %d gen | %d imp] EER=%.4f d'=%.3f",
        enrol_label, probe_label, result.n_subjects,
        result.n_genuine, result.n_impostor, result.eer, result.decidability,
    )
    return result


__all__ = [
    "CrossSessionResult",
    "cross_session_score_pools",
    "cross_session_verification",
]

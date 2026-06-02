"""Wu-style non-invertibility reporting: three correlation distributions + SAR.

The single-template inversion in :mod:`mwf.inversion` quantifies *how close* a
reconstruction ``x̂ = T⁻¹(y, k)`` is to the original feature vector ``x``. On its
own, a correlation of ~0.70 looks alarming; what makes it interpretable is the
side-by-side reference that Wu et al. (TIFS 2019) use as the cancelable-biometrics
gold standard:

  1. **mated** — corr(x̂, x), the reconstruction-vs-original correlation;
  2. **non-mated** — corr(x̂, x'), reconstruction vs *another* subject's feature
     (the chance-level baseline: how much of the ~0.70 is just feature-space
     statistics rather than target-specific leakage);
  3. **genuine reference** — corr(x_a, x_b) between two real samples of the same
     subject (the natural intra-subject ceiling against which mated should be
     judged).

A non-invertible transform pushes mated toward non-mated; a fully invertible one
pushes mated toward the genuine reference. The gap is the figure of merit, not
the absolute mated number.

On top of the distributions we report the **Success Attack Rate** (SAR), Wu et
al.'s operational follow-through: re-protect ``x̂`` under the victim's token and
score it against the stored template at the verifier's EER threshold. A
non-invertible transform yields a SAR comparable to the FAR; an invertible one
inflates it toward the GAR.

Both reports use the multimodal hybrid transform (BioHashing on ECG, IoM on PPG):
the ECG block is inverted with the min-norm pre-image ``Rᵀ y`` (the same √(m/d)
leakage as plain BioHashing), the PPG block with
:func:`mwf.inversion.recover_ppg_iom`'s best-effort winner-direction sum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from . import iom
from .operating_curves import det_curve_from_scores
from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
    IOM_HASHES_RATIO,
    IOM_WINDOW,
)
from .dataset import BiometricSegments
from .feature_transform import (
    ECG_SALT,
    PPG_SALT,
    _apply,
    derive_projection,
    projection_dim,
    transform_multimodal_batch,
)
from .features import extract_features_batch
from .inversion import _safe_corr, recover_ppg_iom
from .pipeline import _token_for_label, preprocess_signals
from .rng import make_rng
from .scoring import cosine_score_matrix, l2_normalise

logger = logging.getLogger(__name__)

DEFAULT_NON_MATED_PAIRS_PER_VICTIM: Final[int] = 8


@dataclass(frozen=True, slots=True)
class NonInvertibilityReport:
    """Aggregate Wu-style non-invertibility numbers.

    Attributes:
        n_victims: Subjects exercised as reconstruction targets.
        n_mated: Mated reconstruction-vs-original pair count.
        n_non_mated: Non-mated reconstruction-vs-other-subject pair count.
        n_genuine_ref: Intra-subject original-vs-original pair count.
        mated_mean: Mean ``|corr(x̂, x)|`` (target leakage).
        mated_std: Std of the mated pool.
        non_mated_mean: Mean ``|corr(x̂, x')|`` (chance baseline).
        non_mated_std: Std of the non-mated pool.
        genuine_ref_mean: Mean ``|corr(x_a, x_b)|`` between two real samples.
        genuine_ref_std: Std of the genuine-reference pool.
        leakage_gap: ``mated_mean − non_mated_mean``. Wu's gold standard is
            ``≈ 0`` (no target-specific leakage).
        sar_type1: Protected-system SAR — fraction of re-protected
            reconstructions that pass the verifier at its EER threshold.
        sar_type2: Raw-feature SAR — fraction of reconstructions that pass an
            unprotected cosine matcher at its EER threshold.
        sar_threshold_type1: Threshold used by :attr:`sar_type1` (protected
            cosine).
        sar_threshold_type2: Threshold used by :attr:`sar_type2` (raw cosine).
        verifier_eer: EER of the unprotected (per-subject token) cosine
            verifier, included as the operating-point reference for SAR-I.
        raw_eer: EER of the raw-feature cosine verifier, the SAR-II reference.
    """

    n_victims: int
    n_mated: int
    n_non_mated: int
    n_genuine_ref: int
    mated_mean: float
    mated_std: float
    non_mated_mean: float
    non_mated_std: float
    genuine_ref_mean: float
    genuine_ref_std: float
    leakage_gap: float
    sar_type1: float
    sar_type2: float
    sar_threshold_type1: float
    sar_threshold_type2: float
    verifier_eer: float
    raw_eer: float


def _reconstruct_features(
    features: NDArray[np.float64],
    tokens: list[str],
    projection_ratio: float,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
) -> NDArray[np.float64]:
    """Min-norm pre-image of the multimodal feature vector for every row.

    Mirrors :func:`mwf.inversion.multimodal_leakage_metrics`: the ECG half is
    recovered with ``Rᵀ y``; the PPG half with the IoM winner-direction sum.
    Real-valued (non-binarised) projection is used as the conservative,
    adversary-favouring upper bound on what is recoverable.

    Args:
        features: ``(n, d)`` extracted feature matrix (even ``d``; ECG‖PPG).
        tokens: One per row.
        projection_ratio: ECG BioHashing ratio ``m/d_ecg``.
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.

    Returns:
        ``(n, d)`` matrix of recovered feature vectors.
    """
    n, d = features.shape
    half = d // 2
    m_ecg = projection_dim(half, projection_ratio)
    n_hashes = iom.hash_count(half, n_hashes_ratio)
    out = np.empty_like(features)
    for i in range(n):
        token = tokens[i]
        ecg = features[i, :half]
        ppg = features[i, half:]
        r_ecg = derive_projection(token + ECG_SALT, half, m_ecg).matrix
        y_ecg = _apply(r_ecg, ecg, binarise=False)
        out[i, :half] = r_ecg.T @ y_ecg
        out[i, half:] = recover_ppg_iom(ppg, token + PPG_SALT, n_hashes, window)
    return out


def _correlation_pools(
    features: NDArray[np.float64],
    recovered: NDArray[np.float64],
    labels: NDArray[np.int64],
    uniq: NDArray[np.int64],
    rng: np.random.Generator,
    non_mated_pairs_per_victim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the three Pearson-correlation pools (mated / non-mated / genuine-ref)."""
    mated: list[float] = []
    non_mated: list[float] = []
    genuine_ref: list[float] = []
    for victim in uniq:
        victim_idx = np.flatnonzero(labels == victim)
        other_idx = np.flatnonzero(labels != victim)
        for i in victim_idx:
            mated.append(_safe_corr(recovered[i], features[i]))
        if other_idx.size and victim_idx.size:
            n_pairs = min(
                victim_idx.size * non_mated_pairs_per_victim, other_idx.size,
            )
            j_choice = rng.choice(other_idx, size=n_pairs, replace=False)
            i_choice = rng.choice(victim_idx, size=n_pairs, replace=True)
            for i, j in zip(i_choice, j_choice):
                non_mated.append(_safe_corr(recovered[i], features[j]))
        if victim_idx.size >= 2:
            perm = rng.permutation(victim_idx)
            for k in range(perm.size - 1):
                genuine_ref.append(_safe_corr(features[perm[k]], features[perm[k + 1]]))
    return (
        np.asarray(mated, dtype=np.float64),
        np.asarray(non_mated, dtype=np.float64),
        np.asarray(genuine_ref, dtype=np.float64),
    )


def _eer_threshold(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64],
) -> tuple[float, float]:
    """Return ``(eer, threshold_at_eer)`` from a score-derived DET curve."""
    if genuine.size < 2 or impostor.size < 2:
        return float("nan"), float("nan")
    det = det_curve_from_scores(genuine, impostor)
    idx = int(np.argmin(np.abs(det.fmr - det.fnmr)))
    eer = float(0.5 * (det.fmr[idx] + det.fnmr[idx]))
    return eer, float(det.thresholds[idx])


def _attack_pairs(
    templates: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Off-diagonal cosine genuine / impostor scores for a template matrix.

    The diagonal (a template scored against itself) is excluded from *both*
    pools — a cosine ``= 1`` self-match would inflate the impostor tail and
    skew the EER threshold downstream.
    """
    sim = cosine_score_matrix(templates, templates)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    off_diagonal = ~np.eye(labels.size, dtype=bool)
    return sim[same], sim[~same & off_diagonal]


def _sar(
    references: NDArray[np.float64],
    probes: NDArray[np.float64],
    labels: NDArray[np.int64],
    uniq: NDArray[np.int64],
    threshold: float,
) -> float:
    """Per-victim SAR: fraction of probe-vs-victim-centroid scores ≥ ``threshold``.

    Uses the L2-normalised mean of the victim's reference templates as the
    enrolment centroid (the standard cosine matcher of :mod:`mwf.stolen_token`).
    """
    if not np.isfinite(threshold):
        return float("nan")
    refs_n = l2_normalise(references)
    probes_n = l2_normalise(probes)
    success = 0
    total = 0
    for victim in uniq:
        mask = labels == victim
        if not np.any(mask):
            continue
        centroid = l2_normalise(refs_n[mask].mean(axis=0, keepdims=True))
        scores = (probes_n[mask] @ centroid.T).ravel()
        success += int(np.sum(scores >= threshold))
        total += int(scores.size)
    return float(success / total) if total else float("nan")


def non_invertibility_analysis(
    segments: BiometricSegments,
    *,
    feature_level: int = DWT_DEFAULT_LEVEL,
    feature_wavelet: str = FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    n_hashes_ratio: float = IOM_HASHES_RATIO,
    window: int = IOM_WINDOW,
    max_victims: int | None = None,
    non_mated_pairs_per_victim: int = DEFAULT_NON_MATED_PAIRS_PER_VICTIM,
    seed: int = DEFAULT_SEED,
    denoise: bool = True,
) -> tuple[NonInvertibilityReport, dict[str, NDArray[np.float64]]]:
    """Wu-style non-invertibility analysis on the multimodal cohort.

    Args:
        segments: ECG/PPG cohort with per-subject labels.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: ECG BioHashing ratio ``m/d_ecg``.
        binarise: Whether the operational ECG block is sign-binarised (the
            re-protected reconstruction reproduces this for SAR-I).
        n_hashes_ratio: IoM code length as a multiple of the PPG block size.
        window: IoM projections per hash ``q``.
        max_victims: Cap on subjects exercised as targets (``None`` = all).
        non_mated_pairs_per_victim: Cap on the non-mated subsample per victim.
        seed: Master RNG seed.
        denoise: Whether to run NeuroKit cleaning before feature extraction.

    Returns:
        Tuple ``(report, pools)``. ``pools`` maps the three population labels
        (``"mated"``, ``"non_mated"``, ``"genuine_ref"``) to their absolute
        correlations — the raw material the KDE figure consumes.
    """
    ecg, ppg = preprocess_signals(
        segments.ecg, segments.ppg, sampling_rate=segments.sampling_rate,
        snr_db=None, denoise=denoise,
    )
    features = extract_features_batch(ecg, ppg, wavelet=feature_wavelet, level=feature_level)
    labels = segments.labels
    uniq = np.unique(labels)
    rng = make_rng(seed)
    if max_victims is not None and max_victims < uniq.size:
        uniq = np.sort(rng.choice(uniq, size=max_victims, replace=False))

    cohort_mask = np.isin(labels, uniq)
    features = features[cohort_mask]
    labels = labels[cohort_mask]

    tokens = [_token_for_label(int(l)) for l in labels]
    recovered = _reconstruct_features(
        features, tokens, projection_ratio,
        n_hashes_ratio=n_hashes_ratio, window=window,
    )

    mated, non_mated, genuine_ref = _correlation_pools(
        features, recovered, labels, uniq, rng,
        non_mated_pairs_per_victim=non_mated_pairs_per_victim,
    )

    # Operating-point references: derive each threshold from the verifier's
    # legitimate genuine/impostor pools (per-subject token for SAR-I; raw
    # features for SAR-II), then ask what fraction of reconstructions pass.
    protected = transform_multimodal_batch(
        features, tokens, projection_ratio=projection_ratio, binarise=binarise,
    )
    rec_protected = transform_multimodal_batch(
        recovered, tokens, projection_ratio=projection_ratio, binarise=binarise,
    )
    gen_p, imp_p = _attack_pairs(protected, labels)
    eer_p, thr_p = _eer_threshold(gen_p, imp_p)
    sar1 = _sar(protected, rec_protected, labels, uniq, thr_p)

    gen_raw, imp_raw = _attack_pairs(features, labels)
    eer_raw, thr_raw = _eer_threshold(gen_raw, imp_raw)
    sar2 = _sar(features, recovered, labels, uniq, thr_raw)

    abs_or = lambda a: np.abs(a) if a.size else a
    mated_abs = abs_or(mated)
    non_mated_abs = abs_or(non_mated)
    genuine_ref_abs = abs_or(genuine_ref)

    report = NonInvertibilityReport(
        n_victims=int(uniq.size),
        n_mated=int(mated.size),
        n_non_mated=int(non_mated.size),
        n_genuine_ref=int(genuine_ref.size),
        mated_mean=float(mated_abs.mean()) if mated.size else float("nan"),
        mated_std=float(mated_abs.std(ddof=1)) if mated.size > 1 else 0.0,
        non_mated_mean=float(non_mated_abs.mean()) if non_mated.size else float("nan"),
        non_mated_std=float(non_mated_abs.std(ddof=1)) if non_mated.size > 1 else 0.0,
        genuine_ref_mean=float(genuine_ref_abs.mean()) if genuine_ref.size else float("nan"),
        genuine_ref_std=float(genuine_ref_abs.std(ddof=1)) if genuine_ref.size > 1 else 0.0,
        leakage_gap=(
            float(mated_abs.mean() - non_mated_abs.mean())
            if mated.size and non_mated.size else float("nan")
        ),
        sar_type1=sar1,
        sar_type2=sar2,
        sar_threshold_type1=thr_p,
        sar_threshold_type2=thr_raw,
        verifier_eer=eer_p,
        raw_eer=eer_raw,
    )
    logger.info(
        "[non-invertibility | %d victims] mated=%.3f non-mated=%.3f ref=%.3f "
        "gap=%.3f SAR-I=%.3f SAR-II=%.3f",
        report.n_victims,
        report.mated_mean, report.non_mated_mean, report.genuine_ref_mean,
        report.leakage_gap, report.sar_type1, report.sar_type2,
    )
    return report, {
        "mated": mated_abs,
        "non_mated": non_mated_abs,
        "genuine_ref": genuine_ref_abs,
    }


__all__ = [
    "NonInvertibilityReport",
    "non_invertibility_analysis",
]

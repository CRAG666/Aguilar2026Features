"""Per-subject ablation: how much of the per_subject win is the key, not biometry?

``per_subject`` reports the regime's signature result — EER ~10⁻⁴ at deep feature
levels. The number is correct, but it is open to a fair reviewer's objection: the
classifier may be separating *tokens*, not biometric content. With one fresh
orthonormal projection per subject, two subjects' template clouds live in two
randomly oriented ``m``-dimensional subspaces of ``ℝ^d``; that geometric
separation alone is roughly impostor-cosine ``≈ 0``, even before any biometric
contribution.

This module isolates the contribution by sweeping the number of subjects that
share one token (the *group size*):

  * **group_size = 1**  — each subject has their own token (the operational
    ``per_subject`` regime; the key dominates because no two subjects are
    projected into the same subspace);
  * **group_size > 1**  — subjects are partitioned into groups; everyone in a
    group shares one token. Impostor pairs *within a group* are now projected
    through the same ``R``, so token-level separation no longer carries the
    score — only the biometric does;
  * **group_size = n_subjects** — the limiting case is ``single_key``, the
    honest biometric-only floor where no two subjects benefit from independent
    projections.

The interpolation answers the reviewer directly: if EER stays low up to
``group_size = n_subjects`` the recognition is biometric; if it collapses from
``~10⁻⁴`` to ``~10⁻²`` between ``group_size = 1`` and ``group_size = n_subjects``
the key was carrying most of the per_subject win.

Impostors stay defined by subject label, so the same/different counts only
change when group_size changes — there is no leakage between same-label rows
of different group_size sweeps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_info import get_eer_stats

from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
)
from .dataset import BiometricSegments
from .feature_transform import transform_multimodal_batch
from .features import extract_features_batch
from .pipeline import preprocess_signals
from .rng import make_rng
from .scoring import cosine_score_matrix, decidability

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AblationPoint:
    """One ablation sweep point.

    Attributes:
        group_size: Number of subjects sharing one token at this point
            (``1`` = per_subject, ``n_subjects`` = single_key).
        n_groups: Number of distinct tokens issued.
        n_genuine: Genuine (same-subject, off-diagonal) score count.
        n_impostor: Impostor (different-subject) score count.
        n_within_group_impostor: Subset of impostor scores where the two
            subjects share a token. ``0`` at ``group_size = 1``; equals the
            full impostor count at ``group_size = n_subjects``.
        eer: EER on the pooled (subject-labelled) cosine scores.
        decidability: Daugman ``d'``.
        within_group_impostor_mean: Mean impostor score *inside* a token group
            — the score level the matcher has to overcome with biometric
            content alone.
        across_group_impostor_mean: Mean impostor score *across* token groups,
            where the random-projection mismatch contributes too.
        genuine_mean: Mean genuine score.
    """

    group_size: int
    n_groups: int
    n_genuine: int
    n_impostor: int
    n_within_group_impostor: int
    eer: float
    decidability: float
    within_group_impostor_mean: float
    across_group_impostor_mean: float
    genuine_mean: float


def _group_token_map(
    subjects: NDArray[np.int64], group_size: int, rng: np.random.Generator,
) -> dict[int, str]:
    """Partition ``subjects`` into roughly equal-size groups, one token each.

    Args:
        subjects: Unique subject labels.
        group_size: Target subjects per group (clamped to ``[1, len(subjects)]``).
        rng: Source of randomness for the partition.

    Returns:
        ``{subject_label: token}`` covering every input subject.
    """
    n = subjects.size
    size = max(1, min(group_size, n))
    shuffled = rng.permutation(subjects)
    n_groups = max(1, int(np.ceil(n / size)))
    groups = np.array_split(shuffled, n_groups)
    return {
        int(s): f"ABL_GROUP_{group_size:03d}_{g_idx:03d}"
        for g_idx, group in enumerate(groups)
        for s in group
    }


def _pair_pools(
    sim: NDArray[np.float64],
    labels: NDArray[np.int64],
    group_id: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split a cosine-score matrix into genuine, within-group and across-group impostor pools.

    The diagonal (a template compared with itself) is dropped from the genuine
    pool — it is the trivial cosine ``= 1`` and would inflate the mean.
    """
    same_subject = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_subject, False)
    same_group = group_id[:, None] == group_id[None, :]
    genuine = sim[same_subject]
    impostor_mask = ~same_subject & ~np.eye(labels.size, dtype=bool)
    np.fill_diagonal(impostor_mask, False)
    within = sim[impostor_mask & same_group]
    across = sim[impostor_mask & ~same_group]
    return genuine, within, across


def per_subject_ablation(
    segments: BiometricSegments,
    *,
    feature_level: int = DWT_DEFAULT_LEVEL,
    feature_wavelet: str = FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    group_sizes: tuple[int, ...] | None = None,
    seed: int = DEFAULT_SEED,
    denoise: bool = True,
) -> list[AblationPoint]:
    """Sweep how many subjects share a key, measuring EER at each setting.

    Features are extracted once (token-independent); each group size only
    changes the per-row token, so the sweep isolates the contribution of token
    uniqueness to the recognition margin.

    Args:
        segments: ECG/PPG cohort.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing ``m/d_ecg`` for the ECG block.
        binarise: Whether to sign-binarise the BioHashing projection.
        group_sizes: Group sizes to sweep. ``None`` picks a sensible default
            spanning ``[1, n_subjects]``.
        seed: Master RNG seed for the partition.
        denoise: Whether to run NeuroKit cleaning before feature extraction.

    Returns:
        One :class:`AblationPoint` per group size, in input order.
    """
    ecg, ppg = preprocess_signals(
        segments.ecg, segments.ppg, sampling_rate=segments.sampling_rate,
        snr_db=None, denoise=denoise,
    )
    features = extract_features_batch(
        ecg, ppg, wavelet=feature_wavelet, level=feature_level,
    )
    labels = segments.labels
    uniq = np.unique(labels)
    n_subjects = int(uniq.size)
    if group_sizes is None:
        # Span 1 .. n_subjects on a roughly log scale; always include the two
        # endpoints so the reader sees per_subject vs single_key at a glance.
        endpoints = {1, n_subjects}
        steps = {2, 5, 10, 25, 50}
        group_sizes = tuple(sorted(s for s in (endpoints | steps) if 1 <= s <= n_subjects))
    rng = make_rng(seed)
    points: list[AblationPoint] = []
    for size in group_sizes:
        token_of = _group_token_map(uniq, size, rng)
        tokens = [token_of[int(l)] for l in labels]
        group_id_map = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
        group_id = np.asarray([group_id_map[t] for t in tokens], dtype=np.int64)
        templates = transform_multimodal_batch(
            features, tokens, projection_ratio=projection_ratio, binarise=binarise,
        )
        sim = cosine_score_matrix(templates, templates)
        genuine, within, across = _pair_pools(sim, labels, group_id)
        impostor = np.concatenate([within, across]) if within.size or across.size else across
        if genuine.size < 2 or impostor.size < 2:
            eer = float("nan")
            deci = float("nan")
        else:
            stats = get_eer_stats(genuine, impostor)
            eer = float(stats.eer)
            deci = decidability(genuine, impostor)
        point = AblationPoint(
            group_size=int(size),
            n_groups=len(group_id_map),
            n_genuine=int(genuine.size),
            n_impostor=int(impostor.size),
            n_within_group_impostor=int(within.size),
            eer=eer,
            decidability=deci,
            within_group_impostor_mean=(
                float(within.mean()) if within.size else float("nan")
            ),
            across_group_impostor_mean=(
                float(across.mean()) if across.size else float("nan")
            ),
            genuine_mean=float(genuine.mean()) if genuine.size else float("nan"),
        )
        logger.info(
            "[per-subject ablation | group_size=%d, n_groups=%d] EER=%.4f d'=%.3f "
            "within_imp=%.3f across_imp=%.3f",
            point.group_size, point.n_groups, point.eer, point.decidability,
            point.within_group_impostor_mean, point.across_group_impostor_mean,
        )
        points.append(point)
    return points


__all__ = [
    "AblationPoint",
    "per_subject_ablation",
]

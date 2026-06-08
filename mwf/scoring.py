"""Score-level utilities shared by verification, cancelability and non-invertibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_stats import get_decidability_value
from sklearn.preprocessing import normalize


def l2_normalise(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise L2 normalise ``x``; zero rows are left untouched.

    Args:
        x: ``(n, d)`` float matrix.

    Returns:
        New ``(n, d)`` matrix with rows of unit L2 norm.
    """
    return normalize(x, norm="l2", axis=1, copy=True)


def cosine_score_matrix(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the cosine-similarity matrix between rows of ``a`` and ``b``.

    Args:
        a: ``(M, d)`` matrix.
        b: ``(N, d)`` matrix.

    Returns:
        ``(M, N)`` cosine-similarity matrix.
    """
    return l2_normalise(a) @ l2_normalise(b).T


def compute_subject_centroids(
    features: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute per-subject mean feature vectors.

    Args:
        features: ``(n, d)`` feature matrix.
        labels: ``(n,)`` subject identifiers.

    Returns:
        Tuple ``(centroids, subject_ids)`` with one centroid per unique label.
    """
    uniq, inv = np.unique(labels, return_inverse=True)
    n_subjects = len(uniq)
    centroids = np.zeros((n_subjects, features.shape[1]), dtype=np.float64)
    np.add.at(centroids, inv, features)
    centroids /= np.bincount(inv, minlength=n_subjects)[:, None]
    return centroids, uniq


def genuine_impostor_split(
    score_matrix: NDArray[np.float64],
    labels_rows: NDArray[np.int64],
    labels_cols: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a score matrix into genuine and impostor pools.

    Args:
        score_matrix: ``(M, N)`` similarity matrix.
        labels_rows: ``(M,)`` labels indexing rows.
        labels_cols: ``(N,)`` labels indexing columns.

    Returns:
        Tuple ``(genuine_scores, impostor_scores)`` as 1-D arrays.
    """
    same = labels_rows[:, None] == labels_cols[None, :]
    return score_matrix[same], score_matrix[~same]


def genuine_impostor_scores(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute cosine scores and split into genuine/impostor pools.

    Args:
        a: ``(n, d)`` template matrix.
        b: ``(n, d)`` template matrix sharing ``labels`` with ``a``.
        labels: ``(n,)`` subject identifiers.

    Returns:
        Tuple ``(genuine_scores, impostor_scores)``.
    """
    return genuine_impostor_split(cosine_score_matrix(a, b), labels, labels)


def znorm(
    scores: NDArray[np.float64], cohort: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Z-normalise ``scores`` against an impostor ``cohort``'s mean and std.

    Cosine scores against different enrolment centroids sit on slightly
    different scales, so a single global accept/reject threshold generalises
    poorly across subjects (and worse to unseen subjects) — the dominant cause
    of the large open-set verification EER. Z-norm rescales each probe's scores
    by the location and spread of its impostor cohort, making one threshold
    comparable across centroids.

    Args:
        scores: Raw similarity scores to normalise.
        cohort: Impostor (non-mated) scores defining the reference distribution.

    Returns:
        ``(scores − μ_cohort) / σ_cohort``; falls back to centring only when
        the cohort has zero spread or fewer than two samples.
    """
    if cohort.size < 2:
        return scores - float(np.mean(cohort)) if cohort.size else scores
    mu = float(np.mean(cohort))
    sd = float(np.std(cohort, ddof=1))
    return (scores - mu) / sd if sd > 0.0 else scores - mu


def decidability(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64]
) -> float:
    """Compute Daugman's decidability index ``d'``.

    Args:
        genuine: Pool of genuine scores.
        impostor: Pool of impostor scores.

    Returns:
        ``d' = |μ_g − μ_i| / √(½ (σ_g² + σ_i²))``, or ``NaN`` if either
        pool has fewer than two samples or both pools have zero variance.
    """
    if genuine.size < 2 or impostor.size < 2:
        return float("nan")
    g_std = float(np.std(genuine, ddof=1))
    i_std = float(np.std(impostor, ddof=1))
    if g_std == 0.0 and i_std == 0.0:
        return float("nan")
    return float(get_decidability_value(
        float(genuine.mean()), g_std, float(impostor.mean()), i_std,
    ))


__all__ = [
    "compute_subject_centroids",
    "cosine_score_matrix",
    "decidability",
    "genuine_impostor_scores",
    "genuine_impostor_split",
    "l2_normalise",
    "znorm",
]

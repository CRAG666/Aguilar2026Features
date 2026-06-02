"""1:1 verification protocol: cosine genuine/impostor EER over CV folds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_info import get_eer_stats

from .constants import DEFAULT_N_FOLDS, DEFAULT_SEED, DEFAULT_SEGMENTS_PER_BLOCK
from .cv_splits import stratified_group_splitter, temporal_block_groups
from .evaluation import make_cv_splitter
from .operating_curves import operating_points
from .pipeline import KeyMode, TemplateBundle
from .rng import make_rng
from .scoring import (
    compute_subject_centroids,
    cosine_score_matrix,
    genuine_impostor_split,
    l2_normalise,
)

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE: Final[int] = DEFAULT_SEED

VerificationMode = Literal["closed_set", "open_set"]


@dataclass(frozen=True, slots=True)
class VerificationFoldResult:
    """Verification metrics for one CV fold.

    Attributes:
        fold_idx: Zero-based fold index.
        n_enrolled_subjects: Subjects enrolled in this fold.
        n_genuine: Genuine score count.
        n_impostor: Impostor score count.
        eer: Equal-error rate.
        decidability: Daugman's ``d'``.
        genuine_mean: Mean of genuine scores.
        impostor_mean: Mean of impostor scores.
        operating_points: FNMR@FMR / FMR@FNMR dictionary.
    """

    fold_idx: int
    n_enrolled_subjects: int
    n_genuine: int
    n_impostor: int
    eer: float
    decidability: float
    genuine_mean: float
    impostor_mean: float
    operating_points: dict[str, float]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Per-fold verification results for one template bundle.

    Attributes:
        mode: ``"closed_set"`` or ``"open_set"``.
        key_mode: Key regime that produced the templates.
        n_features: Template dimensionality.
        fold_results: One :class:`VerificationFoldResult` per CV fold.
    """

    mode: VerificationMode
    key_mode: KeyMode
    n_features: int
    fold_results: tuple[VerificationFoldResult, ...]

    @property
    def n_folds(self) -> int:
        """Number of CV folds."""
        return len(self.fold_results)

    def eer_values(self) -> NDArray[np.float64]:
        """Return per-fold EER values as a 1-D array."""
        return np.fromiter(
            (f.eer for f in self.fold_results),
            dtype=np.float64,
            count=self.n_folds,
        )

    def decidability_values(self) -> NDArray[np.float64]:
        """Return per-fold decidability ``d'`` values as a 1-D array."""
        return np.fromiter(
            (f.decidability for f in self.fold_results),
            dtype=np.float64,
            count=self.n_folds,
        )


def _empty_fold(fold_idx: int, n_enrolled: int) -> VerificationFoldResult:
    """Build an all-NaN fold result for degenerate splits.

    Args:
        fold_idx: Zero-based fold index.
        n_enrolled: Subjects enrolled in this fold.

    Returns:
        A :class:`VerificationFoldResult` with NaN metrics.
    """
    nan = float("nan")
    return VerificationFoldResult(
        fold_idx=fold_idx,
        n_enrolled_subjects=n_enrolled,
        n_genuine=0,
        n_impostor=0,
        eer=nan,
        decidability=nan,
        genuine_mean=nan,
        impostor_mean=nan,
        operating_points={},
    )


def _fold_result(
    fold_idx: int,
    enrolment: NDArray[np.float64],
    uniq: NDArray[np.int64],
    query_feats: NDArray[np.float64],
    query_labels: NDArray[np.int64],
    extra_impostor_scores: NDArray[np.float64] | None = None,
) -> VerificationFoldResult:
    """Score queries against enrolment centroids and summarise the fold.

    Args:
        fold_idx: Fold index used for reporting.
        enrolment: ``(K, d)`` enrolment centroids.
        uniq: ``(K,)`` subject identifiers for ``enrolment``.
        query_feats: ``(M, d)`` query feature matrix.
        query_labels: ``(M,)`` query subject labels.
        extra_impostor_scores: Optional additional impostor scores to merge.

    Returns:
        A :class:`VerificationFoldResult` for the fold.
    """
    sim = cosine_score_matrix(query_feats, enrolment)
    genuine, impostor = genuine_impostor_split(sim, query_labels, uniq)
    if extra_impostor_scores is not None:
        impostor = np.concatenate([impostor, extra_impostor_scores])
    if genuine.size == 0 or impostor.size == 0:
        return _empty_fold(fold_idx, int(uniq.shape[0]))

    stats = get_eer_stats(genuine, impostor)
    return VerificationFoldResult(
        fold_idx=fold_idx,
        n_enrolled_subjects=int(uniq.shape[0]),
        n_genuine=int(genuine.size),
        n_impostor=int(impostor.size),
        eer=float(stats.eer),
        decidability=float(stats.decidability),
        genuine_mean=float(stats.gmean),
        impostor_mean=float(stats.imean),
        operating_points=operating_points(genuine, impostor),
    )


def _score_fold_closed_set(
    feats_n: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    fold_idx: int,
) -> VerificationFoldResult:
    """Score a closed-set fold using subject centroids from ``train_idx``."""
    enrolment, uniq = compute_subject_centroids(feats_n[train_idx], labels[train_idx])
    return _fold_result(fold_idx, enrolment, uniq, feats_n[test_idx], labels[test_idx])


def _score_fold_open_set(
    feats_n: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    fold_idx: int,
    rng: np.random.Generator,
) -> VerificationFoldResult:
    """Score an open-set fold; held-out subjects act as additional impostors."""
    enrol_feats = feats_n[train_idx]
    enrol_labels = labels[train_idx]
    perm = rng.permutation(enrol_feats.shape[0])
    half = enrol_feats.shape[0] // 2
    ref_idx, que_idx = perm[:half], perm[half:]
    enrolment, uniq = compute_subject_centroids(enrol_feats[ref_idx], enrol_labels[ref_idx])
    impostor_out = cosine_score_matrix(feats_n[test_idx], enrolment).ravel()
    return _fold_result(
        fold_idx,
        enrolment,
        uniq,
        enrol_feats[que_idx],
        enrol_labels[que_idx],
        extra_impostor_scores=impostor_out,
    )


def closed_set_score_pools(
    bundle: TemplateBundle,
    n_folds: int = DEFAULT_N_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool genuine/impostor scores from the closed-set verification CV.

    Uses the identical protocol as :func:`run_verification_cv` in
    ``closed_set`` mode — StratifiedGroupKFold over temporal blocks, with
    queries scored against train-fold subject centroids — and concatenates the
    per-fold pools. Because enrolment (train) and query (test) sets are
    disjoint, there are no self-matches and no train/test leakage.

    Args:
        bundle: Template features and subject labels.
        n_folds: Number of CV folds.
        random_state: Seed for the splitter.
        segments_per_block: Block size for temporal grouping.

    Returns:
        Tuple ``(genuine_scores, impostor_scores)`` pooled across folds.
    """
    feats_n = l2_normalise(bundle.features)
    labels = bundle.labels
    groups = temporal_block_groups(labels, segments_per_block=segments_per_block)
    splitter = stratified_group_splitter(n_splits=n_folds, random_state=random_state)
    genuine_pool: list[NDArray[np.float64]] = []
    impostor_pool: list[NDArray[np.float64]] = []
    for train_idx, test_idx in splitter.split(feats_n, labels, groups=groups):
        enrolment, uniq = compute_subject_centroids(feats_n[train_idx], labels[train_idx])
        sim = cosine_score_matrix(feats_n[test_idx], enrolment)
        genuine, impostor = genuine_impostor_split(sim, labels[test_idx], uniq)
        genuine_pool.append(genuine)
        impostor_pool.append(impostor)
    return np.concatenate(genuine_pool), np.concatenate(impostor_pool)


def run_verification_cv(
    bundle: TemplateBundle,
    mode: VerificationMode = "closed_set",
    n_folds: int = DEFAULT_N_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
) -> VerificationResult:
    """Run 1:1 verification CV in closed-set or open-set mode.

    Args:
        bundle: Template features and subject labels.
        mode: ``"closed_set"`` or ``"open_set"``.
        n_folds: Number of CV folds.
        random_state: Seed for splitters and open-set permutation.
        segments_per_block: Block size for temporal grouping (closed-set).

    Returns:
        A :class:`VerificationResult` with one entry per fold.

    Raises:
        ValueError: If ``mode`` is not recognised.
    """
    feats_n = l2_normalise(bundle.features)
    labels = bundle.labels
    rng = make_rng(random_state)

    match mode:
        case "closed_set":
            groups = temporal_block_groups(labels, segments_per_block=segments_per_block)
            splitter = stratified_group_splitter(
                n_splits=n_folds,
                random_state=random_state,
            )
            per_fold = [
                _score_fold_closed_set(feats_n, labels, tr, te, i)
                for i, (tr, te) in enumerate(splitter.split(feats_n, labels, groups=groups))
            ]
        case "open_set":
            splitter = make_cv_splitter(
                "group_kfold",
                n_splits=n_folds,
                random_state=random_state,
            )
            per_fold = [
                _score_fold_open_set(feats_n, labels, tr, te, i, rng)
                for i, (tr, te) in enumerate(splitter.split(feats_n, labels, groups=labels))
            ]
        case _:
            raise ValueError(
                f"Unknown verification mode {mode!r}; expected 'closed_set' or 'open_set'."
            )

    result = VerificationResult(
        mode=mode,
        key_mode=bundle.key_mode,
        n_features=bundle.features.shape[1],
        fold_results=tuple(per_fold),
    )
    eer = result.eer_values()
    deci = result.decidability_values()
    logger.info(
        "[verification | %s | %s | %d feats] EER=%.4f±%.4f d'=%.3f±%.3f",
        mode,
        bundle.key_mode.value,
        bundle.features.shape[1],
        float(np.nanmean(eer)),
        float(np.nanstd(eer, ddof=1)),
        float(np.nanmean(deci)),
        float(np.nanstd(deci, ddof=1)),
    )
    return result


__all__ = [
    "VerificationFoldResult",
    "VerificationMode",
    "VerificationResult",
    "closed_set_score_pools",
    "run_verification_cv",
]

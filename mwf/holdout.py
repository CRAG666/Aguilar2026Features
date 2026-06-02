"""Sealed held-out test-set utilities (temporal and subject-based)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit

from .constants import DEFAULT_SEED, DEFAULT_SPLIT_SEEDS
from .dataset import BiometricSegments

DEFAULT_TEST_FRACTION: Final[float] = 0.20


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    """Train/test partition of a :class:`BiometricSegments` cohort.

    Attributes:
        train: Training half.
        test: Held-out half.
        test_indices: Original indices of ``test`` within the source cohort.
    """

    train: BiometricSegments
    test: BiometricSegments
    test_indices: NDArray[np.int64]

    @property
    def n_train(self) -> int:
        """Number of training segments."""
        return self.train.num_segments

    @property
    def n_test(self) -> int:
        """Number of held-out segments."""
        return self.test.num_segments


def _index_segments(segments: BiometricSegments, mask: NDArray[np.bool_]) -> BiometricSegments:
    """Materialise a sub-cohort from a boolean mask.

    Args:
        segments: Source cohort.
        mask: ``(n,)`` boolean selector.

    Returns:
        A :class:`BiometricSegments` containing only the masked rows.
    """
    return BiometricSegments(
        ecg=segments.ecg[mask],
        ppg=segments.ppg[mask],
        labels=segments.labels[mask],
        sampling_rate=segments.sampling_rate,
    )


def temporal_holdout_per_subject(
    segments: BiometricSegments,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> HoldoutSplit:
    """Hold out the last ``test_fraction`` of each subject's segments.

    Args:
        segments: Source cohort in acquisition order.
        test_fraction: Fraction in ``(0, 1)`` reserved per subject.

    Returns:
        A :class:`HoldoutSplit` keeping every subject in both halves.

    Raises:
        ValueError: If ``test_fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}.")
    test_mask = np.zeros(segments.num_segments, dtype=bool)
    for subj in np.unique(segments.labels):
        idx = np.flatnonzero(segments.labels == subj)
        n_test = max(1, int(np.ceil(idx.size * test_fraction)))
        test_mask[idx[-n_test:]] = True
    return HoldoutSplit(
        train=_index_segments(segments, ~test_mask),
        test=_index_segments(segments, test_mask),
        test_indices=np.flatnonzero(test_mask),
    )


def subject_holdout(
    segments: BiometricSegments,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> HoldoutSplit:
    """Hold out a random subset of subjects as a quasi-external cohort.

    Args:
        segments: Source cohort.
        test_fraction: Fraction of subjects in ``(0, 1)`` to hold out.
        seed: Seed for :class:`GroupShuffleSplit`.

    Returns:
        A :class:`HoldoutSplit` whose halves share no subjects.

    Raises:
        ValueError: If ``test_fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}.")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    placeholder = np.zeros(segments.num_segments, dtype=np.uint8)
    train_idx, _ = next(gss.split(placeholder, groups=segments.labels))
    train_mask = np.zeros(segments.num_segments, dtype=bool)
    train_mask[train_idx] = True
    return HoldoutSplit(
        train=_index_segments(segments, train_mask),
        test=_index_segments(segments, ~train_mask),
        test_indices=np.flatnonzero(~train_mask),
    )


def subject_holdout_multiseed(
    segments: BiometricSegments,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
) -> tuple[HoldoutSplit, ...]:
    """Run :func:`subject_holdout` once per seed.

    Args:
        segments: Source cohort.
        test_fraction: Fraction of subjects to hold out.
        seeds: Seeds to iterate over.

    Returns:
        Tuple with one :class:`HoldoutSplit` per seed.
    """
    return tuple(
        subject_holdout(segments, test_fraction=test_fraction, seed=s)
        for s in seeds
    )


__all__ = [
    "HoldoutSplit",
    "subject_holdout",
    "subject_holdout_multiseed",
    "temporal_holdout_per_subject",
]

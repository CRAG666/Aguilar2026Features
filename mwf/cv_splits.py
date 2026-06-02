"""Group-aware cross-validation splitters for biometric identification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)

from .constants import DEFAULT_SEGMENTS_PER_BLOCK


def temporal_block_groups(
    labels: NDArray[np.int64],
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
) -> NDArray[np.int64]:
    """Assign each segment a globally unique temporal-block ID.

    Within each subject, consecutive segments are chunked into blocks of
    ``segments_per_block`` and given a unique block ID.

    Args:
        labels: ``(n,)`` subject labels in acquisition order (each
            subject's segments must be contiguous).
        segments_per_block: Segments grouped into one block.

    Returns:
        ``(n,)`` array of block IDs unique across subjects.

    Raises:
        ValueError: If ``segments_per_block < 1`` or if ``labels`` is not
            ordered with each subject contiguous.
    """
    if segments_per_block < 1:
        raise ValueError("segments_per_block must be ≥ 1.")
    boundary = np.flatnonzero(np.diff(labels) != 0)
    runs = labels[np.r_[0, boundary + 1]]
    if runs.size != np.unique(runs).size:
        raise ValueError(
            "labels must be in acquisition order (each subject contiguous); "
            "shuffled labels would produce mis-grouped temporal blocks."
        )
    groups = np.empty(labels.shape[0], dtype=np.int64)
    next_block = 0
    for subj in np.unique(labels):
        idx = np.flatnonzero(labels == subj)
        for chunk_start in range(0, idx.size, segments_per_block):
            chunk = idx[chunk_start : chunk_start + segments_per_block]
            groups[chunk] = next_block
            next_block += 1
    return groups


def stratified_group_splitter(
    n_splits: int = 5,
    random_state: int | None = None,
    shuffle: bool = True,
) -> StratifiedGroupKFold:
    """Build a pre-configured :class:`StratifiedGroupKFold` for closed-set CV.

    Args:
        n_splits: Number of folds.
        random_state: Seed for shuffling.
        shuffle: Whether to shuffle groups before splitting.

    Returns:
        Configured :class:`StratifiedGroupKFold` instance.
    """
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


__all__ = [
    "GroupKFold",
    "LeaveOneGroupOut",
    "StratifiedGroupKFold",
    "stratified_group_splitter",
    "temporal_block_groups",
]

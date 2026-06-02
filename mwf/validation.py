"""Shared input validators for batch-array routines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ensure_2d_batch(arr: NDArray[np.float64], name: str = "batch") -> None:
    """Validate that ``arr`` is a 2-D ``(B, N)`` array.

    Args:
        arr: Array to inspect.
        name: Label used in the error message.

    Raises:
        ValueError: If ``arr`` does not have exactly two dimensions.
    """
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D (B, N); got shape {arr.shape}.")


__all__ = ["ensure_2d_batch"]

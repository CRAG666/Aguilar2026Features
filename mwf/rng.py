"""Random number generator factory used across the package."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Build a NumPy ``Generator`` seeded with ``seed``.

    Args:
        seed: Integer seed, or ``None`` for non-deterministic seeding.

    Returns:
        A ``numpy.random.Generator`` instance.
    """
    return np.random.default_rng(seed)


__all__ = ["make_rng"]

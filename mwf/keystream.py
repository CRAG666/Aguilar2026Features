"""Deterministic token-derived key derivation for the cancelable transform.

The cancelable BioHashing transform needs one property from its key material:
that it be *deterministic* and *sensitive to the token* — a one-bit token change
must yield an unrelated projection matrix, and hence an unrelated (renewable,
diverse) template (ISO/IEC 30136). It does **not** need a cryptographic-grade,
NIST-certifiable keystream: cancelability is not encryption, and the stored
object is a feature template, not ciphertext.

We therefore derive the random-projection generator seed directly from the
SHA-256 digest of the token and let :class:`numpy.random.SeedSequence` expand it
into the generator the transform draws its random matrix from. SHA-256 already
provides the avalanche (one flipped token bit ⇒ a fully different 256-bit digest
⇒ a decorrelated projection), so no chaotic map and no SHAKE-256 whitening stage
are involved. The transform (:mod:`mwf.feature_transform`) consumes this
generator.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

_SEED_BYTES: Final[int] = 32  # SHA-256 digest width → 256-bit SeedSequence entropy


def keystream_rng(token: str) -> np.random.Generator:
    """Build a deterministic, token-derived NumPy ``Generator``.

    The SHA-256 digest of ``token`` seeds a :class:`numpy.random.SeedSequence`,
    so a one-bit token edit yields an unrelated generator state — and hence a
    decorrelated random projection. The transform draws its projection matrix
    from the returned generator.

    Args:
        token: Cancelable-template token (key ``k``).

    Returns:
        A deterministic :class:`numpy.random.Generator`.

    Raises:
        ValueError: If ``token`` is empty.
    """
    if not token:
        raise ValueError("Token must be a non-empty string.")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed_words = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(seed_words.tolist()))


__all__ = ["keystream_rng"]

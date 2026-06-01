"""Key-sensitivity check for the cancelable BioHashing transform.

Cancelability — not cryptography — is the design goal here: the stored object is
a feature template, not ciphertext. This module therefore keeps only the
template-level sensitivity measure relevant to renewability / diversity: how much
a cancelable template changes when the token is edited by a single bit (bit error
rate plus residual correlation).

The image-cipher metrics (NPCR/UACI, adjacent-sample correlation, histogram χ²,
Shannon entropy) and the cryptographic keystream certification (NIST SP 800-22,
Lyapunov exponents, chaotic key-space accounting) are intentionally absent: they
answer encryption questions this system does not pose. The ISO/IEC 30136
cancelability criteria — renewability, diversity and unlinkability — live in
:mod:`mwf.cancelability`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from .stats_helpers import std_or_zero

DEFAULT_QUANTISATION_BITS: Final[int] = 8
_RANGE_EPSILON: Final[float] = 1e-12  # below this, treat lo ≈ hi as degenerate

TransformFn = Callable[[str], NDArray[np.float64]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantise_with_range(
    signal: NDArray[np.float64],
    lo: float,
    hi: float,
    bits: int = DEFAULT_QUANTISATION_BITS,
) -> NDArray[np.uint8]:
    """Uniformly quantise ``signal`` onto ``[0, 2^bits − 1]`` over ``[lo, hi]``.

    Args:
        signal: Float array; flattened before quantisation.
        lo: Lower bound of the shared grid.
        hi: Upper bound of the shared grid.
        bits: Quantisation depth.

    Returns:
        ``uint8`` (``bits ≤ 8``) or ``uint16`` (``bits > 8``) samples;
        zeros when ``hi − lo`` is below :data:`_RANGE_EPSILON`.
    """
    levels = (1 << bits) - 1
    s = np.asarray(signal, dtype=np.float64).ravel()
    dtype = np.uint8 if bits <= 8 else np.uint16
    if hi - lo < _RANGE_EPSILON:
        return np.zeros_like(s, dtype=dtype)
    norm = (s - lo) / (hi - lo)
    return np.clip(np.round(norm * levels), 0, levels).astype(dtype)


def _quantise(signal: NDArray[np.float64], bits: int = DEFAULT_QUANTISATION_BITS) -> NDArray[np.uint8]:
    """Quantise ``signal`` to ``[0, 2^bits − 1]`` using its own min/max.

    Args:
        signal: 1-D array of floats.
        bits: Quantisation depth.

    Returns:
        Quantised samples as ``uint8`` or ``uint16`` depending on ``bits``.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    return _quantise_with_range(s, float(s.min()), float(s.max()), bits)


def _flip_one_bit(text: str, bit_idx: int = 0) -> str:
    """Return a copy of ``text`` with the ``bit_idx``-th UTF-8 bit flipped.

    Args:
        text: Source string.
        bit_idx: Zero-based bit index within the encoded byte stream.

    Returns:
        Mutated string, decoded as UTF-8 with a latin-1 fallback.
    """
    raw = bytearray(text.encode("utf-8"))
    byte, bit = divmod(bit_idx, 8)
    raw[byte] ^= 1 << bit
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# ---------------------------------------------------------------------------
# Key sensitivity (BER + cross-correlation under 1-bit token edits)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KeySensitivityReport:
    """Aggregate key-sensitivity statistics over 1-bit token edits.

    A renewable cancelable template must change substantially when its token
    changes: an edited token should yield a decorrelated template (ISO/IEC
    30136 diversity). This report quantifies that template-level sensitivity.

    Attributes:
        n_trials: Successful (non-collapsing) edits.
        bit_error_rate_mean: Mean BER between base and edited template.
        bit_error_rate_std: Sample std of the BER.
        correlation_mean: Mean ABSOLUTE Pearson correlation ``|r|`` between
            the base and edited templates. The absolute value is taken per
            trial before averaging — averaging signed ``r`` would let
            opposite-sign correlations cancel and overstate decorrelation.
        correlation_std: Sample std of those ``|r|`` values.
    """

    n_trials: int
    bit_error_rate_mean: float
    bit_error_rate_std: float
    correlation_mean: float
    correlation_std: float


def _single_trial(
    base: NDArray[np.float64], base_q: NDArray[np.uint8],
    cipher: NDArray[np.float64], bits: int,
) -> tuple[float, float]:
    """Compute one BER and one ``|correlation|`` sample for a token edit.

    Args:
        base: Reference template.
        base_q: Quantised reference template.
        cipher: Template under the edited token.
        bits: Quantisation depth.

    Returns:
        Tuple ``(ber, abs_correlation)``.
    """
    cipher_q = _quantise(cipher, bits=bits)
    xor = np.unpackbits(np.bitwise_xor(base_q, cipher_q).astype(np.uint8))
    r, _ = pearsonr(base.ravel(), cipher.ravel())
    return float(xor.mean()), float(0.0 if np.isnan(r) else abs(r))


def key_sensitivity(
    transform_fn: TransformFn,
    base_password: str,
    n_trials: int = 32,
    bits: int = DEFAULT_QUANTISATION_BITS,
    seed: int = 0,
) -> KeySensitivityReport:
    """Estimate key-sensitivity statistics over ``n_trials`` independent edits.

    Each trial draws an **independent** base token (``f"{base_password}#{i}"``)
    and flips one **random** bit of it, so the trials are i.i.d. draws over the
    token space rather than a deterministic bit-walk of a single password. The
    old sequential-bit-flip design exhausted the short token's bits and produced
    correlated samples whose std understated the true variance; independent draws
    give an honest mean ± std for the renewability/diversity claim.

    Args:
        transform_fn: Token→template callable.
        base_password: Prefix anchoring the per-trial independent base tokens.
        n_trials: Number of independent 1-bit-edit trials.
        bits: Quantisation depth.
        seed: RNG seed for the per-trial random bit selection.

    Returns:
        A :class:`KeySensitivityReport` summarising the trials.

    Raises:
        ValueError: If every edit collapsed back to its base token.
    """
    rng = np.random.default_rng(seed)
    trials: list[tuple[float, float]] = []
    for i in range(n_trials):
        base_token = f"{base_password}#{i}"
        base = transform_fn(base_token)
        base_q = _quantise(base, bits=bits)
        n_bits = len(base_token.encode("utf-8")) * 8
        edited = _flip_one_bit(base_token, int(rng.integers(0, n_bits)))
        if edited == base_token:
            continue
        trials.append(_single_trial(base, base_q, transform_fn(edited), bits))
    if not trials:
        raise ValueError("Every 1-bit edit collapsed back to its base token.")
    bers, corrs = (np.asarray(col, dtype=np.float64) for col in zip(*trials))
    return KeySensitivityReport(
        n_trials=int(bers.size),
        bit_error_rate_mean=float(bers.mean()),
        bit_error_rate_std=std_or_zero(bers),
        correlation_mean=float(corrs.mean()),
        correlation_std=std_or_zero(corrs),
    )


__all__ = [
    "KeySensitivityReport",
    "TransformFn",
    "key_sensitivity",
]

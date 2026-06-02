"""Tests for the SHA-256-seeded cancelable key-derivation RNG."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.keystream import keystream_rng


def test_keystream_rng_deterministic_and_token_sensitive():
    a = keystream_rng("USER_A").random(64)
    assert np.array_equal(a, keystream_rng("USER_A").random(64))  # deterministic
    assert not np.allclose(a, keystream_rng("USER_B").random(64))  # token-sensitive


def test_one_bit_token_edit_decorrelates_the_stream():
    """Avalanche: a one-character token edit yields an unrelated stream.

    This is the renewability / diversity property the cancelable transform
    relies on — SHA-256 supplies the diffusion, no chaotic map required.
    """
    a = keystream_rng("USER_HELLO").random(4096)
    b = keystream_rng("USER_HELLp").random(4096)
    r = float(np.corrcoef(a, b)[0, 1])
    assert abs(r) < 0.1


def test_empty_token_rejected():
    with pytest.raises(ValueError):
        keystream_rng("")

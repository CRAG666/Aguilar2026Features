"""Multimodal (ECG + PPG) wavelet statistical feature extraction.

This is the *Extracción* stage of the cancelable pipeline (see the architecture
diagram): each pre-processed ECG and PPG segment is decomposed with the DWT and
summarised by per-subband statistical descriptors, then the two modalities are
concatenated into a single feature vector ``x`` that the cancelable transform
(:mod:`mwf.feature_transform`) protects.

The per-segment, per-modality descriptors are produced by the shared extractor
``shared.features.wavelet.extract_wavelet_features`` — the same one used by the
``multimodal_biometric_auth`` experiment — so the feature definition is shared
across projects and lives in exactly one place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pywt
from numpy.typing import NDArray

# The shared feature extractor lives at the project root; make it importable
# before the top-level import below (mirrors mwf.dataset's bootstrap so joblib
# workers that re-import this module also find it).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.features.wavelet import extract_wavelet_features  # noqa: E402

from .batch_utils import parallel_row_map  # noqa: E402
from .constants import DWT_DEFAULT_LEVEL, FEATURE_WAVELET  # noqa: E402
from .validation import ensure_2d_batch  # noqa: E402

DEFAULT_WAVELET: Final[str] = FEATURE_WAVELET
DEFAULT_LEVEL: Final[int] = DWT_DEFAULT_LEVEL
# ``shared.features.wavelet`` emits 13 descriptors per subband (mean, std, var,
# kurtosis, skewness, energy, entropy, max, min, median, iqr, range, mad).
STATS_PER_BAND: Final[int] = 13
N_MODALITIES: Final[int] = 2  # ECG + PPG
_SIGNAL_NAMES: Final[tuple[str, str]] = ("ECG", "PPG")
# Sanitisation bounds mirror shared.features.FeatureExtractor so degenerate
# (e.g. all-zero) segments cannot inject NaN/Inf into the feature matrix.
_POSINF: Final[float] = 1e6
_NEGINF: Final[float] = -1e6


def _as_writable_float64(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``signal`` as a writable float64 array, copying only if needed.

    ``pywt.wavedec`` requires a writable buffer even though it only reads. On
    the joblib parallel path large chunks are memory-mapped read-only into the
    workers, so a slice handed to the shared extractor would otherwise raise
    ``ValueError: buffer source array is read-only``. The copy is taken only
    when the input is non-writable, so the writable single-segment path keeps
    its zero-copy fast lane.

    Args:
        signal: 1-D input segment.

    Returns:
        A writable float64 view (or copy) of ``signal``.
    """
    data = np.asarray(signal, dtype=np.float64)
    return data if data.flags.writeable else data.copy()


def max_feature_level(segment_length: int, wavelet: str = DEFAULT_WAVELET) -> int:
    """Return the deepest DWT level a segment of ``segment_length`` supports.

    This is the upper bound for the ``level`` argument of :func:`extract_features`:
    ``pywt.wavedec`` cannot decompose deeper than the wavelet's filter length
    allows for the given signal length. Sweeping ``range(1, max_feature_level(...)
    + 1)`` therefore covers every usable decomposition depth.

    Args:
        segment_length: Per-modality segment length in samples (must be ≥ 1).
        wavelet: Wavelet family whose decomposition filter length sets the bound.

    Returns:
        The maximum usable decomposition depth (``≥ 0``; ``0`` only for segments
        too short to decompose once).

    Raises:
        ValueError: If ``segment_length < 1``.
    """
    if segment_length < 1:
        raise ValueError("segment_length must be ≥ 1.")
    # ``pywt.Wavelet`` is absent from pywt's type stubs though present at runtime.
    dec_len = cast(Any, pywt).Wavelet(wavelet).dec_len
    return pywt.dwt_max_level(segment_length, dec_len)


def feature_dimension(level: int) -> int:
    """Return the multimodal feature-vector length at a given DWT depth.

    A level-``L`` decomposition yields ``L + 1`` subbands, each summarised by
    :data:`STATS_PER_BAND` descriptors, for each of the :data:`N_MODALITIES`
    modalities.

    Args:
        level: DWT decomposition depth (must be ≥ 1).

    Returns:
        ``N_MODALITIES * STATS_PER_BAND * (level + 1)``.

    Raises:
        ValueError: If ``level < 1``.
    """
    if level < 1:
        raise ValueError("level must be ≥ 1.")
    return N_MODALITIES * STATS_PER_BAND * (level + 1)


def extract_features(
    ecg: NDArray[np.float64],
    ppg: NDArray[np.float64],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> NDArray[np.float64]:
    """Return the concatenated ECG‖PPG wavelet-statistic vector for one segment.

    Args:
        ecg: 1-D ECG segment.
        ppg: 1-D PPG segment (same length as ``ecg``).
        wavelet: Wavelet family forwarded to the shared extractor.
        level: DWT decomposition depth (must be ≥ 1).

    Returns:
        Length-``feature_dimension(level)`` vector: the ECG descriptors first
        (approximation band, then details coarse-to-fine), then the PPG
        descriptors in the same order. NaN/Inf values produced by degenerate
        segments are sanitised to finite numbers.

    Raises:
        ValueError: If ``ecg``/``ppg`` are not 1-D or differ in length.
    """
    if ecg.ndim != 1 or ppg.ndim != 1:
        raise ValueError("ecg and ppg must be 1-D arrays.")
    if ecg.shape[0] != ppg.shape[0]:
        raise ValueError(
            f"ECG/PPG length mismatch: {ecg.shape[0]} vs {ppg.shape[0]}."
        )
    feats_ecg = extract_wavelet_features(
        _as_writable_float64(ecg), _SIGNAL_NAMES[0],
        wavelet=wavelet, level=level,
    )
    feats_ppg = extract_wavelet_features(
        _as_writable_float64(ppg), _SIGNAL_NAMES[1],
        wavelet=wavelet, level=level,
    )
    values = np.fromiter(
        (*feats_ecg.values(), *feats_ppg.values()),
        dtype=np.float64,
        count=len(feats_ecg) + len(feats_ppg),
    )
    return np.nan_to_num(values, nan=0.0, posinf=_POSINF, neginf=_NEGINF)


def feature_names(
    segment_length: int,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> list[str]:
    """Return the ordered feature names matching :func:`extract_features`.

    The names depend only on the descriptor layout, not on the sample values,
    so they are read off a zero probe of the right length.

    Args:
        segment_length: Length of one segment (needed to probe the DWT layout).
        wavelet: Wavelet family forwarded to the shared extractor.
        level: DWT decomposition depth.

    Returns:
        ``feature_dimension(level)`` column names, ECG block then PPG block.
    """
    probe = np.zeros(segment_length, dtype=np.float64)
    names_ecg = list(extract_wavelet_features(
        probe, _SIGNAL_NAMES[0], wavelet=wavelet, level=level,
    ))
    names_ppg = list(extract_wavelet_features(
        probe, _SIGNAL_NAMES[1], wavelet=wavelet, level=level,
    ))
    return names_ecg + names_ppg


def _features_chunk(
    chunk: NDArray[np.float64],
    n_samples: int,
    wavelet: str,
    level: int,
) -> NDArray[np.float64]:
    """Extract features for every row of an ``hstack([ecg, ppg])`` chunk.

    Args:
        chunk: ``(k, 2 * n_samples)`` slab; columns ``[:n_samples]`` are ECG,
            ``[n_samples:]`` are PPG.
        n_samples: Per-modality segment length (the split point).
        wavelet: Forwarded to :func:`extract_features`.
        level: Forwarded to :func:`extract_features`.

    Returns:
        ``(k, feature_dimension(level))`` array of stacked feature vectors.
    """
    dim = feature_dimension(level)
    out = np.empty((chunk.shape[0], dim), dtype=np.float64)
    for row in range(chunk.shape[0]):
        ecg_row = chunk[row, :n_samples]
        ppg_row = chunk[row, n_samples:]
        out[row] = extract_features(ecg_row, ppg_row, wavelet=wavelet, level=level)
    return out


def extract_features_batch(
    ecg: NDArray[np.float64],
    ppg: NDArray[np.float64],
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    n_jobs: int | None = None,
) -> NDArray[np.float64]:
    """Extract multimodal features for every segment of a ``(B, N)`` batch.

    Args:
        ecg: ``(B, N)`` float64 ECG batch.
        ppg: ``(B, N)`` float64 PPG batch aligned with ``ecg``.
        wavelet: Forwarded to :func:`extract_features`.
        level: DWT decomposition depth (must be ≥ 1).
        n_jobs: Worker count forwarded to :func:`parallel_row_map`.

    Returns:
        ``(B, feature_dimension(level))`` array in input row order.

    Raises:
        ValueError: If ``ecg``/``ppg`` are not 2-D or have mismatched shapes.
    """
    ensure_2d_batch(ecg, name="ecg")
    ensure_2d_batch(ppg, name="ppg")
    if ecg.shape != ppg.shape:
        raise ValueError(f"Shape mismatch: ECG {ecg.shape} vs PPG {ppg.shape}.")

    n_samples = ecg.shape[1]
    stacked = np.hstack([ecg, ppg])

    def _worker(chunk: NDArray[np.float64]) -> NDArray[np.float64]:
        return _features_chunk(chunk, n_samples, wavelet, level)

    return parallel_row_map(stacked, _worker, n_jobs=n_jobs)


__all__ = [
    "N_MODALITIES",
    "STATS_PER_BAND",
    "extract_features",
    "extract_features_batch",
    "feature_dimension",
    "feature_names",
    "max_feature_level",
]

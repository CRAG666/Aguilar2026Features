"""MIMIC-100 dataset wrapper that yields 6-second ECG/PPG segments at 125 Hz."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

# Project root (the directory that owns the ``Datasets`` and ``shared``
# packages) is three levels up from this file:
#   <root>/cancelable_biometric_signals/aguilar2026features/mwf/dataset.py
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Datasets.MIMIC_100.load import MIMICDatasetLoader  # noqa: E402

logger = logging.getLogger(__name__)

SEGMENT_DURATION_SECONDS: Final[int] = 6
SAMPLING_RATE_HZ: Final[int] = 125
SEGMENT_LENGTH_SAMPLES: Final[int] = SEGMENT_DURATION_SECONDS * SAMPLING_RATE_HZ


@dataclass(frozen=True, slots=True)
class BiometricSegments:
    """Parallel ECG/PPG segments with subject labels.

    Attributes:
        ecg: ``(B, N)`` array of ECG segments.
        ppg: ``(B, N)`` array of PPG segments aligned with ``ecg``.
        labels: ``(B,)`` subject identifiers per segment.
        sampling_rate: Sampling frequency in Hz.
    """

    ecg: NDArray[np.float64]
    ppg: NDArray[np.float64]
    labels: NDArray[np.int64]
    sampling_rate: int

    def __post_init__(self) -> None:
        """Validate that ECG, PPG and labels have consistent shapes.

        Raises:
            ValueError: If ECG/PPG shapes differ or label count mismatches.
        """
        if self.ecg.shape != self.ppg.shape:
            raise ValueError(f"ECG/PPG shape mismatch: {self.ecg.shape} vs {self.ppg.shape}")
        if self.ecg.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Number of segments must match the number of labels: "
                f"{self.ecg.shape[0]} vs {self.labels.shape[0]}"
            )

    @property
    def num_segments(self) -> int:
        """Total number of segments in the bundle."""
        return int(self.ecg.shape[0])

    @property
    def num_subjects(self) -> int:
        """Number of distinct subject identifiers."""
        return int(np.unique(self.labels).size)

    @property
    def segment_length(self) -> int:
        """Number of samples per segment."""
        return int(self.ecg.shape[1])


def load_mimic100(n_jobs: int = -1) -> BiometricSegments:
    """Load MIMIC-100 as fixed-length ECG/PPG segments.

    Args:
        n_jobs: Worker count forwarded to the underlying loader (``-1`` = all).

    Returns:
        A :class:`BiometricSegments` bundle with ECG, PPG, labels and
        sampling rate.
    """
    logger.info("Loading MIMIC-100 (segment_duration=%ds).", SEGMENT_DURATION_SECONDS)
    loader = MIMICDatasetLoader(segment_duration=SEGMENT_DURATION_SECONDS, n_jobs=n_jobs)
    raw = loader.get_segments()

    ecg = np.asarray(raw["ECG"], dtype=np.float64)
    ppg = np.asarray(raw["PPG"], dtype=np.float64)
    labels = np.asarray(raw["labels"], dtype=np.int64)

    logger.info(
        "Loaded %d segments from %d subjects (segment length = %d samples).",
        ecg.shape[0],
        np.unique(labels).size,
        ecg.shape[1],
    )
    return BiometricSegments(ecg=ecg, ppg=ppg, labels=labels, sampling_rate=loader.sampling_rate)


__all__ = [
    "SAMPLING_RATE_HZ",
    "SEGMENT_DURATION_SECONDS",
    "SEGMENT_LENGTH_SAMPLES",
    "BiometricSegments",
    "load_mimic100",
]

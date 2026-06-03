"""MIMIC-100 dataset wrapper that yields 6-second ECG/PPG segments at 125 Hz."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

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


class _SegmentLoader(Protocol):
    """The slice of a ``Datasets`` loader that :func:`segments_from_loader` uses."""

    def sampling_rate_of(self, signal: str) -> int: ...
    def get_signal_segments(self, signal: str) -> dict[str, np.ndarray]: ...


def segments_from_loader(loader: _SegmentLoader) -> BiometricSegments:
    """Adapt a ``Datasets`` loader's ECG and PPG channels into a bundle.

    Reads only ECG and PPG (ignoring any extra channel such as BIDMC's
    respiration) and checks their per-segment labels agree.

    Args:
        loader: A ``Datasets`` loader whose signals include ``"ECG"`` and
            ``"PPG"`` at a shared sampling rate.

    Returns:
        The loader's ECG/PPG segments, subject labels and sampling rate.

    Raises:
        ValueError: If ECG and PPG use different rates or disagree on labels.
    """
    ecg_rate = loader.sampling_rate_of("ECG")
    ppg_rate = loader.sampling_rate_of("PPG")
    if ecg_rate != ppg_rate:
        raise ValueError(
            f"ECG ({ecg_rate} Hz) and PPG ({ppg_rate} Hz) must share a sampling "
            "rate for the multimodal pipeline; resample before adapting."
        )
    ecg_seg = loader.get_signal_segments("ECG")
    ppg_seg = loader.get_signal_segments("PPG")
    if not np.array_equal(ecg_seg["labels"], ppg_seg["labels"]):
        raise ValueError("ECG and PPG produced different per-segment labels.")
    ecg = np.asarray(ecg_seg["ECG"], dtype=np.float64)
    ppg = np.asarray(ppg_seg["PPG"], dtype=np.float64)
    labels = np.asarray(ecg_seg["labels"], dtype=np.int64)
    logger.info(
        "Adapted %d segments from %d subjects (%d samples @ %d Hz) from %s.",
        ecg.shape[0], np.unique(labels).size, ecg.shape[1], ecg_rate,
        type(loader).__name__,
    )
    return BiometricSegments(ecg=ecg, ppg=ppg, labels=labels, sampling_rate=int(ecg_rate))


def load_bidmc(
    segment_duration: int = SEGMENT_DURATION_SECONDS, n_jobs: int = -1,
) -> BiometricSegments:
    """Load BIDMC as fixed-length ECG/PPG segments (lead-II ECG + PLETH @ 125 Hz).

    A 53-subject external cohort at the same 125 Hz and default window as
    MIMIC-100, so the feature pipeline applies unchanged.

    Args:
        segment_duration: Segment length in seconds.
        n_jobs: Worker count (``-1`` = all cores).

    Returns:
        A :class:`BiometricSegments` bundle for BIDMC.
    """
    from Datasets.BIDMC.load import BIDMCDatasetLoader

    logger.info("Loading BIDMC (segment_duration=%ds).", segment_duration)
    loader = BIDMCDatasetLoader(segment_duration=segment_duration, n_jobs=n_jobs)
    return segments_from_loader(loader)


def load_ptt_ppg(
    activity: str,
    segment_duration: int = SEGMENT_DURATION_SECONDS,
    n_jobs: int = -1,
) -> BiometricSegments:
    """Load one PTT-PPG activity as ECG/PPG segments (single-lead ECG + finger PPG @ 500 Hz).

    Each of the 22 subjects records ``sit``/``walk``/``run`` as a separate file,
    so a per-activity bundle is the unit for the cross-activity protocol.

    Args:
        activity: One of ``"sit"``, ``"walk"`` or ``"run"``.
        segment_duration: Segment length in seconds.
        n_jobs: Worker count (``-1`` = all cores).

    Returns:
        A :class:`BiometricSegments` bundle for that activity.

    Raises:
        ValueError: If ``activity`` is not a PTT-PPG activity.
    """
    from Datasets.PTT_PPG.load import ACTIVITIES, PTTPPGDatasetLoader

    if activity not in ACTIVITIES:
        raise ValueError(f"Unknown PTT-PPG activity {activity!r}; choose from {ACTIVITIES}.")
    logger.info("Loading PTT-PPG activity=%r (segment_duration=%ds).", activity, segment_duration)
    loader = PTTPPGDatasetLoader(activity, segment_duration=segment_duration, n_jobs=n_jobs)
    return segments_from_loader(loader)


PTT_PPG_ACTIVITIES: Final[tuple[str, ...]] = ("sit", "walk", "run")


__all__ = [
    "PTT_PPG_ACTIVITIES",
    "SAMPLING_RATE_HZ",
    "SEGMENT_DURATION_SECONDS",
    "SEGMENT_LENGTH_SAMPLES",
    "BiometricSegments",
    "load_bidmc",
    "load_mimic100",
    "load_ptt_ppg",
    "segments_from_loader",
]

"""Centralised constants and configuration objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DEFAULT_SEED: Final[int] = 42
DEFAULT_N_FOLDS: Final[int] = 5
DEFAULT_SEGMENTS_PER_BLOCK: Final[int] = 2
# Repeated-CV seeds: each seed reshuffles the k-fold partition, and the
# Nadeau-Bengio CI aggregates over all seed×fold observations. Three seeds
# (15 folds at k=5) already give a stable mean and a tight corrected CI; the
# variance reduction from a 4th/5th seed is marginal and not worth the ~linear
# runtime cost of the nested-CV tuning grid. Override with --split-seeds to
# restore the previous 5-seed protocol when extra statistical power is wanted.
DEFAULT_SPLIT_SEEDS: Final[tuple[int, ...]] = (42, 43, 44)

# ---------------------------------------------------------------------------
# Wavelet / DWT configuration
# ---------------------------------------------------------------------------
# FEATURE_WAVELET drives the multi-scale statistical extractor
# (shared.features.wavelet). The biorthogonal bior3.3 basis has a symmetric
# (linear-phase) reconstruction filter that does not shift the ECG/PPG fiducial
# points, so the per-subband statistics describe morphology without the small
# phase distortion the orthogonal Daubechies bases introduce.
# DWT_DEFAULT_LEVEL fixes the decomposition depth of that extractor.

FEATURE_WAVELET: Final[str] = "bior3.3"
DWT_DEFAULT_LEVEL: Final[int] = 4

# ---------------------------------------------------------------------------
# NeuroKit physiological cleaning (mwf.clean) — default signal front-end
# ---------------------------------------------------------------------------
# Modality-specific cleaners replace the modality-agnostic VisuShrink denoiser.
# ECG "neurokit": 0.5 Hz high-pass Butterworth + 50 Hz powerline filtering.
# PPG "elgendi":  0.5-8 Hz band-pass Butterworth. Both forwarded verbatim to
# ``neurokit2.ecg_clean`` / ``neurokit2.ppg_clean`` with the cohort sampling rate.
DEFAULT_ECG_CLEAN_METHOD: Final[str] = "neurokit"
DEFAULT_PPG_CLEAN_METHOD: Final[str] = "elgendi"

# ---------------------------------------------------------------------------
# BioHashing cancelable-transform defaults
# ---------------------------------------------------------------------------
# The cancelable transform is a token-keyed random orthonormal projection
# (BioHashing). PROJECTION_RATIO = m / d fixes the protected-template length
# m as a fraction of the extracted feature dimension d. Keeping m < d is what
# makes the projection many-to-one and hence non-invertible: the (d - m)-
# dimensional null space of the projection is irrecoverable even with the key.

DEFAULT_PROJECTION_RATIO: Final[float] = 0.5
DEFAULT_BINARISE: Final[bool] = False
# Standardise the feature vector (per-feature z-score) *before* the projection.
# The wavelet descriptors live on wildly different scales (sub-band energy can
# dwarf entropy by orders of magnitude); on the raw vector a single random
# projection is dominated by the high-variance descriptors and buries the
# discriminative low-variance ones. Equalising scales first lets the projection
# preserve discriminability — decisive under a shared token, where every subject
# is mapped by the *same* R and the projection cannot adapt per user. The
# statistics are non-secret enrolment-time parameters (same status as the
# wavelet basis), not part of the token.
DEFAULT_STANDARDIZE: Final[bool] = True

# ---------------------------------------------------------------------------
# Index-of-Max (IoM) hashing for the PPG block (mwf.iom)
# ---------------------------------------------------------------------------
# The multimodal template protects the two modalities differently:
#   * ECG block → the orthonormal BioHashing projection above (real-valued);
#   * PPG block → Index-of-Max hashing (Jin et al., IEEE TIFS 2018).
# IoM keeps only the *index of the maximum* of each of m token-seeded Gaussian
# projections, discarding magnitudes. This is what makes the PPG template both
# (a) strongly non-invertible — there is no linear min-norm pre-image, so the
# ~sqrt(m/d) leakage of a plain projection collapses — and (b) key-not-learnable
# in the worst case: IoM is a similarity-preserving LSH whose collision rate
# depends only on the angle between feature vectors, *not* on the token, so the
# biometric (not the key) carries the discriminability even under a stolen token.
# IOM_WINDOW = q is the number of Gaussian projections per hash (argmax over q);
# IOM_HASHES_RATIO sets the code length m as a multiple of the PPG block size.
#
# These defaults sit in the favourable corner of the privacy-utility trade-off
# (validated by a parameter sweep): a best-effort, token-aware inversion of the
# IoM code recovers the PPG feature *direction* with a correlation that GROWS
# with m and q (more hashes ⇒ a better averaged direction estimate). A small m
# (ratio 0.25) with a moderate window keeps that leakage below the ~0.78 of the
# plain linear projection while preserving genuine/impostor separability. Note
# the leak cannot be driven to zero without destroying utility — angular
# similarity preservation IS directional information; the operative cancelability
# guarantees are unlinkability, magnitude/exact-preimage non-invertibility, and
# the key-independence of the LSH similarity.
IOM_WINDOW: Final[int] = 16
IOM_HASHES_RATIO: Final[float] = 0.25

# ---------------------------------------------------------------------------
# Bootstrap / statistics defaults
# ---------------------------------------------------------------------------

BOOTSTRAP_CI_LEVEL: Final[float] = 0.95
BOOTSTRAP_RESAMPLES_EVAL: Final[int] = 2000     # for CV metric CIs


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Feature-extraction and BioHashing hyperparameters for every entry point.

    Attributes:
        feature_level: DWT decomposition level for the statistical extractor.
        feature_wavelet: Wavelet family name for feature extraction.
        projection_ratio: Protected-template length as a fraction of the
            extracted feature dimension (``m / d``); must lie in ``(0, 1]``.
        binarise: If ``True``, sign-binarise the projection into a ``±1`` bit
            template (classic BioHashing); otherwise keep the real-valued
            projection.
        standardize: If ``True``, per-feature z-score the feature vector on the
            enrolment cohort before the BioHashing projection (see
            :data:`DEFAULT_STANDARDIZE`).
    """

    feature_level: int = DWT_DEFAULT_LEVEL
    feature_wavelet: str = FEATURE_WAVELET
    projection_ratio: float = DEFAULT_PROJECTION_RATIO
    binarise: bool = DEFAULT_BINARISE
    standardize: bool = DEFAULT_STANDARDIZE


DEFAULT_PIPELINE_CONFIG: Final[PipelineConfig] = PipelineConfig()


__all__ = [
    "BOOTSTRAP_CI_LEVEL",
    "BOOTSTRAP_RESAMPLES_EVAL",
    "DEFAULT_BINARISE",
    "DEFAULT_ECG_CLEAN_METHOD",
    "DEFAULT_N_FOLDS",
    "DEFAULT_PIPELINE_CONFIG",
    "DEFAULT_PPG_CLEAN_METHOD",
    "DEFAULT_PROJECTION_RATIO",
    "DEFAULT_SEED",
    "DEFAULT_SEGMENTS_PER_BLOCK",
    "DEFAULT_SPLIT_SEEDS",
    "DEFAULT_STANDARDIZE",
    "DWT_DEFAULT_LEVEL",
    "FEATURE_WAVELET",
    "IOM_HASHES_RATIO",
    "IOM_WINDOW",
    "PipelineConfig",
]

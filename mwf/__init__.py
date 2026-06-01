"""Cancelable multimodal biometric system: BioHashing on wavelet features.

Variant of Aguilar2026 in which the cancelable transform is a token-keyed random
orthonormal projection (BioHashing) applied to the *feature vector*, not to the
waveform. The pipeline follows the canonical cancelable-biometrics architecture:

    Sensor → Preprocessing (AWGN→NeuroKit clean) → Extracción (multimodal ECG+PPG
    wavelet statistics, via ``shared.features.wavelet``) → Transformación
    cancelable T(x, k) (BioHashing) ← Token k → Plantilla transformada →
    Base de datos / Comparador → Decisión.

The transform is keyed for *cancelability* (renewability / diversity / non-
invertibility), not for cryptographic secrecy: SHA-256 supplies the token
avalanche, an orthonormal random projection supplies diffusion, and projecting
to ``m < d`` dimensions supplies the irreversibility (a destroyed null space).

Evaluation mirrors the signal-domain sibling project:
    * StratifiedGroupKFold CV with mean ± std + 95 % bootstrap / Nadeau-Bengio CIs,
    * three regimes (identity / single_key / per_subject),
    * cancelability protocol (renewability / diversity / Gomez-Barrero D_↔^sys),
    * template key-sensitivity (BER + correlation under 1-bit token edits),
    * inversion / leakage analysis of the random projection,
    * DET / CMC / operating points,
    * deterministic global seeding.
"""

from .constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_RESAMPLES_EVAL,
    BOOTSTRAP_RESAMPLES_TIMING,
    DEFAULT_BINARISE,
    DEFAULT_ECG_CLEAN_METHOD,
    DEFAULT_N_FOLDS,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_PPG_CLEAN_METHOD,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DEFAULT_SEGMENTS_PER_BLOCK,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
    PipelineConfig,
)
from .clean import (
    clean_ecg,
    clean_ecg_batch,
    clean_ppg,
    clean_ppg_batch,
)
from .cancelability import (
    CancelabilityReport,
    UnlinkabilityCurve,
    evaluate_cancelability,
)
from .cv_splits import (
    stratified_group_splitter,
    temporal_block_groups,
)
from .dataset import (
    SAMPLING_RATE_HZ,
    SEGMENT_DURATION_SECONDS,
    SEGMENT_LENGTH_SAMPLES,
    BiometricSegments,
    load_mimic100,
)
from .evaluation import (
    CV_STRATEGIES,
    METRIC_NAMES,
    MetricSummary,
    make_cv_splitter,
    set_global_seeds,
    summarise,
    summarise_run,
)
from .feature_transform import (
    FeatureScaler,
    ProjectionKey,
    derive_projection,
    multimodal_dims,
    projection_dim,
    transform_multimodal,
    transform_multimodal_batch,
)
from .iom import (
    derive_iom,
    iom_dim,
    iom_indices,
    iom_onehot,
)
from .features import (
    N_MODALITIES,
    STATS_PER_BAND,
    extract_features,
    extract_features_batch,
    feature_dimension,
    feature_names,
    max_feature_level,
)
from .holdout import (
    HoldoutSplit,
    subject_holdout,
    subject_holdout_multiseed,
    temporal_holdout_per_subject,
)
from .inversion import (
    InversionReport,
    multimodal_leakage_metrics,
    recover_ppg_iom,
)
from .keystream import keystream_rng
from .metrics import ClassificationMetrics, evaluate
from .non_invertibility import (
    NonInvertibilityReport,
    non_invertibility_analysis,
)
from .per_subject_ablation import (
    AblationPoint,
    per_subject_ablation,
)
from .ratio_sweep import (
    DEFAULT_RATIOS,
    RatioSweepPoint,
    ratio_sweep,
)
from .record_multiplicity import (
    INDEPENDENT,
    SHARED_SUBSPACE,
    RecordMultiplicityReport,
    record_multiplicity_leakage,
    revoked_projections,
)
from .noise import add_awgn_batch
from .operating_curves import (
    CmcCurve,
    DetCurve,
    PrCurve,
    RocCurve,
    cmc_curve,
    det_curve_from_scores,
    operating_points,
    pr_curve_from_scores,
    rank_k_accuracies,
    roc_curve_from_scores,
)
from .pipeline import (
    DEFAULT_SPLIT_SEEDS,
    CrossValidationResult,
    FoldEvaluation,
    KeyMode,
    TemplateBundle,
    build_templates,
    cross_validate_classifier_multiseed,
    make_pipeline,
    preprocess_signals,
)
from .rng import make_rng
from .significance import (
    ComparisonRow,
    compare_classifiers,
)
from .scoring import (
    compute_subject_centroids,
    cosine_score_matrix,
    decidability,
    genuine_impostor_scores,
    genuine_impostor_split,
    l2_normalise,
    znorm,
)
from .security import (
    KeySensitivityReport,
    key_sensitivity,
)
from .stats_helpers import (
    bootstrap_ci_mean,
    nadeau_bengio_ci_mean,
    nadeau_bengio_paired_t,
    std_or_zero,
)
from .stolen_token import (
    StolenTokenResult,
    stolen_token_score_pools,
    stolen_token_verification,
)
from .timing import TimingResult, benchmark
from .verification import (
    VerificationFoldResult,
    VerificationMode,
    VerificationResult,
    closed_set_score_pools,
    run_verification_cv,
)

__all__ = [
    "AblationPoint",
    "BOOTSTRAP_CI_LEVEL",
    "BOOTSTRAP_RESAMPLES_EVAL",
    "BOOTSTRAP_RESAMPLES_TIMING",
    "BiometricSegments",
    "CV_STRATEGIES",
    "CancelabilityReport",
    "ClassificationMetrics",
    "CmcCurve",
    "ComparisonRow",
    "CrossValidationResult",
    "DEFAULT_BINARISE",
    "DEFAULT_ECG_CLEAN_METHOD",
    "DEFAULT_N_FOLDS",
    "DEFAULT_PIPELINE_CONFIG",
    "DEFAULT_PPG_CLEAN_METHOD",
    "DEFAULT_PROJECTION_RATIO",
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "DEFAULT_SEGMENTS_PER_BLOCK",
    "DEFAULT_SPLIT_SEEDS",
    "DWT_DEFAULT_LEVEL",
    "DetCurve",
    "FEATURE_WAVELET",
    "FoldEvaluation",
    "HoldoutSplit",
    "InversionReport",
    "KeyMode",
    "KeySensitivityReport",
    "METRIC_NAMES",
    "MetricSummary",
    "N_MODALITIES",
    "NonInvertibilityReport",
    "PipelineConfig",
    "PrCurve",
    "FeatureScaler",
    "ProjectionKey",
    "RatioSweepPoint",
    "RocCurve",
    "SAMPLING_RATE_HZ",
    "SEGMENT_DURATION_SECONDS",
    "SEGMENT_LENGTH_SAMPLES",
    "STATS_PER_BAND",
    "StolenTokenResult",
    "TemplateBundle",
    "TimingResult",
    "UnlinkabilityCurve",
    "VerificationFoldResult",
    "VerificationMode",
    "VerificationResult",
    "add_awgn_batch",
    "benchmark",
    "bootstrap_ci_mean",
    "build_templates",
    "clean_ecg",
    "clean_ecg_batch",
    "clean_ppg",
    "clean_ppg_batch",
    "closed_set_score_pools",
    "cmc_curve",
    "compare_classifiers",
    "compute_subject_centroids",
    "cosine_score_matrix",
    "cross_validate_classifier_multiseed",
    "decidability",
    "derive_iom",
    "derive_projection",
    "det_curve_from_scores",
    "evaluate",
    "evaluate_cancelability",
    "extract_features",
    "extract_features_batch",
    "feature_dimension",
    "feature_names",
    "genuine_impostor_scores",
    "genuine_impostor_split",
    "iom_dim",
    "iom_indices",
    "iom_onehot",
    "key_sensitivity",
    "keystream_rng",
    "l2_normalise",
    "load_mimic100",
    "make_cv_splitter",
    "make_pipeline",
    "make_rng",
    "max_feature_level",
    "multimodal_dims",
    "multimodal_leakage_metrics",
    "non_invertibility_analysis",
    "per_subject_ablation",
    "ratio_sweep",
    "INDEPENDENT",
    "SHARED_SUBSPACE",
    "RecordMultiplicityReport",
    "record_multiplicity_leakage",
    "revoked_projections",
    "nadeau_bengio_ci_mean",
    "nadeau_bengio_paired_t",
    "operating_points",
    "pr_curve_from_scores",
    "preprocess_signals",
    "projection_dim",
    "rank_k_accuracies",
    "recover_ppg_iom",
    "roc_curve_from_scores",
    "run_verification_cv",
    "set_global_seeds",
    "std_or_zero",
    "stolen_token_score_pools",
    "stolen_token_verification",
    "stratified_group_splitter",
    "subject_holdout",
    "subject_holdout_multiseed",
    "summarise",
    "summarise_run",
    "temporal_block_groups",
    "temporal_holdout_per_subject",
    "transform_multimodal",
    "transform_multimodal_batch",
    "znorm",
]

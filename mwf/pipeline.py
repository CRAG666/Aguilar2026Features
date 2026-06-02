"""End-to-end pipeline: BioHashing cancelable templates and CV evaluation.

Order of operations (matching the architecture diagram):

    Sensor → Preprocessing (AWGN→NeuroKit clean) → Extracción (multimodal
    wavelet features) → Transformación cancelable T(x, k) (BioHashing) → Plantilla.

Unlike the signal-domain sibling project, the cancelable transform here acts on
the *feature vector*, not on the waveform: features are extracted first, then a
token-keyed random orthonormal projection protects them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Final

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .clean import clean_ecg_batch, clean_ppg_batch
from .constants import (
    DEFAULT_BINARISE,
    DEFAULT_ECG_CLEAN_METHOD,
    DEFAULT_N_FOLDS,
    DEFAULT_PPG_CLEAN_METHOD,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SEED,
    DEFAULT_SEGMENTS_PER_BLOCK,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_STANDARDIZE,
    DWT_DEFAULT_LEVEL,
    FEATURE_WAVELET,
    PipelineConfig,
)
from .cv_splits import stratified_group_splitter, temporal_block_groups
from .dataset import BiometricSegments
from .feature_transform import transform_multimodal_batch
from .features import extract_features_batch, feature_dimension
from .metrics import ClassificationMetrics, evaluate
from .noise import add_awgn_batch
from .operating_curves import DEFAULT_RANKS, rank_k_accuracies

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE: Final[int] = DEFAULT_SEED
# Environment-overridable parallelism for the (seed × fold) sweep in
# cross_validate_classifier_multiseed. ``1`` recovers sequential behaviour,
# ``-1`` (default) uses every logical core.
CV_N_JOBS: Final[int] = int(os.environ.get("AGUILAR_FEATURES_CV_N_JOBS", "-1"))
DEFAULT_FEATURE_LEVEL: Final[int] = DWT_DEFAULT_LEVEL
DEFAULT_FEATURE_WAVELET: Final[str] = FEATURE_WAVELET
DEFAULT_ECG_CLEAN: Final[str] = DEFAULT_ECG_CLEAN_METHOD
DEFAULT_PPG_CLEAN: Final[str] = DEFAULT_PPG_CLEAN_METHOD
SUBJECT_TOKEN_PREFIX: Final[str] = "USER_"
SHARED_TOKEN: Final[str] = "SHARED_KEY_FOR_BIOMETRIC_CEILING"


class KeyMode(str, Enum):
    """Key regime under which templates are built.

    Attributes:
        IDENTITY: No protection; the raw multimodal feature vector is stored
            (the unprotected biometric ceiling).
        SINGLE_KEY: BioHashing with one shared token for every subject.
        PER_SUBJECT: BioHashing with one token per subject (operational).
    """

    IDENTITY = "identity"
    SINGLE_KEY = "single_key"
    PER_SUBJECT = "per_subject"


@cache
def _label_to_letters(label: int) -> str:
    """Convert a positive integer label to spreadsheet-column letters.

    Args:
        label: Strictly positive subject identifier.

    Returns:
        ``"A"``, ``"B"`` … ``"Z"``, ``"AA"``, ``"AB"``, … encoding ``label``.

    Raises:
        ValueError: If ``label <= 0``.
    """
    if label <= 0:
        raise ValueError("Subject labels must be positive.")
    letters = []
    n = label
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _token_for_label(label: int) -> str:
    """Return the per-subject token for one label.

    Args:
        label: Subject identifier.

    Returns:
        Token string for use with the BioHashing transform.
    """
    return f"{SUBJECT_TOKEN_PREFIX}{_label_to_letters(label)}"


def _tokens_for(labels: NDArray[np.int64], key_mode: KeyMode) -> list[str] | None:
    """Build the list of tokens for a batch under the given key regime.

    Args:
        labels: ``(n,)`` subject labels.
        key_mode: Regime selector.

    Returns:
        List of tokens aligned with ``labels``, or ``None`` for
        :attr:`KeyMode.IDENTITY`.

    Raises:
        ValueError: If ``key_mode`` is not recognised.
    """
    match key_mode:
        case KeyMode.IDENTITY:
            return None
        case KeyMode.SINGLE_KEY:
            return [SHARED_TOKEN] * labels.shape[0]
        case KeyMode.PER_SUBJECT:
            return [_token_for_label(int(lbl)) for lbl in labels]
    raise ValueError(f"Unknown key_mode {key_mode!r}.")


def preprocess_signals(
    ecg: NDArray[np.float64],
    ppg: NDArray[np.float64],
    sampling_rate: int,
    snr_db: float | None = None,
    noise_seed: int = 0,
    denoise: bool = True,
    ecg_method: str = DEFAULT_ECG_CLEAN,
    ppg_method: str = DEFAULT_PPG_CLEAN,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the shared AWGN→NeuroKit-clean front-end to an ECG/PPG batch.

    Args:
        ecg: ``(B, N)`` ECG batch.
        ppg: ``(B, N)`` PPG batch aligned with ``ecg``.
        sampling_rate: Sampling frequency in Hz (forwarded to the NeuroKit
            cleaners; ignored when ``denoise`` is ``False``).
        snr_db: AWGN SNR injected pre-clean (``None`` to skip).
        noise_seed: Seed for AWGN injection (PPG uses ``noise_seed + 1``).
        denoise: Whether to run NeuroKit physiological cleaning.
        ecg_method: Method forwarded to ``neurokit2.ecg_clean``.
        ppg_method: Method forwarded to ``neurokit2.ppg_clean``.

    Returns:
        Tuple ``(ecg, ppg)`` after optional noise injection and cleaning.
    """
    if snr_db is not None:
        logger.info("Injecting AWGN at SNR = %.1f dB (seed=%d).", snr_db, noise_seed)
        ecg = add_awgn_batch(ecg, snr_db=snr_db, seed=noise_seed)
        ppg = add_awgn_batch(ppg, snr_db=snr_db, seed=noise_seed + 1)
    if denoise:
        ecg = clean_ecg_batch(ecg, sampling_rate=sampling_rate, method=ecg_method)
        ppg = clean_ppg_batch(ppg, sampling_rate=sampling_rate, method=ppg_method)
    return ecg, ppg


@dataclass(frozen=True, slots=True)
class TemplateBundle:
    """Feature templates produced by :func:`build_templates`.

    Attributes:
        features: ``(B, d)`` template matrix (raw features for IDENTITY, or
            the ``(B, m)`` BioHashing projection otherwise).
        labels: ``(B,)`` subject labels aligned with ``features``.
        key_mode: Regime under which the templates were built.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    key_mode: KeyMode


def build_templates(
    segments: BiometricSegments,
    feature_level: int = DEFAULT_FEATURE_LEVEL,
    feature_wavelet: str = DEFAULT_FEATURE_WAVELET,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    standardize: bool = DEFAULT_STANDARDIZE,
    snr_db: float | None = None,
    noise_seed: int = 0,
    denoise: bool = True,
    ecg_method: str = DEFAULT_ECG_CLEAN,
    ppg_method: str = DEFAULT_PPG_CLEAN,
    key_mode: KeyMode = KeyMode.PER_SUBJECT,
    config: PipelineConfig | None = None,
) -> TemplateBundle:
    """Run AWGN → clean → multimodal feature extraction → (BioHash | identity).

    Args:
        segments: ECG/PPG cohort.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing template length as a fraction ``m/d``.
        binarise: Whether to sign-binarise the BioHashing projection.
        standardize: Whether to per-feature z-score on the cohort before the
            projection (see :data:`DEFAULT_STANDARDIZE`). Ignored for
            :attr:`KeyMode.IDENTITY` (no projection). The scaler is fitted on
            the whole cohort, which is the enrolment population; for a strictly
            leakage-free per-fold estimate, fit a :class:`FeatureScaler` on the
            train split and pass it to :func:`transform_multimodal_batch` directly.
        snr_db: AWGN SNR injected pre-clean (``None`` to skip).
        noise_seed: Seed for AWGN injection.
        denoise: Whether to run NeuroKit physiological cleaning.
        ecg_method: Method forwarded to ``neurokit2.ecg_clean``.
        ppg_method: Method forwarded to ``neurokit2.ppg_clean``.
        key_mode: Key regime (see :class:`KeyMode`).
        config: Optional :class:`PipelineConfig` overriding the per-knob args.

    Returns:
        A :class:`TemplateBundle` ready for CV.
    """
    if config is not None:
        feature_level = config.feature_level
        feature_wavelet = config.feature_wavelet
        projection_ratio = config.projection_ratio
        binarise = config.binarise
        standardize = config.standardize

    ecg, ppg = preprocess_signals(
        segments.ecg,
        segments.ppg,
        sampling_rate=segments.sampling_rate,
        snr_db=snr_db,
        noise_seed=noise_seed,
        denoise=denoise,
        ecg_method=ecg_method,
        ppg_method=ppg_method,
    )

    # Extraction happens once, independent of the key regime: the cancelable
    # transform acts on the feature vector, not the signal.
    features = extract_features_batch(
        ecg, ppg, wavelet=feature_wavelet, level=feature_level,
    )

    if key_mode == KeyMode.IDENTITY:
        templates = features
    else:
        tokens = _tokens_for(segments.labels, key_mode)
        assert tokens is not None
        templates = transform_multimodal_batch(
            features,
            tokens,
            projection_ratio=projection_ratio,
            binarise=binarise,
            standardize=standardize,
        )

    logger.info(
        "Built %d templates [feat=%s J=%d | regime=%s | ratio=%.2f binarise=%s std=%s] → %d dims.",
        templates.shape[0],
        feature_wavelet,
        feature_level,
        key_mode.value,
        projection_ratio,
        binarise,
        standardize,
        templates.shape[1],
    )
    return TemplateBundle(features=templates, labels=segments.labels.copy(), key_mode=key_mode)


def _score_matrix(estimator: ClassifierMixin, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the ``predict_proba`` score matrix as ``float64``.

    Args:
        estimator: Fitted classifier exposing ``predict_proba``.
        x: ``(n, d)`` feature matrix.

    Returns:
        ``(n, n_classes)`` probability matrix.
    """
    return np.asarray(estimator.predict_proba(x), dtype=np.float64)


def make_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """Wrap ``classifier`` in a ``StandardScaler`` + classifier pipeline.

    Args:
        classifier: Classifier instance to use as the final step.

    Returns:
        A scikit-learn :class:`Pipeline`.
    """
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", classifier)])


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """All per-fold quantities produced in one fit/predict pass.

    Attributes:
        metrics: Macro classification metrics for the fold.
        rank_accuracies: ``{"rank_<k>_accuracy": value}`` for the requested ranks.
    """

    metrics: ClassificationMetrics
    rank_accuracies: dict[str, float]


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Aggregated CV result for one classifier and key regime.

    Attributes:
        classifier: Classifier name.
        n_features: Template dimensionality.
        key_mode: Key regime under which templates were built.
        fold_metrics: One :class:`ClassificationMetrics` per fold.
        n_folds_per_seed: Folds per CV repetition (the ``k`` in k-fold);
            ``0`` when unknown. Used by the Nadeau-Bengio CI correction.
        fold_extras: One ``{metric: value}`` dict per fold, aligned with
            ``fold_metrics``, holding the rank-``k`` draws of that fold.
            Empty when extras were not computed.
    """

    classifier: str
    n_features: int
    key_mode: KeyMode
    fold_metrics: tuple[ClassificationMetrics, ...]
    n_folds_per_seed: int = 0
    fold_extras: tuple[dict[str, float], ...] = ()

    @property
    def n_folds(self) -> int:
        """Total fold observations aggregated (``n_split_seeds × n_folds_per_seed``)."""
        return len(self.fold_metrics)

    def per_metric_values(self, name: str) -> NDArray[np.float64]:
        """Return the per-fold values of one metric.

        Args:
            name: Metric attribute name (e.g. ``"auc"``).

        Returns:
            1-D array of per-fold values.
        """
        return np.array([getattr(m, name) for m in self.fold_metrics], dtype=np.float64)

    def per_extra_values(self, name: str) -> NDArray[np.float64]:
        """Return the per-fold values of one rank-``k`` extra.

        Args:
            name: Key present in :attr:`fold_extras` (e.g. ``"rank_1_accuracy"``).

        Returns:
            1-D array of per-fold values; empty when no extras were computed.
        """
        return np.array([e[name] for e in self.fold_extras], dtype=np.float64)


def _fit_fold(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    classifier: ClassifierMixin,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]
]:
    """Fit a fresh pipeline on the train slice and predict on the test slice.

    Args:
        features: ``(B, d)`` feature matrix.
        labels: ``(B,)`` subject labels.
        train_idx: Integer indices selecting the train slice.
        test_idx: Integer indices selecting the test slice.
        classifier: Classifier prototype; cloned before fitting.

    Returns:
        Tuple ``(y_test, y_pred, y_score, classes)`` ready for scoring.
    """
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    pipe = make_pipeline(clone(classifier))
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    y_score = _score_matrix(pipe, x_test)
    classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
    return y_test, y_pred, y_score, classes


def _fit_eval_fold_full(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    classifier: ClassifierMixin,
    ranks: tuple[int, ...],
) -> FoldEvaluation:
    """Fit once and derive classification metrics and rank-``k`` accuracies.

    Args:
        features: ``(B, d)`` feature matrix.
        labels: ``(B,)`` subject labels.
        train_idx: Integer indices selecting the train slice.
        test_idx: Integer indices selecting the test slice.
        classifier: Classifier prototype; cloned before fitting.
        ranks: Ranks for the CMC accuracies.

    Returns:
        A :class:`FoldEvaluation` bundling the two per-fold outputs.
    """
    y_test, y_pred, y_score, classes = _fit_fold(
        features, labels, train_idx, test_idx, classifier
    )
    metrics = evaluate(y_test, y_pred, y_score, classes=classes)
    rank_acc = rank_k_accuracies(y_test, y_score, classes, ranks=ranks)
    return FoldEvaluation(metrics=metrics, rank_accuracies=rank_acc)


def cross_validate_classifier_multiseed(
    bundle: TemplateBundle,
    classifier_name: str,
    classifier: ClassifierMixin,
    n_folds: int = DEFAULT_N_FOLDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    n_jobs: int | None = None,
) -> CrossValidationResult:
    """Aggregate fold metrics across multiple split seeds.

    Each fold is fitted once; the classification metrics and rank-``k``
    accuracies are both derived from that single fit.

    Args:
        bundle: Templates and labels to evaluate.
        classifier_name: Label used in logs and results.
        classifier: Classifier prototype.
        n_folds: Folds per seed.
        split_seeds: Seeds driving independent CV runs.
        segments_per_block: Block size for temporal grouping.
        ranks: Ranks for the per-fold CMC accuracies stored in ``fold_extras``.
        n_jobs: Parallel workers; ``None`` reads :data:`CV_N_JOBS`.

    Returns:
        A :class:`CrossValidationResult` whose ``fold_metrics`` /
        ``fold_extras`` flatten all ``(seed, fold)`` observations.
    """
    effective_n_jobs = CV_N_JOBS if n_jobs is None else n_jobs
    groups = temporal_block_groups(bundle.labels, segments_per_block=segments_per_block)

    # Materialise every (seed, fold_idx) train/test partition deterministically
    # in seed order.
    fold_specs: list[tuple[int, int, NDArray[np.int64], NDArray[np.int64]]] = []
    for seed in split_seeds:
        splitter = stratified_group_splitter(n_splits=n_folds, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(bundle.features, bundle.labels, groups=groups)
        ):
            fold_specs.append((seed, fold_idx, np.asarray(train_idx), np.asarray(test_idx)))

    if effective_n_jobs == 1 or len(fold_specs) <= 1:
        fold_evals = [
            _fit_eval_fold_full(bundle.features, bundle.labels, tr, te, classifier, ranks)
            for _, _, tr, te in fold_specs
        ]
    else:
        # inner_max_num_threads=1 prevents CPU oversubscription (and keeps BLAS
        # reduction orders stable for reproducibility) when workers themselves
        # call into multithreaded libraries.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fold_evals = Parallel(n_jobs=effective_n_jobs)(
                delayed(_fit_eval_fold_full)(
                    bundle.features,
                    bundle.labels,
                    tr,
                    te,
                    classifier,
                    ranks,
                )
                for _, _, tr, te in fold_specs
            )

    fold_metrics = [fe.metrics for fe in fold_evals]
    fold_extras = [dict(fe.rank_accuracies) for fe in fold_evals]

    for (seed, fold_idx, _, _), m, fe in zip(fold_specs, fold_metrics, fold_extras):
        logger.debug(
            "[%s | %d feats | %s | seed %d fold %d] Acc=%.4f AUC=%.4f EER=%.4f F1=%.4f R1=%.4f",
            classifier_name,
            bundle.features.shape[1],
            bundle.key_mode.value,
            seed,
            fold_idx,
            m.accuracy,
            m.auc,
            m.eer,
            m.f1,
            fe.get("rank_1_accuracy", float("nan")),
        )

    acc_vals = np.array([m.accuracy for m in fold_metrics], dtype=np.float64)
    f1_vals = np.array([m.f1 for m in fold_metrics], dtype=np.float64)
    r1_vals = np.array([fe.get("rank_1_accuracy", float("nan")) for fe in fold_extras], dtype=np.float64)
    logger.info(
        "[%s | %d feats | %s] %d seeds × %d folds = %d obs | Acc=%.4f±%.4f F1=%.4f±%.4f R1=%.4f±%.4f",
        classifier_name,
        bundle.features.shape[1],
        bundle.key_mode.value,
        len(split_seeds),
        n_folds,
        len(fold_metrics),
        float(np.nanmean(acc_vals)),
        float(np.nanstd(acc_vals, ddof=1)),
        float(np.nanmean(f1_vals)),
        float(np.nanstd(f1_vals, ddof=1)),
        float(np.nanmean(r1_vals)),
        float(np.nanstd(r1_vals, ddof=1)),
    )
    return CrossValidationResult(
        classifier=classifier_name,
        n_features=bundle.features.shape[1],
        key_mode=bundle.key_mode,
        fold_metrics=tuple(fold_metrics),
        n_folds_per_seed=n_folds,
        fold_extras=tuple(fold_extras),
    )


__all__ = [
    "CrossValidationResult",
    "DEFAULT_SPLIT_SEEDS",
    "FoldEvaluation",
    "KeyMode",
    "SHARED_TOKEN",
    "TemplateBundle",
    "build_templates",
    "cross_validate_classifier_multiseed",
    "feature_dimension",
    "make_pipeline",
    "preprocess_signals",
]

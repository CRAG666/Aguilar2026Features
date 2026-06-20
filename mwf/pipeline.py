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
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import islice
from types import MappingProxyType
from typing import Final, NamedTuple

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import Classifier
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
from .feature_transform import FeatureScaler, transform_multimodal_batch
from .features import extract_features_batch, feature_dimension
from .metrics import ClassificationMetrics, evaluate
from .noise import add_awgn_batch
from .operating_curves import DEFAULT_RANKS, rank_k_accuracies
from .progress import track, tqdm_joblib

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
        scaler: The :class:`FeatureScaler` fitted on this cohort's features
            before the projection (``None`` for IDENTITY or when standardisation
            is disabled). Pass it to a held-out cohort's :func:`build_templates`
            to standardise the test set with *train* statistics — the
            leakage-free path used by the sealed holdout.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    key_mode: KeyMode
    scaler: FeatureScaler | None = None


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
    scaler: FeatureScaler | None = None,
    config: PipelineConfig | None = None,
) -> TemplateBundle:
    """Run AWGN → clean → multimodal feature extraction → (BioHash | identity).

    Args:
        segments: ECG/PPG cohort.
        feature_level: DWT depth for feature extraction.
        feature_wavelet: Wavelet family for feature extraction.
        projection_ratio: BioHashing template length as a fraction ``m/d``.
        binarise: Whether to sign-binarise the BioHashing projection.
        standardize: Whether to per-feature z-score before the projection (see
            :data:`DEFAULT_STANDARDIZE`). Ignored for :attr:`KeyMode.IDENTITY`
            (no projection). When ``scaler`` is ``None`` the scaler is fitted on
            *this* cohort; pass a train-fitted ``scaler`` to standardise a
            held-out cohort with train statistics (the leakage-free holdout path).
        scaler: Optional pre-fitted :class:`FeatureScaler`. When given it
            overrides ``standardize`` and is applied verbatim, so test segments
            never see their own statistics. The fitted (or supplied) scaler is
            returned on :attr:`TemplateBundle.scaler`.
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

    bundle = build_templates_from_features(
        features,
        segments.labels,
        key_mode=key_mode,
        projection_ratio=projection_ratio,
        binarise=binarise,
        standardize=standardize,
        scaler=scaler,
    )
    logger.info(
        "Built %d templates [feat=%s J=%d | regime=%s | ratio=%.2f binarise=%s std=%s] → %d dims.",
        bundle.features.shape[0],
        feature_wavelet,
        feature_level,
        key_mode.value,
        projection_ratio,
        binarise,
        standardize,
        bundle.features.shape[1],
    )
    return bundle


def build_templates_from_features(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    key_mode: KeyMode = KeyMode.PER_SUBJECT,
    projection_ratio: float = DEFAULT_PROJECTION_RATIO,
    binarise: bool = DEFAULT_BINARISE,
    standardize: bool = DEFAULT_STANDARDIZE,
    scaler: FeatureScaler | None = None,
) -> TemplateBundle:
    """Apply the cancelable transform to already-extracted features.

    The clean→extract front-end of :func:`build_templates` depends only on the
    signals and DWT level, never on the key regime, so a caller sweeping several
    regimes over one cohort can extract the features **once** and call this per
    regime — skipping the redundant NeuroKit cleaning and DWT that otherwise
    dominate a multi-regime sweep. For a given ``features`` the result is
    byte-identical to :func:`build_templates`.

    Args:
        features: ``(B, d)`` multimodal feature matrix (ECG‖PPG).
        labels: ``(B,)`` subject labels aligned with ``features``.
        key_mode: Key regime (see :class:`KeyMode`).
        projection_ratio: BioHashing template length as a fraction ``m/d``.
        binarise: Whether to sign-binarise the BioHashing projection.
        standardize: Whether to z-score before the projection when no ``scaler``
            is supplied (ignored for :attr:`KeyMode.IDENTITY`).
        scaler: Optional pre-fitted scaler; overrides ``standardize`` and is
            applied verbatim (the leakage-free held-out path).

    Returns:
        A :class:`TemplateBundle` ready for CV.
    """
    effective_scaler: FeatureScaler | None = None
    if key_mode == KeyMode.IDENTITY:
        templates = features
    else:
        tokens = _tokens_for(labels, key_mode)
        assert tokens is not None
        # Fit the standardiser on this cohort only when one is not supplied, so
        # a held-out cohort can reuse the train-fitted scaler (leakage-free).
        effective_scaler = (
            scaler if scaler is not None
            else (FeatureScaler.fit(features) if standardize else None)
        )
        templates = transform_multimodal_batch(
            features,
            tokens,
            projection_ratio=projection_ratio,
            binarise=binarise,
            standardize=False,
            scaler=effective_scaler,
        )
    return TemplateBundle(
        features=templates,
        labels=labels.copy(),
        key_mode=key_mode,
        scaler=effective_scaler,
    )


def class_score_matrix(
    estimator: BaseEstimator, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a ``(n, n_classes)`` per-class score matrix as ``float64``.

    Prefers ``predict_proba`` when the estimator exposes it; otherwise falls
    back to ``decision_function``. Every downstream identification metric
    (macro AUC, EER, AP) and the CMC rank accuracies are computed one-vs-rest
    per class, so they need only a per-class score that ranks samples within a
    column — not a calibrated, sum-to-one posterior. This lets the uncalibrated
    SVM (``probability=False``) be scored through its decision function without
    changing any reported number, while probability classifiers keep their
    posteriors. The 1-D binary ``decision_function`` is expanded to the two
    class-aligned columns ``[-score, +score]``.

    Args:
        estimator: Fitted classifier (or pipeline / search) exposing either
            ``predict_proba`` or ``decision_function``.
        x: ``(n, d)`` feature matrix.

    Returns:
        ``(n, n_classes)`` score matrix whose columns align with ``classes_``.
    """
    proba = getattr(estimator, "predict_proba", None)
    if proba is not None:
        return np.asarray(proba(x), dtype=np.float64)
    decision = estimator.decision_function
    scores = np.asarray(decision(x), dtype=np.float64)
    if scores.ndim == 1:  # binary one-vs-rest → two class-aligned columns
        scores = np.column_stack([-scores, scores])
    return scores


DEFAULT_INNER_FOLDS: Final[int] = 3
DEFAULT_TUNE_SCORING: Final[str] = "f1_macro"


def make_pipeline(classifier: Classifier) -> Pipeline:
    """Wrap ``classifier`` in a ``StandardScaler`` + classifier pipeline.

    Args:
        classifier: Classifier instance to use as the final step.

    Returns:
        A scikit-learn :class:`Pipeline`.
    """
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", classifier)])


def _build_fold_estimator(
    classifier: Classifier,
    param_grid: Mapping[str, list] | None,
    groups_train: NDArray[np.int64] | None,
    labels_train: NDArray[np.int64],
    inner_folds: int,
    scoring: str,
) -> tuple[Pipeline | GridSearchCV, bool]:
    """Build the per-fold estimator, tuning hyperparameters when a grid is given.

    Returns:
        Tuple ``(estimator, needs_groups)``. When ``param_grid`` is non-empty and
        the train slice has enough groups for a stratified group inner CV, the
        estimator is a :class:`~sklearn.model_selection.GridSearchCV` over the
        scaler+clf pipeline (``needs_groups`` is ``True``); otherwise it is the
        plain pipeline with the fixed hyperparameters (``needs_groups`` ``False``).
        Falling back keeps small/degenerate folds working instead of crashing the
        inner split.
    """
    pipe = make_pipeline(clone(classifier))
    if not param_grid or groups_train is None:
        return pipe, False
    # StratifiedGroupKFold needs every class present in at least ``inner_folds``
    # distinct groups; bail to the fixed config when the train slice is too thin.
    enough_groups_per_class = all(
        np.unique(groups_train[labels_train == cls]).size >= inner_folds
        for cls in np.unique(labels_train)
    )
    if not enough_groups_per_class or np.unique(groups_train).size < inner_folds:
        return pipe, False
    inner_cv = stratified_group_splitter(n_splits=inner_folds, random_state=DEFAULT_SEED)
    search = GridSearchCV(
        pipe,
        param_grid=dict(param_grid),
        cv=inner_cv,
        scoring=scoring,
        refit=True,
        n_jobs=1,  # outer parallelism already saturates cores; avoid oversubscription
        error_score="raise",
    )
    return search, True


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
    classifier: Classifier,
    param_grid: Mapping[str, list] | None = None,
    groups: NDArray[np.int64] | None = None,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    tune_scoring: str = DEFAULT_TUNE_SCORING,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]
]:
    """Fit a fresh (optionally tuned) pipeline on the train slice and predict.

    When ``param_grid`` is supplied, hyperparameters are selected by a group-aware
    inner cross-validation on the *train slice only* — the outer fold's test slice
    is never seen during selection, so the reported scores stay unbiased (nested
    CV). Otherwise the fixed-hyperparameter pipeline is fitted directly.

    Args:
        features: ``(B, d)`` feature matrix.
        labels: ``(B,)`` subject labels.
        train_idx: Integer indices selecting the train slice.
        test_idx: Integer indices selecting the test slice.
        classifier: Classifier prototype; cloned before fitting.
        param_grid: Optional ``clf__``-prefixed grid enabling the inner tuning CV.
        groups: ``(B,)`` group ids (temporal blocks) for the inner group split;
            required for tuning.
        inner_folds: Folds for the inner tuning CV.
        tune_scoring: Scoring metric optimised by the inner CV.

    Returns:
        Tuple ``(y_test, y_pred, y_score, classes)`` ready for scoring.
    """
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = None if groups is None else np.asarray(groups)[train_idx]
    estimator, needs_groups = _build_fold_estimator(
        classifier, param_grid, groups_train, y_train, inner_folds, tune_scoring,
    )
    if needs_groups:
        estimator.fit(x_train, y_train, groups=groups_train)
    else:
        estimator.fit(x_train, y_train)
    y_pred = np.asarray(estimator.predict(x_test), dtype=np.int64)
    y_score = class_score_matrix(estimator, x_test)
    # ``.classes_`` is exposed by both Pipeline (via the final step) and
    # GridSearchCV (via the refit best estimator), so it works for either path.
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    return y_test, y_pred, y_score, classes


def _fit_eval_fold_full(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    classifier: Classifier,
    ranks: tuple[int, ...],
    param_grid: Mapping[str, list] | None = None,
    groups: NDArray[np.int64] | None = None,
) -> FoldEvaluation:
    """Fit once and derive classification metrics and rank-``k`` accuracies.

    Args:
        features: ``(B, d)`` feature matrix.
        labels: ``(B,)`` subject labels.
        train_idx: Integer indices selecting the train slice.
        test_idx: Integer indices selecting the test slice.
        classifier: Classifier prototype; cloned before fitting.
        ranks: Ranks for the CMC accuracies.
        param_grid: Optional inner-CV tuning grid (see :func:`_fit_fold`).
        groups: Group ids for the inner group split when tuning.

    Returns:
        A :class:`FoldEvaluation` bundling the two per-fold outputs.
    """
    y_test, y_pred, y_score, classes = _fit_fold(
        features, labels, train_idx, test_idx, classifier,
        param_grid=param_grid, groups=groups,
    )
    metrics = evaluate(y_test, y_pred, y_score, classes=classes)
    rank_acc = rank_k_accuracies(y_test, y_score, classes, ranks=ranks)
    return FoldEvaluation(metrics=metrics, rank_accuracies=rank_acc)


class _FoldJob(NamedTuple):
    """One self-contained outer-fold fit/evaluate unit.

    Carries everything a worker needs so the whole CV grid can be flattened into
    a single job list and scheduled by one pool. ``features``/``labels``/``groups``
    are held as the *same* array objects across the jobs of a bundle so joblib
    memory-maps each matrix once instead of re-pickling it per fold.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    train_idx: NDArray[np.int64]
    test_idx: NDArray[np.int64]
    classifier: Classifier
    param_grid: Mapping[str, list] | None
    groups: NDArray[np.int64] | None


# Coarse relative cost of a single fold fit, keyed by classifier class name, used
# only to order the work for longest-first scheduling, never affects results.
# Neural nets and forests dominate; trees and linear models are cheap.
_CLASSIFIER_COST_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType({
    "MLPClassifier": 20.0,
    "RandomForestClassifier": 8.0,
    "SVC": 4.0,
    "DecisionTreeClassifier": 1.0,
    "LogisticRegression": 1.0,
})


def _job_cost(job: _FoldJob) -> float:
    """Rough relative cost of one fold job, for longest-processing-time scheduling.

    A scheduling hint only; it never affects results. Combines the train-slice
    size, the tuning-grid cardinality (the dominant ``--tune`` multiplier: e.g.
    MLP's 12 configs vs LR's 4) and a coarse per-classifier weight, so the heavy
    MLP/RF fits start before the cheap LR/DT ones and the short jobs backfill the
    tail, tightening the makespan of the flat pool.
    """
    grid_card = 1
    if job.param_grid:
        for values in job.param_grid.values():
            grid_card *= len(values)
    weight = _CLASSIFIER_COST_WEIGHTS.get(type(job.classifier).__name__, 1.0)
    return job.train_idx.shape[0] * grid_card * weight


def _evaluate_fold_jobs(
    jobs: Sequence[_FoldJob], *, ranks: tuple[int, ...], n_jobs: int | None,
    desc: str = "CV folds",
) -> list[FoldEvaluation]:
    """Evaluate fold jobs, sequentially or over one loky pool, preserving order.

    The single scheduling point for every CV path: callers flatten their grid
    into ``jobs`` and get results back in the same order, so sequential and
    parallel execution are byte-identical.

    In the parallel path the jobs are *dispatched* heaviest-first (see
    :func:`_job_cost`) so the long MLP/RF folds do not strand cores in the tail,
    then the results are un-permuted back to input order: the longest-first
    schedule changes only wall-clock, never the returned values or their order.

    The array fields are passed to :func:`joblib.delayed` *individually* (not as
    the whole ``_FoldJob``): joblib only memory-maps arrays that are top-level
    call arguments, so unpacking is what keeps each bundle's templates shared
    read-only across workers. Do not "simplify" this by passing ``job`` directly
    — that silently re-pickles the feature matrix on every fold.
    """
    effective_n_jobs = CV_N_JOBS if n_jobs is None else n_jobs
    if effective_n_jobs == 1 or len(jobs) <= 1:
        return [
            _fit_eval_fold_full(
                j.features, j.labels, j.train_idx, j.test_idx, j.classifier, ranks,
                param_grid=j.param_grid, groups=j.groups,
            )
            for j in track(jobs, desc=desc)
        ]
    # Longest-processing-time-first: dispatch the costliest folds before the cheap
    # ones so idle workers backfill the tail. ``sorted`` is stable, so equal-cost
    # jobs keep input order and the schedule stays deterministic. The original
    # index rides along so results can be un-permuted back to input order.
    ordered = sorted(enumerate(jobs), key=lambda t: _job_cost(t[1]), reverse=True)
    # inner_max_num_threads=1 prevents CPU oversubscription (and keeps BLAS
    # reduction orders stable for reproducibility) when workers call into
    # multithreaded libraries.
    with parallel_config(backend="loky", inner_max_num_threads=1), \
            tqdm_joblib(len(ordered), desc=desc):
        dispatched = list(
            Parallel(n_jobs=effective_n_jobs)(
                delayed(_fit_eval_fold_full)(
                    job.features, job.labels, job.train_idx, job.test_idx,
                    job.classifier, ranks, job.param_grid, job.groups,
                )
                for _idx, job in ordered
            )
        )
    # Un-permute: restore input order so callers see schedule-independent output.
    indices = (idx for idx, _job in ordered)
    return [r for _, r in sorted(zip(indices, dispatched, strict=True), key=lambda t: t[0])]


def cross_validate_classifier_multiseed(
    bundle: TemplateBundle,
    classifier_name: str,
    classifier: Classifier,
    n_folds: int = DEFAULT_N_FOLDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    n_jobs: int | None = None,
    param_grid: Mapping[str, list] | None = None,
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
        param_grid: Optional ``clf__``-prefixed hyperparameter grid. When given,
            each outer fold tunes the classifier with a group-aware inner CV on
            its train slice only (nested CV) — the unbiased way to report a tuned
            model. When ``None`` the fixed reference configuration is used.

    Returns:
        A :class:`CrossValidationResult` whose ``fold_metrics`` /
        ``fold_extras`` flatten all ``(seed, fold)`` observations.
    """
    groups, fold_specs = _make_fold_specs(
        bundle, n_folds=n_folds, split_seeds=split_seeds,
        segments_per_block=segments_per_block,
    )
    jobs = [
        _FoldJob(bundle.features, bundle.labels, tr, te, classifier, param_grid, groups)
        for _seed, _fold_idx, tr, te in fold_specs
    ]
    fold_evals = _evaluate_fold_jobs(jobs, ranks=ranks, n_jobs=n_jobs)
    return _assemble_cv_result(
        classifier_name, bundle, fold_evals,
        n_folds_per_seed=n_folds, n_split_seeds=len(split_seeds),
    )


def _make_fold_specs(
    bundle: TemplateBundle,
    *,
    n_folds: int,
    split_seeds: tuple[int, ...],
    segments_per_block: int,
) -> tuple[
    NDArray[np.int64],
    list[tuple[int, int, NDArray[np.int64], NDArray[np.int64]]],
]:
    """Materialise every ``(seed, fold_idx, train, test)`` partition for one bundle.

    Deterministic in seed order — the same partitions whether they are later
    evaluated sequentially or fanned out across workers — so results never
    depend on the execution schedule.

    Returns:
        Tuple ``(groups, fold_specs)``: the temporal-block group ids (reused for
        the inner tuning split) and the per-fold train/test index arrays.
    """
    groups = temporal_block_groups(bundle.labels, segments_per_block=segments_per_block)
    fold_specs: list[tuple[int, int, NDArray[np.int64], NDArray[np.int64]]] = []
    for seed in split_seeds:
        splitter = stratified_group_splitter(n_splits=n_folds, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(bundle.features, bundle.labels, groups=groups)
        ):
            fold_specs.append((seed, fold_idx, np.asarray(train_idx), np.asarray(test_idx)))
    return groups, fold_specs


def _assemble_cv_result(
    classifier_name: str,
    bundle: TemplateBundle,
    fold_evals: Sequence[FoldEvaluation],
    *,
    n_folds_per_seed: int,
    n_split_seeds: int,
) -> CrossValidationResult:
    """Aggregate per-fold evaluations into a :class:`CrossValidationResult`.

    Shared by :func:`cross_validate_classifier_multiseed` and
    :func:`cross_validate_tasks` so both produce byte-identical aggregates
    regardless of how the folds were scheduled.
    """
    fold_metrics = [fe.metrics for fe in fold_evals]
    fold_extras = [dict(fe.rank_accuracies) for fe in fold_evals]

    acc_vals = np.array([m.accuracy for m in fold_metrics], dtype=np.float64)
    f1_vals = np.array([m.f1 for m in fold_metrics], dtype=np.float64)
    r1_vals = np.array(
        [fe.get("rank_1_accuracy", float("nan")) for fe in fold_extras], dtype=np.float64
    )
    logger.info(
        "[%s | %d feats | %s] %d seeds × %d folds = %d obs | Acc=%.4f±%.4f F1=%.4f±%.4f R1=%.4f±%.4f",
        classifier_name,
        bundle.features.shape[1],
        bundle.key_mode.value,
        n_split_seeds,
        n_folds_per_seed,
        len(fold_metrics),
        float(np.nanmean(acc_vals)) if acc_vals.size else float("nan"),
        float(np.nanstd(acc_vals, ddof=1)) if acc_vals.size > 1 else 0.0,
        float(np.nanmean(f1_vals)) if f1_vals.size else float("nan"),
        float(np.nanstd(f1_vals, ddof=1)) if f1_vals.size > 1 else 0.0,
        float(np.nanmean(r1_vals)) if r1_vals.size else float("nan"),
        float(np.nanstd(r1_vals, ddof=1)) if r1_vals.size > 1 else 0.0,
    )
    return CrossValidationResult(
        classifier=classifier_name,
        n_features=bundle.features.shape[1],
        key_mode=bundle.key_mode,
        fold_metrics=tuple(fold_metrics),
        n_folds_per_seed=n_folds_per_seed,
        fold_extras=tuple(fold_extras),
    )


def _single_threaded(estimator: Classifier) -> Classifier:
    """Clone ``estimator`` forcing ``n_jobs=1`` when it requests multi-core work.

    When the outer fold pool already saturates every core, an estimator that
    *also* parallelises internally (e.g. ``RandomForestClassifier(n_jobs=-1)``)
    oversubscribes the CPU — dozens of threads fighting for 12 cores, which is
    slower than one thread per worker. ``n_jobs`` only governs parallelism, not
    the fitted model (the trees are fixed by ``random_state``), so forcing it to
    1 leaves every reported number identical while removing the contention.
    """
    est = clone(estimator)
    # Only constrain estimators that actually request multi-core work (e.g.
    # ``RandomForestClassifier(n_jobs=-1)``). Skip ``None``/``1``: those are
    # already single-threaded, and on estimators where ``n_jobs`` is deprecated
    # and inert (e.g. ``LogisticRegression`` since sklearn 1.8) setting it would
    # only raise a ``FutureWarning`` while changing nothing.
    if est.get_params().get("n_jobs") not in (None, 1):
        est.set_params(n_jobs=1)
    return est


@dataclass(frozen=True, slots=True)
class CVTask:
    """One independent identification CV evaluation (regime × level × classifier).

    Attributes:
        key: Opaque caller-supplied identifier used to map a result back to its
            grouping (e.g. ``(regime, level, classifier_name)``).
        bundle: Templates and labels to evaluate.
        classifier_name: Label used in logs and the result.
        classifier: Classifier prototype (cloned per fold).
        param_grid: Optional ``clf__``-prefixed grid enabling per-fold nested CV.
    """

    key: Hashable
    bundle: TemplateBundle
    classifier_name: str
    classifier: Classifier
    param_grid: Mapping[str, list] | None = None


def cross_validate_tasks(
    tasks: Sequence[CVTask],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
    segments_per_block: int = DEFAULT_SEGMENTS_PER_BLOCK,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    n_jobs: int | None = None,
    desc: str = "Identification CV",
) -> list[CrossValidationResult]:
    """Cross-validate many ``(bundle, classifier)`` tasks over **one** worker pool.

    The flat list of *every* ``(task, seed, fold)`` evaluation across all tasks
    is submitted in a single :class:`joblib.Parallel` call, so the pool stays
    full until the very end — one scheduling tail for the whole grid instead of
    one per task. This is the genuine win over calling
    :func:`cross_validate_classifier_multiseed` in a Python loop, where each call
    only parallelises its own ``n_folds × n_seeds`` folds and leaves cores idle
    on every task's tail.

    Determinism / rigour: fold partitions come from :func:`_make_fold_specs`
    (seed-ordered, schedule-independent), each task's classifier is run
    single-threaded to avoid nested-pool oversubscription (no effect on the
    fitted model), and results are regrouped in task order — so the returned
    list is element-for-element identical to evaluating each task on its own.
    joblib memory-maps each distinct feature matrix once and shares it read-only
    across the workers, so passing the same ``bundle`` to several tasks does not
    re-serialise its templates per fold.

    Args:
        tasks: Independent evaluations; the result list aligns with this order.
        n_folds: Folds per seed.
        split_seeds: Seeds driving the repeated CV.
        segments_per_block: Block size for temporal grouping.
        ranks: Ranks for the per-fold CMC accuracies.
        n_jobs: Worker count; ``None`` reads :data:`CV_N_JOBS` (``1`` runs the
            flat list sequentially).
        desc: Progress-bar label naming the current stage.

    Returns:
        One :class:`CrossValidationResult` per task, in input order.
    """
    tasks = list(tasks)

    # Flatten every (task, seed, fold) into one job list. The classifier is run
    # single-threaded so the outer pool — not a nested one — owns the cores; the
    # bundle's arrays are reused by reference so joblib memmaps each matrix once.
    jobs: list[_FoldJob] = []
    fold_counts: list[int] = []
    for task in tasks:
        groups, specs = _make_fold_specs(
            task.bundle, n_folds=n_folds, split_seeds=split_seeds,
            segments_per_block=segments_per_block,
        )
        clf = _single_threaded(task.classifier)
        fold_counts.append(len(specs))
        jobs.extend(
            _FoldJob(task.bundle.features, task.bundle.labels, tr, te, clf, task.param_grid, groups)
            for _seed, _fold_idx, tr, te in specs
        )

    evals = iter(_evaluate_fold_jobs(jobs, ranks=ranks, n_jobs=n_jobs, desc=desc))
    # Regroup the ordered evaluations back into one result per task.
    return [
        _assemble_cv_result(
            task.classifier_name, task.bundle, list(islice(evals, count)),
            n_folds_per_seed=n_folds, n_split_seeds=len(split_seeds),
        )
        for task, count in zip(tasks, fold_counts, strict=True)
    ]


__all__ = [
    "CVTask",
    "CrossValidationResult",
    "DEFAULT_SPLIT_SEEDS",
    "FoldEvaluation",
    "KeyMode",
    "SHARED_TOKEN",
    "TemplateBundle",
    "build_templates",
    "build_templates_from_features",
    "class_score_matrix",
    "cross_validate_classifier_multiseed",
    "cross_validate_tasks",
    "feature_dimension",
    "make_pipeline",
    "preprocess_signals",
]

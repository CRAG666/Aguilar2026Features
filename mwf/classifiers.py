"""Factories for the five reference classifiers (DT, LR, SVM, RF, MLP)."""

from __future__ import annotations

from types import MappingProxyType
from collections.abc import Callable, Mapping
from typing import Final

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import DEFAULT_SEED

CLASSIFIER_NAMES: Final[tuple[str, ...]] = ("MLP", "LR", "SVM", "DT", "RF")
DEFAULT_RANDOM_STATE: Final[int] = DEFAULT_SEED

# Every reference classifier is both a ``BaseEstimator`` and a ``ClassifierMixin``,
# so this alias satisfies ``sklearn.clone`` and the pipeline/scoring helpers alike.
type Classifier = (
    DecisionTreeClassifier | SVC | LogisticRegression
    | RandomForestClassifier | MLPClassifier
)


def make_decision_tree() -> DecisionTreeClassifier:
    """Build the reference decision-tree classifier.

    Returns:
        A seeded :class:`DecisionTreeClassifier` ready to fit.
    """
    return DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE)


def make_svm() -> SVC:
    """Build the reference RBF SVM classifier.

    ``probability=False``: the reported identification metrics (macro AUC, EER,
    AP) are all computed one-vs-rest **per class**, which only needs a monotonic
    per-class score, not a calibrated, sum-to-one posterior. The scoring layer
    (:func:`mwf.pipeline.class_score_matrix`) falls back to ``decision_function``
    for SVM, so Platt calibration — whose internal 5-fold CV makes a probability
    SVM roughly an order of magnitude slower to fit — is pure overhead here and
    is dropped. Rank-based metrics are invariant to the swap, so the numbers are
    unchanged in expectation.

    Returns:
        A seeded :class:`SVC` configured with ``C=275`` (no probability layer).
    """
    return SVC(
        C=275,
        gamma="scale",
        kernel="rbf",
        probability=False,
        random_state=DEFAULT_RANDOM_STATE,
    )


def make_logistic_regression() -> LogisticRegression:
    """Build the reference logistic-regression classifier.

    Returns:
        A seeded :class:`LogisticRegression` with ``max_iter=15000``.
    """
    return LogisticRegression(
        C=1.0,
        max_iter=15000,
        random_state=DEFAULT_RANDOM_STATE,
        solver="lbfgs",
    )


def make_random_forest() -> RandomForestClassifier:
    """Build the reference random-forest classifier.

    Returns:
        A seeded :class:`RandomForestClassifier` with 79 entropy trees.
    """
    return RandomForestClassifier(
        criterion="entropy",
        max_features="sqrt",
        n_estimators=79,
        n_jobs=-1,
        random_state=144,
    )


def make_mlp() -> MLPClassifier:
    """Build the reference MLP classifier (one hidden layer of 10 units).

    Returns:
        A seeded :class:`MLPClassifier` with early stopping enabled.
    """
    return MLPClassifier(
        hidden_layer_sizes=(10,),
        early_stopping=True,
        max_iter=5000,
        random_state=86,
        solver="adam",
    )


_FACTORIES: Final[Mapping[str, Callable[[], Classifier]]] = MappingProxyType(
    {
        "MLP": make_mlp,
        "LR": make_logistic_regression,
        "SVM": make_svm,
        "DT": make_decision_tree,
        "RF": make_random_forest,
    }
)


def build_classifier(name: str) -> Classifier:
    """Instantiate a fresh classifier by name.

    Args:
        name: Identifier from :data:`CLASSIFIER_NAMES`.

    Returns:
        A new, seeded classifier instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _FACTORIES[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown classifier {name!r}. Expected one of {CLASSIFIER_NAMES}."
        ) from exc


# Hyperparameter search grids for nested cross-validation. The single fixed
# configurations above are the *centre* of each grid; enabling ``--tune`` selects
# the per-fold winner with a group-aware inner CV so the reported outer-CV scores
# carry no selection-on-the-evaluation-data bias (the alternative — citing the
# fixed configs as if untuned — is only honest if they were never chosen by
# looking at this cohort's scores). Keys are prefixed ``clf__`` so the grid
# applies to the classifier step of the scaler+clf pipeline.
_PARAM_GRIDS: Final[Mapping[str, Mapping[str, tuple]]] = MappingProxyType(
    {
        "MLP": MappingProxyType({
            "clf__hidden_layer_sizes": ((10,), (25,), (50,), (50, 25)),
            "clf__alpha": (1e-4, 1e-3, 1e-2),
        }),
        "LR": MappingProxyType({
            "clf__C": (0.1, 1.0, 10.0, 100.0),
        }),
        "SVM": MappingProxyType({
            "clf__C": (1.0, 10.0, 100.0, 275.0),
            "clf__gamma": ("scale", "auto"),
        }),
        "DT": MappingProxyType({
            "clf__max_depth": (None, 10, 20, 40),
            "clf__criterion": ("gini", "entropy"),
        }),
        "RF": MappingProxyType({
            "clf__n_estimators": (79, 150, 300),
            "clf__max_features": ("sqrt", "log2"),
        }),
    }
)


def build_param_grid(name: str) -> dict[str, list]:
    """Return the ``clf__``-prefixed hyperparameter grid for one classifier.

    Args:
        name: Identifier from :data:`CLASSIFIER_NAMES`.

    Returns:
        A ``{param: [values]}`` grid suitable for
        :class:`sklearn.model_selection.GridSearchCV` over the scaler+clf
        pipeline. Empty when the classifier has no tunable grid registered.
    """
    grid = _PARAM_GRIDS.get(name, {})
    return {key: list(values) for key, values in grid.items()}


__all__ = ["CLASSIFIER_NAMES", "build_classifier", "build_param_grid"]

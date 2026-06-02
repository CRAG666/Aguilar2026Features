"""Factories for the five reference classifiers (DT, LR, SVM, RF, MLP)."""

from __future__ import annotations

from types import MappingProxyType
from collections.abc import Callable, Mapping
from typing import Final

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import DEFAULT_SEED

CLASSIFIER_NAMES: Final[tuple[str, ...]] = ("MLP", "LR", "SVM", "DT", "RF")
DEFAULT_RANDOM_STATE: Final[int] = DEFAULT_SEED


def make_decision_tree() -> DecisionTreeClassifier:
    """Build the reference decision-tree classifier.

    Returns:
        A seeded :class:`DecisionTreeClassifier` ready to fit.
    """
    return DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE)


def make_svm() -> SVC:
    """Build the reference RBF SVM classifier with probability estimates.

    Returns:
        A seeded :class:`SVC` configured with ``C=275`` and ``probability=True``.
    """
    return SVC(
        C=275,
        gamma="scale",
        kernel="rbf",
        probability=True,
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


_FACTORIES: Final[Mapping[str, Callable[[], ClassifierMixin]]] = MappingProxyType(
    {
        "MLP": make_mlp,
        "LR": make_logistic_regression,
        "SVM": make_svm,
        "DT": make_decision_tree,
        "RF": make_random_forest,
    }
)


def build_classifier(name: str) -> ClassifierMixin:
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


__all__ = ["CLASSIFIER_NAMES", "build_classifier"]

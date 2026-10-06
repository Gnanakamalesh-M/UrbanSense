"""Per-prediction feature contributions (SPEC section 27).

    Every major prediction should provide an explanation.
    Use explainability methods such as feature importance and SHAP where
    appropriate.
    Never claim that a correlated feature is necessarily causal.

**Which method, and why.** SHAP is tried first and, on this project, works:
``shap.TreeExplainer`` supports ``HistGradientBoostingRegressor`` as of shap
0.52, it handles the ``NaN`` values this pipeline deliberately preserves, and it
is *exactly additive* -- measured on the real model, the largest discrepancy
between ``base value + sum(contributions)`` and the prediction itself was
0.000000 across 200 rows. Additivity is the reason to prefer it: the
contributions genuinely decompose the prediction rather than merely ranking the
inputs.

It is nonetheless an **optional** dependency. CI runs Python 3.11 and this
machine runs 3.14; shap 0.52 ships a cp312-abi3 wheel, so the two environments
cannot be guaranteed the same shap. Rather than pin a version that breaks one of
them, the explainer is pluggable:

- ``shap`` present and able to explain the estimator -> exact SHAP values.
- otherwise -> **occlusion**: replace one feature with ``NaN``, re-predict, and
  record the change. The estimator handles ``NaN`` natively, so this is a
  genuine "without this input" counterfactual rather than an imputed stand-in.

Which one ran is recorded on every explanation and in the registry.
:attr:`Explanation.is_additive` says whether the contributions decompose the
prediction or merely rank the inputs, because presenting occlusion deltas as a
decomposition would be a quiet lie -- they do not sum to anything in particular.

**Language.** Nothing here says a feature *caused* anything. The model found an
association in historical data; that is all a gradient-boosted tree can know.
The rendering vocabulary is fixed, and a test greps it for causal verbs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from urbansense.explainability.factors import FactorGroups

#: Rows sampled for the global importance pass. Permutation importance refits
#: nothing but re-predicts once per feature per repeat, so the whole validation
#: split would be wasteful for a number that stabilizes quickly.
DEFAULT_GLOBAL_SAMPLE = 1500

#: Repeats for permutation importance.
DEFAULT_PERMUTATION_REPEATS = 5


class AttributionMethod(StrEnum):
    """How the contributions were computed."""

    #: Exact Shapley values from shap's tree explainer. Additive.
    SHAP_TREE = "shap_tree"
    #: Replace one feature with NaN and re-predict. Not additive.
    OCCLUSION = "occlusion"
    #: Shuffle one feature across rows and measure the error increase. Global.
    PERMUTATION = "permutation"


#: Maps a design matrix straight to predictions. The forecaster's own
#: ``predict`` takes a FeatureTable and guards the column order, which is right
#: for callers but wrong here: occlusion and permutation both need to feed the
#: model matrices they have deliberately altered. :meth:`FactorAttributor.
#: from_forecaster` takes the column order from the fitted model, so the
#: guarantee that guard provides is preserved at construction instead.
MatrixPredictor = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Forecaster(Protocol):
    """The slice of a fitted forecaster this module needs."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Column order the model was fitted on."""
        ...

    @property
    def estimator(self) -> object:
        """The underlying estimator."""
        ...


class TreeExplainerLike(Protocol):
    """The slice of shap's tree explainer this module uses.

    Declared structurally so shap stays an optional import: nothing here needs
    the package to be installed for the types to check.
    """

    expected_value: NDArray[np.float64] | float

    def shap_values(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed contributions, one row per sample."""
        ...


@dataclass(frozen=True)
class FactorContribution:
    """One factor's contribution to one prediction.

    Attributes:
        factor: Readable factor name.
        contribution: Signed contribution in the target's units, summed over the
            factor's features.
        share: Absolute contribution as a fraction of all absolute
            contributions. A ranking aid; it does not mean the factor supplied
            that fraction of the predicted value.
        features: Per-feature signed contributions, largest magnitude first.
    """

    factor: str
    contribution: float
    share: float
    features: tuple[tuple[str, float], ...] = ()

    @property
    def direction(self) -> str:
        """Which way this factor moved the prediction."""
        if self.contribution > 0:
            return "upward"
        return "downward" if self.contribution < 0 else "neutral"

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "factor": self.factor,
            "contribution": round(self.contribution, 3),
            "share": round(self.share, 4),
            "direction": self.direction,
            "features": [[name, round(value, 3)] for name, value in self.features],
        }

    def describe(self) -> str:
        """One line, in deliberately non-causal language."""
        return (
            f"{self.factor:<20} {self.share:>6.1%}  contributed {self.contribution:+,.0f} "
            f"{self.direction} to the prediction"
        )


@dataclass(frozen=True)
class Explanation:
    """Why one prediction came out where it did.

    Attributes:
        method: How the contributions were computed.
        prediction: The value being explained.
        baseline: The model's average output, when the method provides one.
        factors: Contributions by factor, largest absolute first.
        is_additive: Whether ``baseline + sum(contributions)`` reconstructs the
            prediction. False for occlusion, and the renderer says so.
    """

    method: AttributionMethod
    prediction: float
    baseline: float
    factors: tuple[FactorContribution, ...]
    is_additive: bool

    @property
    def top_factors(self) -> tuple[str, ...]:
        """Factor names, strongest first."""
        return tuple(item.factor for item in self.factors)

    @property
    def reconstruction_error(self) -> float:
        """How far ``baseline + sum(contributions)`` sits from the prediction."""
        total = self.baseline + sum(item.contribution for item in self.factors)
        return float(abs(total - self.prediction))

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "method": self.method.value,
            "additive": self.is_additive,
            "prediction": round(self.prediction, 3),
            "baseline": round(self.baseline, 3),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "factors": [item.as_dict() for item in self.factors],
            "interpretation_note": (
                "these are associations the model learned from historical data, "
                "not causes; a factor that contributed to a prediction did not "
                "necessarily bring the traffic about"
            ),
        }

    def describe(self, limit: int = 6) -> str:
        """Human-readable summary, in the shape SPEC section 27 shows."""
        nature = (
            "additive"
            if self.is_additive
            else "not additive -- shares rank the inputs, they do not decompose the prediction"
        )
        lines = [f"contributing factors ({self.method.value}, {nature}):"]
        lines.extend(f"  {item.describe()}" for item in self.factors[:limit])
        if self.is_additive:
            lines.append(
                f"  baseline {self.baseline:,.0f} plus the contributions above "
                f"reconstructs the predicted {self.prediction:,.0f}"
            )
        lines.append("  these are associations the model learned from past data, not causes")
        return "\n".join(lines)


def _shap_explainer(estimator: object) -> TreeExplainerLike | None:
    """Build a shap tree explainer, or ``None`` when shap cannot be used.

    Every failure mode -- not installed, installed but unable to parse this
    estimator -- lands on the same fallback, because from the caller's point of
    view they are the same situation: no SHAP values are available.
    """
    try:
        import shap
    except ImportError:
        return None
    try:
        return cast(TreeExplainerLike, shap.TreeExplainer(estimator))
    except Exception:
        # shap raises assorted exception types for unsupported models.
        return None


@dataclass
class FactorAttributor:
    """Explains predictions as grouped factor contributions.

    Attributes:
        predict_matrix: Maps a design matrix to predictions.
        groups: Factor grouping for the feature names.
        feature_names: Column order of the design matrix.
        method: Which attribution method is in use.
    """

    predict_matrix: MatrixPredictor
    groups: FactorGroups
    feature_names: tuple[str, ...]
    method: AttributionMethod
    _explainer: TreeExplainerLike | None = None
    _baseline: float = 0.0

    @classmethod
    def from_forecaster(
        cls,
        forecaster: Forecaster,
        *,
        groups: FactorGroups,
        background: NDArray[np.float64] | None = None,
    ) -> FactorAttributor:
        """Build from a fitted forecaster, taking its column order from it.

        Args:
            forecaster: The fitted model. Its ``feature_names`` become the
                column order, so the explainer cannot disagree with the model
                about which column is which.
            groups: Factor grouping. Validated against the feature names here,
                so a misconfigured grouping fails at construction rather than
                when someone asks for an explanation.
            background: Rows used to compute the occlusion baseline, when SHAP
                is unavailable.
        """
        estimator = forecaster.estimator
        predict = getattr(estimator, "predict", None)
        if not callable(predict):
            raise TypeError("the forecaster's estimator exposes no predict()")
        return cls.build(
            predict,
            groups=groups,
            feature_names=forecaster.feature_names,
            estimator=estimator,
            background=background,
        )

    @classmethod
    def build(
        cls,
        predict_matrix: MatrixPredictor,
        *,
        groups: FactorGroups,
        feature_names: tuple[str, ...],
        estimator: object | None = None,
        background: NDArray[np.float64] | None = None,
    ) -> FactorAttributor:
        """Pick the best available attribution method.

        Args:
            predict_matrix: Maps a design matrix to predictions, used for the
                occlusion fallback.
            groups: Factor grouping, validated against ``feature_names`` here.
            feature_names: Design-matrix column order.
            estimator: The underlying tree estimator, for shap.
            background: Rows used to compute the occlusion baseline.
        """
        groups.assign(feature_names)  # raises on an ungrouped or ambiguous feature

        explainer = _shap_explainer(estimator) if estimator is not None else None
        if explainer is not None:
            baseline = float(np.ravel(np.asarray(explainer.expected_value))[0])
            return cls(
                predict_matrix=predict_matrix,
                groups=groups,
                feature_names=feature_names,
                method=AttributionMethod.SHAP_TREE,
                _explainer=explainer,
                _baseline=baseline,
            )

        baseline = float(np.mean(predict_matrix(background))) if background is not None else 0.0
        return cls(
            predict_matrix=predict_matrix,
            groups=groups,
            feature_names=feature_names,
            method=AttributionMethod.OCCLUSION,
            _baseline=baseline,
        )

    @property
    def is_additive(self) -> bool:
        """Whether this method decomposes the prediction exactly."""
        return self.method is AttributionMethod.SHAP_TREE

    def _contributions(self, row: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed per-feature contributions for one row."""
        if self._explainer is not None:
            values = self._explainer.shap_values(row.reshape(1, -1))
            return np.asarray(values, dtype=np.float64).reshape(-1)

        # Occlusion: blank one feature at a time. The estimator handles NaN
        # natively, so this asks "what would the model say without this input"
        # rather than substituting a value the data never contained.
        base = float(self.predict_matrix(row.reshape(1, -1))[0])
        occluded = np.repeat(row.reshape(1, -1), len(self.feature_names), axis=0)
        for index in range(len(self.feature_names)):
            occluded[index, index] = np.nan
        without = np.asarray(self.predict_matrix(occluded), dtype=np.float64)
        return base - without

    def explain_row(self, row: NDArray[np.float64]) -> Explanation:
        """Explain one prediction as grouped factor contributions."""
        contributions = self._contributions(row)
        prediction = float(self.predict_matrix(row.reshape(1, -1))[0])

        per_factor: dict[str, float] = {}
        per_feature: dict[str, list[tuple[str, float]]] = {}
        for name, value in zip(self.feature_names, contributions, strict=True):
            factor = self.groups.factor_for(name)
            per_factor[factor] = per_factor.get(factor, 0.0) + float(value)
            per_feature.setdefault(factor, []).append((name, float(value)))

        total = sum(abs(value) for value in per_factor.values())
        factors = tuple(
            FactorContribution(
                factor=factor,
                contribution=value,
                share=(abs(value) / total if total else 0.0),
                features=tuple(sorted(per_feature[factor], key=lambda pair: -abs(pair[1]))),
            )
            for factor, value in sorted(per_factor.items(), key=lambda pair: -abs(pair[1]))
        )

        return Explanation(
            method=self.method,
            prediction=prediction,
            baseline=self._baseline,
            factors=factors,
            is_additive=self.is_additive,
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the registry."""
        return {
            "method": self.method.value,
            "additive": self.is_additive,
            "chosen_because": (
                "shap.TreeExplainer supports HistGradientBoostingRegressor and is exactly additive"
                if self.method is AttributionMethod.SHAP_TREE
                else "shap is unavailable or cannot parse this estimator, so "
                "contributions come from NaN-occlusion, which ranks inputs but "
                "does not decompose the prediction"
            ),
            "factors": list(self.groups.names),
        }


@dataclass(frozen=True)
class GlobalImportance:
    """Which factors matter across a dataset, not for one prediction.

    Attributes:
        method: How it was computed.
        dataset: Which split it was computed on.
        by_factor: Factor name to importance share.
        by_feature: Raw per-feature importances.
    """

    method: AttributionMethod
    dataset: str
    by_factor: dict[str, float]
    by_feature: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "method": self.method.value,
            "dataset": self.dataset,
            "by_factor": {k: round(v, 4) for k, v in self.by_factor.items()},
            "by_feature": {k: round(v, 4) for k, v in self.by_feature.items()},
            "note": (
                "importance is association, not causation: a feature the model "
                "leans on may be standing in for something it never observed"
            ),
        }

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [f"global factor importance ({self.method.value}, {self.dataset}):"]
        for factor, share in sorted(self.by_factor.items(), key=lambda pair: -pair[1]):
            lines.append(f"  {factor:<20} {share:>6.1%}")
        lines.append("  importance is association, not causation")
        return "\n".join(lines)


def global_importance(
    predict_matrix: MatrixPredictor,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    groups: FactorGroups,
    feature_names: tuple[str, ...],
    dataset: str = "validation",
    sample: int = DEFAULT_GLOBAL_SAMPLE,
    repeats: int = DEFAULT_PERMUTATION_REPEATS,
    seed: int = 20260101,
) -> GlobalImportance:
    """Permutation importance, grouped into factors.

    Computed on **validation**: permuting the training data would measure what
    the model memorized, and permuting test would spend the held-out split on a
    diagnostic.
    """
    from sklearn.inspection import permutation_importance

    generator = np.random.default_rng(seed)
    if x.shape[0] > sample:
        chosen = generator.choice(x.shape[0], size=sample, replace=False)
        x, y = x[chosen], y[chosen]

    class _Wrapped:
        """Adapts the forecaster to the estimator interface sklearn expects."""

        def fit(self, *_: object) -> _Wrapped:
            return self

        def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
            return predict_matrix(data)

        def score(self, data: NDArray[np.float64], target: NDArray[np.float64]) -> float:
            # Negative MAE: higher is better, so a larger drop means a more
            # important feature.
            return -float(np.mean(np.abs(target - predict_matrix(data))))

    result = permutation_importance(
        _Wrapped(), x, y, n_repeats=repeats, random_state=seed, scoring=None
    )
    raw = {
        name: max(float(value), 0.0)
        for name, value in zip(feature_names, result.importances_mean, strict=True)
    }

    per_factor: dict[str, float] = {}
    for name, value in raw.items():
        factor = groups.factor_for(name)
        per_factor[factor] = per_factor.get(factor, 0.0) + value

    total = sum(per_factor.values())
    shares = {factor: (value / total if total else 0.0) for factor, value in per_factor.items()}
    return GlobalImportance(
        method=AttributionMethod.PERMUTATION,
        dataset=dataset,
        by_factor=shares,
        by_feature=raw,
    )

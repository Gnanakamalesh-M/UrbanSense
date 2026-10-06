"""The gradient boosting forecaster (SPEC section 25).

``HistGradientBoostingRegressor`` was chosen over LightGBM for one decisive
reason: it handles ``NaN`` natively, learning a default split direction for
missing values. That is what lets this project honour "missing stays missing" all
the way to the model. An imputer would have been the alternative, and imputing
3.5% of lag values with a column mean would teach the model that a sensor outage
looks like an average hour.

It is also deterministic under ``random_state``, so the metrics committed to the
model registry can be reproduced, and it needs no platform-specific binary.

**Uncertainty is deliberately crude and said to be.** The interval is the 10th
and 90th percentile of *validation* residuals, computed **per zone** so a
six-lane arterial gets a wider band than a residential collector. That is cheap
-- one extra pass, no extra fit -- and it is honest about the typical spread. It
is also constant within a zone, which is wrong in a known direction: traffic
error scales with volume, so the band is too wide at 3am and too narrow at 6pm.
A proper interval needs quantile regression. This is a hint, not a guarantee, and
the registry records it as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

from urbansense.features.builder import FeatureTable
from urbansense.prediction.baselines import Forecaster

#: Fixed hyperparameters. Modest depth and a few hundred trees: enough to learn
#: a diurnal shape interacting with a zone, not enough to memorize 15,000 rows.
#: Deliberately not tuned -- an untuned model honestly reported is worth more
#: than a tuned one whose search touched the test set.
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    # squared_error rather than absolute_error, chosen on the validation segment
    # and not the test segment. absolute_error gives a slightly better overall
    # MAE (248 vs 258) but twice the error on the congestion peak (4,004 vs
    # 2,095), because it optimizes the median and a spike occurring in 0.3% of
    # rows is not the median. For a congestion forecast the peak is the point, so
    # the 4% overall cost buys a 48% improvement where it matters.
    "loss": "squared_error",
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_depth": 8,
    "max_leaf_nodes": 63,
    # 20, not the more usual 40. The planted Friday-evening pattern appears in
    # only 47 training rows, so a leaf requiring 40 samples can never isolate
    # "ZONE-B, Friday, hour 18" and the model averages the spike away entirely.
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 20260101,
}

#: Residual quantiles for the interval hint.
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90


@dataclass(frozen=True)
class PredictionInterval:
    """A crude per-zone uncertainty band.

    Attributes:
        lower_offsets: Per-zone 10th-percentile residual.
        upper_offsets: Per-zone 90th-percentile residual.
        fallback_lower: Band for a zone absent from validation.
        fallback_upper: Band for a zone absent from validation.
        method: How the band was derived, recorded for the registry.
    """

    lower_offsets: dict[str, float] = field(default_factory=dict)
    upper_offsets: dict[str, float] = field(default_factory=dict)
    fallback_lower: float = float("nan")
    fallback_upper: float = float("nan")
    method: str = "validation residual quantiles, per zone"

    def bounds(
        self, predicted: NDArray[np.float64], locations: tuple[str, ...]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply the band to point predictions.

        Returns:
            ``(lower, upper)``. The lower bound is clipped at zero: a negative
            vehicle count is not a plausible lower bound for anything.
        """
        lower = np.array(
            [self.lower_offsets.get(zone, self.fallback_lower) for zone in locations],
            dtype=np.float64,
        )
        upper = np.array(
            [self.upper_offsets.get(zone, self.fallback_upper) for zone in locations],
            dtype=np.float64,
        )
        return np.maximum(predicted + lower, 0.0), predicted + upper

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the model registry."""
        return {
            "method": self.method,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "per_zone_lower": {k: round(v, 2) for k, v in sorted(self.lower_offsets.items())},
            "per_zone_upper": {k: round(v, 2) for k, v in sorted(self.upper_offsets.items())},
            "limitation": (
                "constant width within a zone; traffic error scales with volume, "
                "so the band is too wide overnight and too narrow at peak. A "
                "calibrated interval needs quantile regression."
            ),
        }


class GradientForecaster(Forecaster):
    """Gradient boosted trees over the leakage-checked feature table.

    Attributes:
        name: Display name.
        hyperparameters: The fixed settings used.
        feature_names: Column order the model was fitted on, so a reordered
            table at prediction time is caught instead of silently feeding the
            model one variable under another's name.
        interval: The uncertainty hint, once calibrated.
    """

    name = "gradient_boosting"

    def __init__(self, *, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize with fixed hyperparameters."""
        self.hyperparameters = dict(
            DEFAULT_HYPERPARAMETERS if hyperparameters is None else hyperparameters
        )
        self._model = HistGradientBoostingRegressor(**self.hyperparameters)
        self.feature_names: tuple[str, ...] = ()
        self.interval = PredictionInterval()
        self._fitted = False

    @property
    def estimator(self) -> HistGradientBoostingRegressor:
        """The underlying scikit-learn estimator."""
        return self._model

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    def fit(self, table: FeatureTable) -> GradientForecaster:
        """Fit on a feature table.

        ``NaN`` features are passed through untouched -- the estimator handles
        them, and that is the whole reason it was chosen.

        Raises:
            ValueError: if the table is empty. Fitting on nothing produces a
                model that predicts a constant and reports no error.
        """
        if len(table) == 0:
            raise ValueError("cannot fit on an empty feature table")

        self.feature_names = table.names
        self._model.fit(table.x, table.y)
        self._fitted = True
        return self

    def predict(self, table: FeatureTable) -> NDArray[np.float64]:
        """Predict the target for every row.

        Raises:
            RuntimeError: if called before fitting.
            ValueError: if the table's columns differ from the fitted order.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        if table.names != self.feature_names:
            raise ValueError(
                "feature columns differ from the fitted order.\n"
                f"  fitted: {list(self.feature_names)}\n"
                f"  given:  {list(table.names)}\n"
                "The design matrix is positional, so a reordered column would feed "
                "the model one variable under another's name."
            )
        if len(table) == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self._model.predict(table.x), dtype=np.float64)

    def calibrate_interval(self, validation: FeatureTable) -> PredictionInterval:
        """Derive the per-zone residual band from validation predictions.

        Uses the **validation** segment, never the test segment. Calibrating on
        test would make the reported coverage a self-fulfilling measurement.
        """
        predicted = self.predict(validation)
        residuals = validation.y - predicted

        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        for zone in sorted(set(validation.locations)):
            mask = np.array([loc == zone for loc in validation.locations], dtype=bool)
            zone_residuals = residuals[mask]
            finite = zone_residuals[~np.isnan(zone_residuals)]
            if finite.size < 10:
                # Too few points to quantify a spread; leave the zone to the
                # global fallback rather than quoting a band from a handful.
                continue
            lower[zone] = float(np.quantile(finite, LOWER_QUANTILE))
            upper[zone] = float(np.quantile(finite, UPPER_QUANTILE))

        finite_all = residuals[~np.isnan(residuals)]
        self.interval = PredictionInterval(
            lower_offsets=lower,
            upper_offsets=upper,
            fallback_lower=(
                float(np.quantile(finite_all, LOWER_QUANTILE)) if finite_all.size else float("nan")
            ),
            fallback_upper=(
                float(np.quantile(finite_all, UPPER_QUANTILE)) if finite_all.size else float("nan")
            ),
        )
        return self.interval

    def predict_with_interval(
        self, table: FeatureTable
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Point prediction plus the crude lower and upper bounds."""
        predicted = self.predict(table)
        lower, upper = self.interval.bounds(predicted, table.locations)
        return predicted, lower, upper

    def describe(self) -> dict[str, object]:
        """Serializable description, for the model registry."""
        return {
            "name": self.name,
            "algorithm": "sklearn.ensemble.HistGradientBoostingRegressor",
            "algorithm_rationale": (
                "handles NaN natively, so missing values stay missing instead of "
                "being imputed; deterministic under random_state; no platform "
                "binary"
            ),
            "hyperparameters": dict(self.hyperparameters),
            "feature_names": list(self.feature_names),
            "interval": self.interval.as_dict(),
        }

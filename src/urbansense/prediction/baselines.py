"""Naive baselines (SPEC section 38).

Baselines come first, and they are not a formality. A gradient model that cannot
beat "the traffic next hour will be what it is now" has learned nothing worth
deploying, and without the comparison printed next to it, nobody would know.

Both baselines read a single existing feature column rather than touching the
raw series. That is deliberate: they go through exactly the same feature table,
and therefore the same leakage probe, as the real model. A baseline that reached
into the data by its own path could be accidentally leaky and would then set an
impossibly high bar that the honest model appears to fail.

Neither baseline is fitted. ``fit`` exists so they share the forecaster
interface, and it does nothing at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from urbansense.features.builder import FeatureTable


class Forecaster(ABC):
    """Something that predicts the target for a feature table.

    Attributes:
        name: Display name used in the results table and the registry.
    """

    name: str = "forecaster"

    @abstractmethod
    def fit(self, table: FeatureTable) -> Forecaster:
        """Fit on a table. Returns self, so calls can be chained."""

    @abstractmethod
    def predict(self, table: FeatureTable) -> NDArray[np.float64]:
        """Predict the target for every row.

        ``NaN`` is a legitimate output meaning "cannot answer this row" -- for a
        baseline whose source feature is missing, inventing a number would be
        imputation by another name. The metrics count an unanswered row as a
        full error rather than skipping it, so declining is never free.
        """

    def describe(self) -> dict[str, object]:
        """Serializable description, for the model registry."""
        return {"name": self.name, "algorithm": type(self).__name__, "hyperparameters": {}}


class NaiveLastValue(Forecaster):
    """Predict the most recent known value.

    For a 1h horizon this is the value at ``t`` predicting ``t + 1h``, and it is
    a genuinely strong baseline for hourly traffic -- consecutive hours are
    highly correlated. It is weakest exactly where it matters, on the morning and
    evening ramps where volume changes fastest, which is the gap a real model has
    to earn its keep in.
    """

    name = "naive_last"

    def __init__(self, *, feature: str = "lag_1h") -> None:
        """Initialize.

        Args:
            feature: Feature column holding the most recent known value.
        """
        self.feature = feature

    def fit(self, table: FeatureTable) -> NaiveLastValue:
        """Nothing to fit."""
        return self

    def predict(self, table: FeatureTable) -> NDArray[np.float64]:
        """Return the lag column as the prediction.

        Raises:
            KeyError: if the feature is not in the table. Failing loudly beats
                silently predicting zeros and reporting a terrible baseline that
                the real model then trivially beats.
        """
        if self.feature not in table.names:
            raise KeyError(
                f"{self.name} needs the {self.feature!r} feature; the table has {list(table.names)}"
            )
        return np.asarray(table.column(self.feature), dtype=np.float64).copy()

    def describe(self) -> dict[str, object]:
        """Serializable description."""
        return {
            "name": self.name,
            "algorithm": type(self).__name__,
            "hyperparameters": {"feature": self.feature},
        }


class SameHourLastWeek(Forecaster):
    """Predict the value from the same hour one week earlier.

    Captures the weekly shape -- including the planted Friday evening peak --
    without learning anything, which makes it the honest bar for "did the model
    add value beyond knowing what day it is?". It is blind to the level shift
    caused by drift, since a week-old value predates the change.

    At a 1h horizon the week-old value for the target is ``lag_168h`` taken from
    the sample time; at longer horizons the nearest available weekly lag is used
    and the offset is recorded, because pretending to an exactness the features
    do not provide would misreport what the baseline actually saw.
    """

    name = "same_hour_last_week"

    def __init__(self, *, feature: str = "lag_168h") -> None:
        """Initialize.

        Args:
            feature: Feature column holding the value one week before the sample
                time.
        """
        self.feature = feature

    def fit(self, table: FeatureTable) -> SameHourLastWeek:
        """Nothing to fit."""
        return self

    def predict(self, table: FeatureTable) -> NDArray[np.float64]:
        """Return the weekly lag column as the prediction.

        Raises:
            KeyError: if the feature is not in the table.
        """
        if self.feature not in table.names:
            raise KeyError(
                f"{self.name} needs the {self.feature!r} feature; the table has {list(table.names)}"
            )
        return np.asarray(table.column(self.feature), dtype=np.float64).copy()

    def describe(self) -> dict[str, object]:
        """Serializable description."""
        return {
            "name": self.name,
            "algorithm": type(self).__name__,
            "hyperparameters": {"feature": self.feature},
        }

"""Rolling drift monitoring over a stream in time order (SPEC section 29).

The monitor walks a feature table forward in fixed steps and asks, at each step,
whether the recent window differs from the reference. **It never looks forward.**
Each evaluation sees only rows whose sample time is at or before the instant
being evaluated, which is the same discipline the feature builder and the
leakage probe enforce one layer down.

**Gates are calibrated on the reference period, then frozen.** The monitor runs
in two passes over the reference stretch: the first collects each statistic's
own week-to-week values, the second turns them into gates. Nothing from the
monitored period feeds back into a gate, so a detection can never be the result
of the gate moving.

**The reference must not overlap the training period**, and ``training_end``
turns that from advice into a guard. An overlapping reference is partly
in-sample, which understates the reference error and makes every later window
look drifted. This is not hypothetical: a first version of this phase used a
reference straddling the training boundary, and a drift-free dataset produced 17
false alarms -- reference MAE 140 against an honest 180. Measured with a
strictly out-of-sample reference, the gate tightens from 1.94 to 1.15 and the
false alarms go to zero, so the honest reference is also the more sensitive one.

**Which features are watched was settled by measurement, after a first guess
got it wrong.** Pre-drift maximum PSI against post-drift values, weekly windows,
2.5-month reference:

===================  ==============  ===============  ========
feature              pre-drift max   post-drift       verdict
===================  ==============  ===============  ========
roll_mean_24h        0.125           0.375 - 1.26     usable
roll_mean_168h       4.995           3.65 - 8.98      unusable
lag_1h / lag_24h     0.013           0.018 - 0.034    too weak
roll_mean_3h         0.013           0.026 - 0.044    too weak
rainfall_0h          1.326           0.003            unusable
===================  ==============  ===============  ========

Three findings, each of which changed the default:

- **A feature averaged over at least the monitoring window cannot be monitored
  with it.** ``roll_mean_168h`` is a one-week mean compared across one-week
  windows, so within any window it is nearly constant -- roughly one independent
  value per zone. The reference quantile edges are then packed together, mass
  moves wholesale between bins, and PSI reads 5 or 6 on a perfectly stable
  stretch. This is not a tuning problem, so it is not fixed by tuning:
  :meth:`MonitorConfig.unmonitorable` rejects any feature whose reach is at
  least the window, and a future config change cannot reintroduce it.
- **Zero-inflated series are unusable here.** Rainfall is mostly zeros with wet
  spells, so its weekly PSI swings to 1.3 with nothing wrong, while the planted
  shift moves it not at all (0.003). It is excluded for that reason, not
  because it misbehaved once.
- **Point lags are real but too small for the published gate.** ``lag_24h``
  separates 0.013 from 0.018: a true signal, and far below 0.25. Watching them
  would add noise without ever firing.

So ``roll_mean_24h`` is the default watched set, alone.

**Which family should be trusted to act was also settled by measurement, and
the answer is not the obvious one.** Running the same monitor against a
drift-free copy of the dataset -- same seed, the level shift simply switched off
-- gives:

====================  ===============  ==================
family                drifted stream   drift-free stream
====================  ===============  ==================
feature drift         16 firings       15 firings
performance drift     16 firings       10 firings
**prediction drift**  15 firings       **0 firings**
====================  ===============  ==================

Feature and performance drift fire on *both* streams, because this generator has
annual seasonality: a champion trained on autumn sees genuinely different inputs
and is genuinely worse by spring with nothing having changed in the city. Only
prediction drift separates a real shift from seasonal degradation.

Performance drift is still the fastest -- four days against prediction drift's
eleven -- so it earns its place as a leading indicator. But acting on it alone
would retrain the model every spring, so
:attr:`EpisodeTracker.required_kinds` gates candidate creation on prediction
drift and the rest are reported as corroboration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from urbansense.drift.detectors import (
    DEFAULT_MIN_REFERENCE_WINDOWS,
    DEFAULT_PSI_BINS,
    DEFAULT_SIGMA_MULTIPLIER,
    PSI_MAJOR,
    DriftKind,
    DriftSignal,
    DriftStatus,
    feature_drift,
    performance_drift,
    prediction_drift,
    psi,
    sigma_gate,
)
from urbansense.drift.events import DriftEvent, EpisodeTracker
from urbansense.features.builder import FeatureTable

#: Replay step. Weekly keeps each window around 650 rows on this dataset, which
#: is enough for a stable PSI while still noticing a shift within days.
DEFAULT_STEP = timedelta(days=7)

#: How far back a monitored window reaches. Equal to the step, so windows tile
#: the stream without overlapping -- overlapping windows would correlate
#: consecutive evaluations and make an episode look longer than it is.
DEFAULT_WINDOW = timedelta(days=7)

#: Features watched by default: the 24-hour rolling mean, alone. Chosen by
#: measurement rather than by reasoning -- see the table in the module
#: docstring for what each alternative scored and why it was rejected.
DEFAULT_WATCHED_FEATURES: tuple[str, ...] = ("roll_mean_24h",)

#: Metrics excluded from distributional monitoring regardless of configuration.
#: Rainfall is zero-inflated -- mostly dry hours with wet spells -- so its
#: weekly PSI swings by more than a planted level shift does. A detector that
#: fires on the weather is worse than no detector.
UNMONITORABLE_METRICS: frozenset[str] = frozenset({"rainfall"})

#: Minimum rows in a window before it is judged at all.
MIN_WINDOW_ROWS = 50


@dataclass(frozen=True)
class MonitorConfig:
    """How the monitor runs.

    Attributes:
        step: How far the replay advances each evaluation.
        window: How far back each evaluation looks.
        watched_features: Feature columns compared distributionally.
        psi_gate: Feature-drift gate. The published 0.25 convention.
        sigma_multiplier: Standard deviations above the reference mean for the
            prediction and performance gates.
        min_reference_windows: Reference windows needed before a sigma gate is
            considered established.
        bins: PSI bins.
    """

    step: timedelta = DEFAULT_STEP
    window: timedelta = DEFAULT_WINDOW
    watched_features: tuple[str, ...] = DEFAULT_WATCHED_FEATURES
    psi_gate: float = PSI_MAJOR
    sigma_multiplier: float = DEFAULT_SIGMA_MULTIPLIER
    min_reference_windows: int = DEFAULT_MIN_REFERENCE_WINDOWS
    bins: int = DEFAULT_PSI_BINS

    def unmonitorable(self, table: FeatureTable) -> tuple[tuple[str, str], ...]:
        """Watched features that cannot honestly be compared, with the reason.

        Two exclusions, both structural rather than empirical:

        - A feature whose backward reach is at least the monitoring window is
          near-constant within that window, so PSI measures the packing of the
          reference quantiles rather than any change in the city.
        - A zero-inflated covariate swings more between quiet windows than a
          real shift moves it.

        Returning the reasons rather than silently filtering means the run
        header can say what it declined to watch.
        """
        window_hours = self.window.total_seconds() / 3600.0
        reach = {spec.name: spec.reach_hours for spec in table.feature_set.specs}
        metric = {spec.name: spec.metric for spec in table.feature_set.specs}

        rejected: list[tuple[str, str]] = []
        for name in self.watched_features:
            if name not in table.names:
                rejected.append((name, "not a column of this feature table"))
                continue
            if metric.get(name) in UNMONITORABLE_METRICS:
                rejected.append(
                    (
                        name,
                        f"{metric[name]} is zero-inflated; its weekly PSI moves more "
                        "than a real shift does",
                    )
                )
                continue
            if reach.get(name, 0) >= window_hours:
                rejected.append(
                    (
                        name,
                        f"reach {reach[name]}h is at least the {window_hours:.0f}h "
                        "window, so the feature is near-constant within it and PSI "
                        "is not interpretable",
                    )
                )
        return tuple(rejected)

    def monitorable(self, table: FeatureTable) -> tuple[str, ...]:
        """Watched features that survive :meth:`unmonitorable`."""
        excluded = {name for name, _ in self.unmonitorable(table)}
        return tuple(name for name in self.watched_features if name not in excluded)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "step_days": self.step.total_seconds() / 86400.0,
            "window_days": self.window.total_seconds() / 86400.0,
            "watched_features": list(self.watched_features),
            "psi_gate": self.psi_gate,
            "sigma_multiplier": self.sigma_multiplier,
            "min_reference_windows": self.min_reference_windows,
            "bins": self.bins,
        }


@dataclass(frozen=True)
class DriftGates:
    """The gates in force, and where each came from.

    Attributes:
        feature_psi: Feature-drift gate, from the published convention.
        prediction_psi: Prediction-drift gate, from reference variability.
        performance_ratio: Performance-drift gate, from reference variability.
        reference_error: The out-of-sample reference error.
        reference_windows: Reference windows the sigma gates were estimated on.
        established: Whether there were enough reference windows to estimate.
    """

    feature_psi: float
    prediction_psi: float
    performance_ratio: float
    reference_error: float
    reference_windows: int
    established: bool

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "feature_psi": self.feature_psi,
            "feature_psi_source": "published PSI convention (0.25), not fitted",
            "prediction_psi": (
                None if np.isnan(self.prediction_psi) else round(self.prediction_psi, 6)
            ),
            "performance_ratio": (
                None if np.isnan(self.performance_ratio) else round(self.performance_ratio, 4)
            ),
            "sigma_gate_source": "reference period only; never the monitored period",
            "reference_error": round(self.reference_error, 4),
            "reference_error_basis": "out-of-sample",
            "reference_windows": self.reference_windows,
            "established": self.established,
        }

    def describe(self) -> str:
        """Human-readable summary."""
        if not self.established:
            return (
                f"gates NOT established: only {self.reference_windows} reference "
                "window(s); sigma-based detectors will report insufficient_reference"
            )
        return (
            f"gates from {self.reference_windows} reference window(s): "
            f"feature PSI {self.feature_psi:.3f} (convention), "
            f"prediction PSI {self.prediction_psi:.4f}, "
            f"performance ratio {self.performance_ratio:.3f} "
            f"(reference MAE {self.reference_error:,.1f}, out-of-sample)"
        )


#: Maps a feature table to one prediction per row.
PredictFn = Callable[[FeatureTable], NDArray[np.float64]]


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    """Rows whose sample time falls in ``[start, end)``."""
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)


def _mae(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Mean absolute error, or ``nan`` on an empty window."""
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


@dataclass
class DriftMonitor:
    """Watches a stream for the three kinds of drift.

    Attributes:
        reference: The reference window -- what "normal" means. Must be out of
            sample for the champion, or the reference error is understated.
        predict: The champion's prediction function.
        config: How the monitor runs.
        training_end: Exclusive upper bound of the champion's training period,
            so the last training sample sits strictly before it. When given,
            the reference is checked against it at construction. A reference
            beginning exactly at this instant is correct -- that row was not
            trained on.
        gates: The gates in force, once calibrated.
        tracker: Episode and cooldown bookkeeping.

    Raises:
        ValueError: if ``training_end`` is given and the reference starts
            strictly before it. An overlapping reference deflates the reference
            error and turns a stable stream into a stream of false alarms.
    """

    reference: FeatureTable
    predict: PredictFn
    config: MonitorConfig = field(default_factory=MonitorConfig)
    training_end: datetime | None = None
    gates: DriftGates | None = None
    tracker: EpisodeTracker = field(default_factory=EpisodeTracker)
    _reference_predictions: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Refuse a reference that the champion has already seen."""
        if self.training_end is None or len(self.reference) == 0:
            return
        earliest = min(self.reference.sample_times)
        if earliest < self.training_end:
            raise ValueError(
                f"the reference window starts {earliest.isoformat()}, before the "
                f"training end {self.training_end.isoformat()}. An in-sample reference "
                "understates the reference error -- on this dataset that turned a "
                "drift-free stream into 17 false alarms. Start the reference after "
                "the training period."
            )

    def calibrate(self) -> DriftGates:
        """Establish gates from the reference period's own variability.

        The reference is sliced into windows of the configured size and each
        statistic is computed window-against-whole-reference. The spread of
        those values is what a gate is built from, so a gate reflects how much
        this stream moves when nothing is wrong.

        Returns:
            The gates. When the reference is too short to slice into
            ``min_reference_windows`` windows, the sigma gates come back as
            ``nan`` and :attr:`DriftGates.established` is false -- the monitor
            then reports ``INSUFFICIENT_REFERENCE`` rather than inventing a
            number.
        """
        reference_predictions = np.asarray(self.predict(self.reference), dtype=np.float64)
        self._reference_predictions = reference_predictions
        reference_error = _mae(self.reference.y, reference_predictions)

        prediction_values: list[float] = []
        ratio_values: list[float] = []

        start = min(self.reference.sample_times)
        last = max(self.reference.sample_times)
        while start <= last:
            end = start + self.config.window
            slice_ = _window(self.reference, start, end)
            start = end
            if len(slice_) < MIN_WINDOW_ROWS:
                continue
            predicted = np.asarray(self.predict(slice_), dtype=np.float64)
            index = psi(reference_predictions, predicted, bins=self.config.bins)
            if not np.isnan(index):
                prediction_values.append(index)
            window_error = _mae(slice_.y, predicted)
            if reference_error > 0 and not np.isnan(window_error):
                ratio_values.append(window_error / reference_error)

        windows = min(len(prediction_values), len(ratio_values))
        established = windows >= self.config.min_reference_windows
        self.gates = DriftGates(
            feature_psi=self.config.psi_gate,
            prediction_psi=sigma_gate(
                prediction_values,
                multiplier=self.config.sigma_multiplier,
                minimum_windows=self.config.min_reference_windows,
            ),
            performance_ratio=sigma_gate(
                ratio_values,
                multiplier=self.config.sigma_multiplier,
                minimum_windows=self.config.min_reference_windows,
                # A gate below 1.0 would fire on a window that is *better* than
                # the reference, which is not drift worth retraining for.
                floor=1.0,
            ),
            reference_error=reference_error,
            reference_windows=windows,
            established=established,
        )
        return self.gates

    def evaluate(self, table: FeatureTable, *, at: datetime) -> DriftEvent:
        """Judge the window ending at ``at``.

        Only rows strictly before ``at`` are read, so an evaluation can never
        see the instant it is deciding about.
        """
        gates = self.gates if self.gates is not None else self.calibrate()
        recent = _window(table, at - self.config.window, at)

        if len(recent) < MIN_WINDOW_ROWS:
            return DriftEvent(
                detected_at=at,
                window_start=at - self.config.window,
                window_end=at,
                signals=(
                    DriftSignal(
                        kind=DriftKind.PERFORMANCE,
                        subject="window",
                        status=DriftStatus.INSUFFICIENT_DATA,
                        statistic=float("nan"),
                        threshold=float("nan"),
                        reference_rows=len(self.reference),
                        recent_rows=len(recent),
                        detail={"reason": f"only {len(recent)} rows in the window"},
                    ),
                ),
            )

        signals: list[DriftSignal] = []

        # --- feature drift ------------------------------------------------
        # The guard runs here rather than at construction so a table whose
        # columns differ from the configured set is reported, not crashed on.
        for name in self.config.monitorable(self.reference):
            signals.append(
                feature_drift(
                    name,
                    self.reference.column(name),
                    recent.column(name),
                    psi_gate=gates.feature_psi,
                    bins=self.config.bins,
                )
            )

        predicted = np.asarray(self.predict(recent), dtype=np.float64)
        reference_predictions = (
            self._reference_predictions
            if self._reference_predictions is not None
            else np.asarray(self.predict(self.reference), dtype=np.float64)
        )

        # --- prediction drift ---------------------------------------------
        if np.isnan(gates.prediction_psi):
            signals.append(
                DriftSignal(
                    kind=DriftKind.PREDICTION,
                    subject="prediction",
                    status=DriftStatus.INSUFFICIENT_REFERENCE,
                    statistic=float("nan"),
                    threshold=float("nan"),
                    reference_rows=len(self.reference),
                    recent_rows=len(recent),
                    detail={
                        "reason": (
                            f"{gates.reference_windows} reference window(s); "
                            f"{self.config.min_reference_windows} needed to estimate a gate"
                        )
                    },
                )
            )
        else:
            signals.append(
                prediction_drift(
                    reference_predictions,
                    predicted,
                    gate=gates.prediction_psi,
                    bins=self.config.bins,
                )
            )

        # --- performance drift --------------------------------------------
        if np.isnan(gates.performance_ratio):
            signals.append(
                DriftSignal(
                    kind=DriftKind.PERFORMANCE,
                    subject="mae",
                    status=DriftStatus.INSUFFICIENT_REFERENCE,
                    statistic=float("nan"),
                    threshold=float("nan"),
                    reference_rows=len(self.reference),
                    recent_rows=len(recent),
                    detail={"reason": "no sigma gate could be estimated"},
                )
            )
        else:
            signals.append(
                performance_drift(
                    _mae(recent.y, predicted),
                    reference_error=gates.reference_error,
                    gate_ratio=gates.performance_ratio,
                    recent_rows=len(recent),
                    reference_rows=len(self.reference),
                )
            )

        return DriftEvent(
            detected_at=at,
            window_start=min(recent.sample_times),
            window_end=max(recent.sample_times),
            signals=tuple(signals),
        )

    def replay(
        self,
        table: FeatureTable,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[tuple[DriftEvent, ...], tuple[datetime, ...]]:
        """Walk the stream forward, evaluating every step.

        Args:
            table: The stream to monitor, in no particular row order -- windows
                are taken by timestamp, not by position.
            start: First evaluation instant. Defaults to one window after the
                earliest sample, so the first window is full.
            end: Last evaluation instant. Defaults to the latest sample.

        Returns:
            Every event in time order, and the instants at which a candidate
            update should be raised. The second tuple is shorter than the
            first: a continuing episode inside its cooldown produces an event
            but not a candidate.
        """
        if self.gates is None:
            self.calibrate()

        earliest = min(table.sample_times)
        latest = max(table.sample_times)
        moment = start if start is not None else earliest + self.config.window
        final = end if end is not None else latest

        events: list[DriftEvent] = []
        candidates: list[datetime] = []
        while moment <= final:
            event = self.evaluate(table, at=moment)
            stamped, raise_candidate = self.tracker.observe(event)
            events.append(stamped)
            if raise_candidate:
                candidates.append(moment)
            moment = moment + self.config.step

        self.tracker.close(final)
        return tuple(events), tuple(candidates)


def summarize(events: Sequence[DriftEvent]) -> str:
    """A compact table of a replay, one line per evaluation."""
    lines = [f"{len(events)} evaluation(s), {sum(1 for e in events if e.is_drift)} fired"]
    lines.extend(f"  {event.describe()}" for event in events)
    return "\n".join(lines)

"""Per-arm metrics for the three-arm comparison (SPEC section 38).

    Measure: MAE, RMSE, Precision, Recall, F1, Calibration, Prediction error.

Nothing here computes a new definition of anything. MAE and RMSE come from
Phase 4's :func:`~urbansense.evaluation.metrics.evaluate_slice`, the congestion
threshold from Phase 6's :mod:`~urbansense.prediction.congestion`, and the Brier
score from Phase 6's :mod:`~urbansense.prediction.calibration`. An experiment
that re-implemented its own MAE could report an improvement that existed only in
the new implementation.

Three choices shape what the numbers mean, and all three are deliberate:

**The congestion threshold is shared across arms.** It is fitted once on the
training segment and is a *definition* of congestion, not part of any model.
Letting each arm fit its own would let an arm look accurate by moving the
goalposts -- predict high, raise the threshold, and precision improves without
any prediction getting better.

**The probability calibrator is also shared, and fixed.** All three arms start
from the same frozen champion, so one calibrator fitted on the reference window
is applied to every arm's predictions throughout. Recalibrating after each
retrain would be more realistic and would also confound the result: a Brier
improvement could then come from the calibrator rather than the regressor, and
the experiment is about the regressor. This is recorded as a limitation rather
than presented as a neutral choice.

**Cost sits beside accuracy, not in a footnote.** An arm that retrains twice as
often will usually look better; whether that is worth it is the actual question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from urbansense.adaptation.replay import ArmResult, RetrainCost
from urbansense.evaluation.metrics import SliceMetrics
from urbansense.features.builder import FeatureTable
from urbansense.prediction.calibration import ProbabilityCalibrator, brier_score
from urbansense.prediction.congestion import CongestionThresholds


@dataclass(frozen=True)
class ClassificationScore:
    """Congestion detection quality at a fixed threshold.

    Attributes:
        true_positive: Congested hours correctly called congested.
        false_positive: Quiet hours called congested.
        false_negative: Congested hours missed.
        true_negative: Quiet hours correctly left alone.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        """Share of congestion calls that were right.

        ``nan`` when the arm never called congestion -- which is not precision
        of 1.0, and reporting it as such would flatter a model that simply
        never raises an alarm.
        """
        called = self.true_positive + self.false_positive
        return self.true_positive / called if called else float("nan")

    @property
    def recall(self) -> float:
        """Share of real congestion that was caught."""
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else float("nan")

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        precision, recall = self.precision, self.recall
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            return float("nan")
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": None if np.isnan(self.precision) else round(self.precision, 4),
            "recall": None if np.isnan(self.recall) else round(self.recall, 4),
            "f1": None if np.isnan(self.f1) else round(self.f1, 4),
        }


def congestion_score(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    locations: tuple[str, ...],
    thresholds: CongestionThresholds,
) -> ClassificationScore:
    """Score congestion detection against the shared train-only threshold."""
    actual_flags = np.array(
        [thresholds.is_congested(locations[i], float(actual[i])) for i in range(actual.size)],
        dtype=bool,
    )
    predicted_flags = np.array(
        [thresholds.is_congested(locations[i], float(predicted[i])) for i in range(predicted.size)],
        dtype=bool,
    )
    return ClassificationScore(
        true_positive=int(np.sum(actual_flags & predicted_flags)),
        false_positive=int(np.sum(~actual_flags & predicted_flags)),
        false_negative=int(np.sum(actual_flags & ~predicted_flags)),
        true_negative=int(np.sum(~actual_flags & ~predicted_flags)),
    )


@dataclass(frozen=True)
class ArmMetrics:
    """Everything measured for one arm on one seed.

    Attributes:
        arm: Which arm.
        seed: Which generated dataset.
        rows: Rows scored. Identical across arms by construction.
        mae: Mean absolute error.
        rmse: Root mean squared error.
        slices: Per-slice errors.
        congestion: Detection quality at the shared threshold.
        brier: Brier score of the calibrated congestion probability.
        reference_brier: Brier score of always predicting the base rate, so the
            probability figure can be read against something.
        cost: Retrains, fits and training rows.
        models_used: Models that served during the replay.
    """

    arm: str
    seed: int
    rows: int
    mae: float
    rmse: float
    slices: tuple[SliceMetrics, ...]
    congestion: ClassificationScore
    brier: float
    reference_brier: float
    cost: RetrainCost
    models_used: tuple[str, ...]

    def slice_named(self, name: str) -> SliceMetrics | None:
        """One slice by name."""
        return next((item for item in self.slices if item.name == name), None)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "arm": self.arm,
            "seed": self.seed,
            "rows": self.rows,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "slices": [item.as_dict() for item in self.slices],
            "congestion": self.congestion.as_dict(),
            "brier": round(self.brier, 6),
            "reference_brier_base_rate_only": round(self.reference_brier, 6),
            "cost": self.cost.as_dict(),
            "models_used": list(self.models_used),
        }


def score_arm(
    arm: ArmResult,
    *,
    seed: int,
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    locations: tuple[str, ...],
    thresholds: CongestionThresholds,
    calibrator: ProbabilityCalibrator,
) -> ArmMetrics:
    """Compute every metric for one arm.

    Args:
        arm: The arm's replay result, for its slices and cost.
        seed: The generated dataset this is for.
        actual: Observed targets.
        predicted: That arm's predictions for the same rows.
        locations: Zone per row.
        thresholds: The shared, train-only congestion definition.
        calibrator: The shared, fixed probability calibrator.
    """
    zone_thresholds = np.array(
        [thresholds.by_zone.get(zone, float("nan")) for zone in locations], dtype=np.float64
    )
    probabilities = calibrator.calibrate(calibrator.raw(predicted, zone_thresholds))
    outcomes = np.array(
        [thresholds.is_congested(locations[i], float(actual[i])) for i in range(actual.size)],
        dtype=bool,
    )
    base_rate = float(outcomes.mean()) if outcomes.size else float("nan")

    return ArmMetrics(
        arm=arm.name,
        seed=seed,
        rows=arm.rows,
        mae=arm.overall.mae,
        rmse=arm.overall.rmse,
        slices=arm.slices,
        congestion=congestion_score(actual, predicted, locations, thresholds),
        brier=brier_score(probabilities, outcomes),
        reference_brier=brier_score(np.full_like(probabilities, base_rate), outcomes),
        cost=arm.cost,
        models_used=arm.models_used,
    )


@dataclass(frozen=True)
class Aggregate:
    """One metric's mean and spread across seeds.

    Attributes:
        mean: Mean across seeds.
        sd: Sample standard deviation. ``nan`` for a single seed, which is
            honest: one dataset has no spread, and printing 0.0 would imply
            a precision the run does not have.
        seeds: How many seeds contributed.
        values: The per-seed values, kept so a reader can see the spread rather
            than only its summary.
    """

    mean: float
    sd: float
    seeds: int
    values: tuple[float, ...] = ()

    @classmethod
    def of(cls, values: list[float]) -> Aggregate:
        """Summarize a metric across seeds, ignoring unknowns."""
        clean = [value for value in values if not np.isnan(value)]
        if not clean:
            return cls(mean=float("nan"), sd=float("nan"), seeds=0)
        array = np.asarray(clean, dtype=np.float64)
        return cls(
            mean=float(array.mean()),
            sd=float(array.std(ddof=1)) if array.size > 1 else float("nan"),
            seeds=int(array.size),
            values=tuple(round(float(value), 4) for value in array),
        )

    def describe(self, *, places: int = 1) -> str:
        """``235.9 +/- 4.2`` style, or just the mean when there is one seed."""
        if np.isnan(self.mean):
            return "n/a"
        if np.isnan(self.sd):
            return f"{self.mean:,.{places}f}"
        return f"{self.mean:,.{places}f} +/- {self.sd:,.{places}f}"

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "mean": None if np.isnan(self.mean) else round(self.mean, 6),
            "sd": None if np.isnan(self.sd) else round(self.sd, 6),
            "seeds": self.seeds,
            "values": list(self.values),
        }


def within_one_sd(left: Aggregate, right: Aggregate) -> bool:
    """Whether two aggregates are too close to call.

    The gap between the means is compared against the larger of the two spreads.
    A difference smaller than the run-to-run variation is not a finding, and
    calling it one is the most common way a comparison like this misleads.
    """
    if np.isnan(left.mean) or np.isnan(right.mean):
        return False
    spreads = [value for value in (left.sd, right.sd) if not np.isnan(value)]
    if not spreads:
        # One seed each: no spread was measured, so no difference can be called
        # significant. Saying "too close to call" is the honest default.
        return True
    return abs(left.mean - right.mean) <= max(spreads)


def evaluation_locations(tables: list[FeatureTable]) -> tuple[str, ...]:
    """Zone per scored row, concatenated in the order the arms were scored."""
    return tuple(zone for table in tables for zone in table.locations)


def evaluation_record_ids(tables: list[FeatureTable]) -> tuple[str, ...]:
    """Record id per scored row.

    Used by the test that every arm saw identical rows: comparing row *counts*
    would pass even if two arms had been scored on different rows that happened
    to be equally numerous.
    """
    return tuple(record for table in tables for record in table.record_ids)

"""Congestion probability, and whether it can be believed (SPEC section 25).

A regressor predicts a volume; the question a reader actually asks is "how
likely is this to be congested". Turning one into the other needs a model of how
wrong the regressor usually is, which is what the training residuals are:

    P(y > threshold)  =  sf((threshold - prediction) / sigma)

with ``sigma`` the spread of the **training** residuals. The alternative, a
separate classifier, was measured and rejected: calibrated Brier 0.0564 on
validation but **0.0920 on test**, against 0.0745 for this approach, because the
classifier learned the training period's 10% congestion rate and the real rate
is 28% by the test period. It would also have let the probability and the
predicted volume disagree with each other, which this cannot.

**A raw probability is not a calibrated one.** "70%" should mean it happens
about 70% of the time, and nothing about the formula above guarantees that. So
the raw score is passed through an isotonic regression fitted on the
**validation** split -- :meth:`ProbabilityCalibrator.fit` takes one table and
has no parameter for test data -- and the result is reported with a Brier score
and a reliability table beside it. A calibration claim without the table is
decoration, so :meth:`CalibrationReport.describe` always prints both.

**The honest caveat, measured on this data.** The calibrator is fitted where
congestion occurs 23.4% of the time and applied where it occurs 28.1% of the
time, because the city is drifting. It will therefore be systematically
*under*-confident on test. That is visible in the reliability table, and
:attr:`CalibrationReport.base_rate_shift` states it numerically rather than
leaving a reader to infer it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.isotonic import IsotonicRegression

#: Bins in the reliability table. Ten is enough to see a systematic bias on a
#: few thousand rows without splitting them so finely that each bin is noise.
DEFAULT_RELIABILITY_BINS = 10

#: Floor on the residual spread, so a degenerate fit cannot produce a
#: step-function probability that is 0 or 1 everywhere.
MIN_SIGMA = 1e-6

#: Numerator of the "rule of three" bound used to keep calibrated probabilities
#: away from exactly 0 and 1. Isotonic regression maps a bin where nothing
#: happened to precisely zero, which reads as *impossible* -- but observing no
#: events in n trials only bounds the rate at roughly 3/n at 95% confidence, and
#: on this data those hours do congest 2.4% of the time in the test period. The
#: clamp turns an unsupportable "never" into a small number, which is what the
#: evidence actually supports.
RULE_OF_THREE = 3.0


def brier_score(probabilities: NDArray[np.float64], outcomes: NDArray[np.bool_]) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better.

    The reference point that makes it meaningful is the base rate: always
    predicting the observed frequency scores ``p(1-p)``, so a Brier score is
    only good news relative to that, never on its own.
    """
    if probabilities.size == 0:
        return float("nan")
    return float(np.mean((probabilities - outcomes.astype(np.float64)) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    """One row of a reliability table.

    Attributes:
        lower: Bin's lower probability edge.
        upper: Bin's upper probability edge.
        count: Predictions in the bin.
        mean_predicted: Mean probability said.
        observed_frequency: Fraction that actually happened.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float

    @property
    def gap(self) -> float:
        """Observed minus predicted. Positive means under-confident."""
        return self.observed_frequency - self.mean_predicted

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "range": f"{self.lower:.1f}-{self.upper:.1f}",
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "observed_frequency": round(self.observed_frequency, 4),
            "gap": round(self.gap, 4),
        }


def reliability_table(
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.bool_],
    *,
    bins: int = DEFAULT_RELIABILITY_BINS,
) -> tuple[ReliabilityBin, ...]:
    """Group predictions by stated probability and count what happened.

    Empty bins are omitted rather than reported as zero: "we never said 90% and
    it never happened" and "we said 90% and it never happened" are opposite
    findings, and a zero row would read as the second.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[ReliabilityBin] = []
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        # The last bin owns its upper edge, so a probability of exactly 1.0
        # is counted rather than dropped.
        chosen = (
            (probabilities >= low) & (probabilities <= high)
            if index == bins - 1
            else (probabilities >= low) & (probabilities < high)
        )
        count = int(chosen.sum())
        if count == 0:
            continue
        rows.append(
            ReliabilityBin(
                lower=low,
                upper=high,
                count=count,
                mean_predicted=float(probabilities[chosen].mean()),
                observed_frequency=float(outcomes[chosen].mean()),
            )
        )
    return tuple(rows)


def expected_calibration_error(bins: tuple[ReliabilityBin, ...]) -> float:
    """Count-weighted mean absolute gap across the reliability bins."""
    total = sum(item.count for item in bins)
    if total == 0:
        return float("nan")
    return float(sum(item.count * abs(item.gap) for item in bins) / total)


@dataclass(frozen=True)
class CalibrationReport:
    """How well the probabilities held up on one dataset.

    Attributes:
        dataset: Which split this describes.
        rows: Predictions scored.
        base_rate: Observed frequency of congestion in this dataset.
        reference_brier: Brier score of always predicting the fitting base
            rate. The number the model has to beat to have said anything.
        raw_brier: Brier score before calibration.
        calibrated_brier: Brier score after calibration.
        bins: The reliability table.
        calibration_error: Count-weighted mean absolute gap.
        base_rate_shift: Fitting base rate minus this dataset's, when they
            differ. The reason a calibrator can be systematically off.
        in_sample: Whether this is the split the isotonic map was fitted on. If
            it is, a flat reliability table is arithmetic rather than evidence,
            and saying so is the difference between reporting calibration and
            performing it.
    """

    dataset: str
    rows: int
    base_rate: float
    reference_brier: float
    raw_brier: float
    calibrated_brier: float
    bins: tuple[ReliabilityBin, ...] = ()
    calibration_error: float = float("nan")
    base_rate_shift: float = 0.0
    in_sample: bool = False

    @property
    def improved(self) -> bool:
        """Whether calibration helped rather than hurt."""
        return self.calibrated_brier <= self.raw_brier

    @property
    def beats_base_rate(self) -> bool:
        """Whether the probabilities say more than the base rate alone."""
        return self.calibrated_brier < self.reference_brier

    @property
    def direction(self) -> str:
        """Which way the probabilities are biased, in plain words."""
        error = expected_calibration_error(self.bins)
        if np.isnan(error) or error < 0.02:
            return "well matched"
        weighted = sum(item.count * item.gap for item in self.bins)
        return "under-confident" if weighted > 0 else "over-confident"

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the registry."""
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "base_rate": round(self.base_rate, 4),
            "reference_brier_base_rate_only": round(self.reference_brier, 4),
            "raw_brier": round(self.raw_brier, 4),
            "calibrated_brier": round(self.calibrated_brier, 4),
            "calibration_improved": self.improved,
            "beats_base_rate": self.beats_base_rate,
            "expected_calibration_error": round(expected_calibration_error(self.bins), 4),
            "direction": self.direction,
            "base_rate_shift_from_fitting_split": round(self.base_rate_shift, 4),
            "in_sample_for_the_calibrator": self.in_sample,
            "reliability": [item.as_dict() for item in self.bins],
        }

    def describe(self) -> str:
        """The scores and the table together.

        Never one without the other: a Brier score alone says nothing about
        whether a stated 70% means 70%.
        """
        lines = [
            f"calibration on {self.dataset} ({self.rows:,} rows, base rate {self.base_rate:.3f})",
            f"  brier  raw {self.raw_brier:.4f}  calibrated {self.calibrated_brier:.4f}"
            f"  base-rate-only {self.reference_brier:.4f}",
            f"  expected calibration error {expected_calibration_error(self.bins):.4f}"
            f"  ({self.direction})",
        ]
        if self.in_sample:
            lines.append(
                "  NOTE: this is the split the isotonic map was fitted on, so a flat "
                "table here is arithmetic, not evidence. Judge the calibration on a "
                "split the map never saw."
            )
        if abs(self.base_rate_shift) >= 0.01:
            lines.append(
                f"  NOTE: the calibrator was fitted where congestion occurs "
                f"{self.base_rate + self.base_rate_shift:.3f} of the time and applied "
                f"where it occurs {self.base_rate:.3f} of the time, so it is "
                f"{self.direction} here by construction"
            )
        if not self.beats_base_rate:
            lines.append(
                "  WARNING: these probabilities do not beat predicting the base "
                "rate every time; they should not be presented as informative"
            )
        lines.append("  bin        n   predicted   observed       gap")
        for item in self.bins:
            lines.append(
                f"  {item.lower:.1f}-{item.upper:.1f}  {item.count:>5}      "
                f"{item.mean_predicted:.3f}      {item.observed_frequency:.3f}"
                f"    {item.gap:+.3f}"
            )
        return "\n".join(lines)


@dataclass
class ProbabilityCalibrator:
    """Turns predicted volumes into calibrated congestion probabilities.

    Attributes:
        sigma: Spread of the training residuals, the regressor's own estimate of
            how wrong it usually is.
        isotonic: The fitted monotone map, or ``None`` before fitting.
        fitted_on: Which split the isotonic map was fitted on.
        fitting_base_rate: Congestion rate where it was fitted, kept so a later
            report can state the shift.
    """

    sigma: float
    isotonic: IsotonicRegression | None = None
    fitted_on: str = ""
    fitting_base_rate: float = float("nan")
    _residual_rows: int = field(default=0, repr=False)
    _fitting_rows: int = field(default=0, repr=False)

    @property
    def is_fitted(self) -> bool:
        """Whether the isotonic map is available."""
        return self.isotonic is not None

    @classmethod
    def from_residuals(
        cls, actual: NDArray[np.float64], predicted: NDArray[np.float64]
    ) -> ProbabilityCalibrator:
        """Build from **training** residuals.

        Args:
            actual: Observed targets from the training segment.
            predicted: In-sample predictions for those rows.
        """
        residuals = actual - predicted
        sigma = float(np.std(residuals)) if residuals.size else float("nan")
        return cls(sigma=max(sigma, MIN_SIGMA), _residual_rows=int(residuals.size))

    def raw(
        self, predicted: NDArray[np.float64], thresholds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Uncalibrated P(actual > threshold) from the residual spread.

        A normal approximation to the residual distribution. It is an
        approximation -- traffic residuals are not exactly normal -- which is
        precisely why the isotonic step exists and why the reliability table is
        reported.
        """
        standardized = (thresholds - predicted) / self.sigma
        result: NDArray[np.float64] = np.asarray(stats.norm.sf(standardized), dtype=np.float64)
        return np.clip(result, 0.0, 1.0)

    def fit(
        self,
        raw_probabilities: NDArray[np.float64],
        outcomes: NDArray[np.bool_],
        *,
        dataset: str = "validation",
    ) -> ProbabilityCalibrator:
        """Fit the isotonic map on the **validation** split.

        Args:
            raw_probabilities: Uncalibrated scores for the validation rows.
            outcomes: Whether each of those rows was actually congested.
            dataset: Name recorded as the fitting split.

        There is no test data in this signature, and that is the point: the test
        split is scored by :meth:`report` and never fitted on.

        Raises:
            ValueError: if the outcomes are all one class. An isotonic map from
                a single class is a constant, and a constant probability
                presented as calibrated would be actively misleading.
        """
        if raw_probabilities.size == 0:
            raise ValueError("cannot calibrate on an empty validation set")
        if len(set(outcomes.tolist())) < 2:
            raise ValueError(
                "the calibration set contains only one outcome class; an isotonic "
                "map fitted on it would be a constant, which cannot be honestly "
                "described as calibrated"
            )

        self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            raw_probabilities, outcomes.astype(np.float64)
        )
        self.fitted_on = dataset
        self.fitting_base_rate = float(outcomes.mean())
        self._fitting_rows = int(outcomes.size)
        return self

    @property
    def floor(self) -> float:
        """Smallest probability this calibrator will state.

        Isotonic regression maps a bin where nothing happened to exactly zero,
        and zero means impossible. Observing no events in ``n`` rows only bounds
        the rate at about ``3/n``, so that bound is the floor -- and ``1 -``
        it the ceiling. Without this the output would claim certainty in both
        directions from the absence of evidence.
        """
        if self._fitting_rows <= 0:
            return 0.0
        return float(min(RULE_OF_THREE / self._fitting_rows, 0.5))

    def calibrate(self, raw_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw scores through the fitted isotonic regression.

        Returns the raw scores unchanged when unfitted, so a caller can still
        get a probability -- but :attr:`is_fitted` is false and every report
        says so rather than implying a calibration that never happened.
        """
        if self.isotonic is None:
            return np.clip(raw_probabilities, 0.0, 1.0)
        mapped: NDArray[np.float64] = np.asarray(
            self.isotonic.predict(raw_probabilities), dtype=np.float64
        )
        return np.clip(mapped, self.floor, 1.0 - self.floor)

    def report(
        self,
        raw_probabilities: NDArray[np.float64],
        outcomes: NDArray[np.bool_],
        *,
        dataset: str,
        bins: int = DEFAULT_RELIABILITY_BINS,
    ) -> CalibrationReport:
        """Score the probabilities on a dataset and build the reliability table."""
        calibrated = self.calibrate(raw_probabilities)
        base_rate = float(outcomes.mean()) if outcomes.size else float("nan")
        reference = self.fitting_base_rate if not np.isnan(self.fitting_base_rate) else base_rate
        reference_probabilities = np.full_like(calibrated, reference)
        table = reliability_table(calibrated, outcomes, bins=bins)
        return CalibrationReport(
            dataset=dataset,
            rows=int(outcomes.size),
            base_rate=base_rate,
            reference_brier=brier_score(reference_probabilities, outcomes),
            raw_brier=brier_score(raw_probabilities, outcomes),
            calibrated_brier=brier_score(calibrated, outcomes),
            bins=table,
            calibration_error=expected_calibration_error(table),
            base_rate_shift=(
                float(self.fitting_base_rate - base_rate)
                if not np.isnan(self.fitting_base_rate)
                else 0.0
            ),
            in_sample=bool(self.fitted_on and dataset == self.fitted_on),
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the registry."""
        return {
            "method": "normal residual tail, isotonic-recalibrated",
            "residual_sigma": round(self.sigma, 4),
            "residual_rows": self._residual_rows,
            "residuals_from": "train",
            "isotonic_fitted_on": self.fitted_on or "not fitted",
            "isotonic_fitting_base_rate": (
                round(self.fitting_base_rate, 4) if not np.isnan(self.fitting_base_rate) else None
            ),
            "note": ("the isotonic map is fitted on validation only; test is scored, never fitted"),
        }

"""Training a challenger without letting it see the decision horizon.

    A challenger never trains on data after the time it is being evaluated for.

That rule is enforced in three independent ways here, because it is the one
mistake that would make every number in this phase meaningless and would do so
*flatteringly* -- a leaky challenger wins its comparison and gets promoted.

1. **The filter.** :func:`available_at` keeps only rows whose sample time *and*
   target time are strictly before the decision time. Filtering on the sample
   time alone would be wrong: a sample at ``t`` predicts ``t + horizon``, so a
   row sampled an hour before the decision carries a target from after it.
2. **The gap.** The evaluation window starts a configured gap after the
   decision time, so the furthest-reaching feature of an evaluation row still
   cannot touch a training row.
3. **The probe.** :func:`train_challenger` runs the Phase 4 leakage probe on
   the challenger's own training data and returns the report. A challenger that
   fails it is rejected by the promotion rules with that as the reason -- never
   promoted, never quietly dropped.

:attr:`ChallengerFit.training_end` is returned so the caller, the decision
record and the tests can all assert the same thing independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from urbansense.adaptation.config import TrainingWindow
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.features.builder import FeatureTable
from urbansense.features.leakage import LeakageReport, detect_leakage
from urbansense.prediction.gradient import GradientForecaster

#: Smallest training set a challenger is allowed. Below this the fit is noise,
#: and a model trained on it would lose its comparison for the wrong reason.
MIN_TRAINING_ROWS = 500


def available_at(table: FeatureTable, decision_time: datetime) -> FeatureTable:
    """Rows that genuinely existed before ``decision_time``.

    Both the sample time and the target time must be strictly earlier. The
    target matters because a row sampled at ``t`` carries the observed value at
    ``t + horizon``: filtering on the sample time alone would hand a challenger
    trained "up to Thursday" a Friday measurement, which is precisely the
    leakage SPEC section 33 forbids.
    """
    mask = np.array(
        [
            sample < decision_time and target < decision_time
            for sample, target in zip(table.sample_times, table.target_times, strict=True)
        ],
        dtype=bool,
    )
    return table.select(mask)


def window_rows(
    table: FeatureTable, *, decision_time: datetime, window: TrainingWindow
) -> FeatureTable:
    """The training rows for one window, as of the decision time."""
    available = available_at(table, decision_time)
    span = window.span
    if span is None or len(available) == 0:
        return available
    earliest = decision_time - span
    mask = np.array([moment >= earliest for moment in available.sample_times], dtype=bool)
    return available.select(mask)


@dataclass(frozen=True)
class ChallengerFit:
    """A trained challenger and the evidence about what it was allowed to see.

    Attributes:
        model: The fitted forecaster.
        window: The window it was trained on.
        training_rows: Rows it trained on.
        training_start: Earliest sample time in its training data.
        training_end: Latest sample time in its training data. Strictly before
            the decision time, and the tests assert exactly that.
        target_end: Latest *target* time in its training data. The figure that
            actually matters for leakage, and the one a sample-time-only filter
            would get wrong.
        decision_time: The horizon it was trained up to.
        leakage: The probe's report on this challenger's own data.
    """

    model: GradientForecaster
    window: TrainingWindow
    training_rows: int
    training_start: datetime
    training_end: datetime
    target_end: datetime
    decision_time: datetime
    leakage: LeakageReport | None = None

    @property
    def leakage_clean(self) -> bool:
        """Whether the probe passed. ``True`` when the probe was not run."""
        return True if self.leakage is None else self.leakage.is_clean

    @property
    def respects_horizon(self) -> bool:
        """Whether nothing it saw reaches the decision time.

        Checks the target end, not just the sample end: the target is the later
        of the two and is the one a naive filter leaks.
        """
        return self.target_end < self.decision_time

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "window": self.window.name,
            "window_days": self.window.days,
            "training_rows": self.training_rows,
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "target_end": self.target_end.isoformat(),
            "decision_time": self.decision_time.isoformat(),
            "respects_horizon": self.respects_horizon,
            "leakage_clean": self.leakage_clean,
            "leakage": self.leakage.as_dict() if self.leakage is not None else None,
        }

    def describe(self) -> str:
        """Human-readable summary."""
        return (
            f"challenger[{self.window.name}]: {self.training_rows:,} rows "
            f"{self.training_start:%Y-%m-%d}..{self.training_end:%Y-%m-%d} "
            f"(targets to {self.target_end:%Y-%m-%d}, decision "
            f"{self.decision_time:%Y-%m-%d}), leakage "
            f"{'clean' if self.leakage_clean else 'FAILED'}"
        )


class ChallengerError(RuntimeError):
    """Raised when a challenger cannot honestly be trained."""


def train_challenger(
    table: FeatureTable,
    *,
    decision_time: datetime,
    window: TrainingWindow,
    reconciled: ReconciledResult | None = None,
    timezone_offset_hours: float = 0.0,
    probe_leakage: bool = True,
    min_rows: int = MIN_TRAINING_ROWS,
) -> ChallengerFit:
    """Train one challenger on data available before ``decision_time``.

    Args:
        table: The full feature table. Filtered here rather than by the caller,
            so there is one place the horizon is enforced.
        decision_time: Nothing at or after this instant may be used.
        window: Which training window to apply.
        reconciled: The unified layer, for the leakage probe. The probe needs
            observations rather than a design matrix.
        timezone_offset_hours: Local offset, passed to the probe.
        probe_leakage: Whether to run the probe. Off only where the caller has
            already run it for an identical feature set, since the probe is the
            slowest step in a replay.
        min_rows: Minimum training rows.

    Returns:
        A :class:`ChallengerFit`.

    Raises:
        ChallengerError: if too few rows are available before the decision
            time. Returning a model fitted on 40 rows would lose its comparison
            for a reason that has nothing to do with drift.
    """
    rows = window_rows(table, decision_time=decision_time, window=window)
    if len(rows) < min_rows:
        raise ChallengerError(
            f"only {len(rows)} rows available before {decision_time.isoformat()} for "
            f"window {window.name!r}; {min_rows} needed to train a challenger worth "
            "comparing"
        )

    model = GradientForecaster().fit(rows)
    report: LeakageReport | None = None
    if probe_leakage and reconciled is not None:
        # Probed against the challenger's own feature set, so a challenger
        # cannot inherit a clean bill of health from the champion's check.
        report = detect_leakage(
            reconciled, rows.feature_set, timezone_offset_hours=timezone_offset_hours
        )

    return ChallengerFit(
        model=model,
        window=window,
        training_rows=len(rows),
        training_start=min(rows.sample_times),
        training_end=max(rows.sample_times),
        target_end=max(rows.target_times),
        decision_time=decision_time,
        leakage=report,
    )


def evaluation_window(
    table: FeatureTable,
    *,
    decision_time: datetime,
    gap: timedelta,
    span: timedelta,
) -> FeatureTable:
    """The window a candidate's challengers are judged on.

    Starts ``gap`` after the decision time so the furthest-reaching feature of
    an evaluation row cannot touch a training row, and runs for ``span``.
    Everything in it is strictly future relative to every challenger.
    """
    start = decision_time + gap
    end = start + span
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)

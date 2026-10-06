"""Congestion thresholds and severity (SPEC section 25).

A regression gives a vehicle count. "Congested" is a judgement laid over it, and
this module holds that judgement in one place so nothing downstream invents its
own version.

**The threshold is fitted on training data only.** That rule is enforced
structurally rather than by discipline: :func:`fit_thresholds` accepts a single
:class:`~urbansense.features.builder.FeatureTable` and has no parameter through
which a validation or test table could arrive. A caller who wanted to leak would
have to concatenate the splits themselves, which is visible at the call site.

Two consequences of that rule are worth stating plainly, because they look like
bugs and are not:

- The threshold **ages**. On the demo data the congested rate is 0.100 over the
  training period, 0.234 over validation and 0.281 over test. The city got
  busier -- that is the planted drift plus seasonality -- and a definition
  anchored to the past therefore marks more of the present as congested. The
  alternative, re-fitting per period, would let the test split set its own pass
  mark.
- A zone with no training rows gets **no threshold**, not a borrowed one.
  Predicting congestion for a zone we have never seen is not something this
  module will fake.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from urbansense.config.models import CongestionConfig
from urbansense.features.builder import FeatureTable
from urbansense.schemas.enums import Severity


@dataclass(frozen=True)
class CongestionThresholds:
    """Per-zone congestion thresholds, with the provenance of the fit.

    Attributes:
        by_zone: Threshold per zone, in the metric's canonical unit.
        config: The configuration the fit followed.
        fitted_on: Name of the segment the fit used. Recorded so a report can
            state it rather than a reader having to trust it.
        rows: Rows the fit saw.
        first_sample: Earliest sample time in the fitting data.
        last_sample: Latest sample time in the fitting data.
    """

    by_zone: dict[str, float]
    config: CongestionConfig
    fitted_on: str
    rows: int
    first_sample: str
    last_sample: str

    def threshold_for(self, location_id: str) -> float | None:
        """Threshold for one zone, or ``None`` when it was never fitted."""
        return self.by_zone.get(location_id)

    def ratio(self, location_id: str, predicted: float) -> float:
        """Predicted volume as a multiple of the zone's threshold.

        Returns:
            ``nan`` when the zone has no threshold or the prediction is unknown.
            Not zero -- an unknown ratio is not a quiet "not congested".
        """
        threshold = self.by_zone.get(location_id)
        if threshold is None or threshold <= 0.0 or np.isnan(predicted):
            return float("nan")
        return float(predicted) / threshold

    def is_congested(self, location_id: str, value: float) -> bool:
        """Whether a volume exceeds its zone's threshold."""
        threshold = self.by_zone.get(location_id)
        if threshold is None or np.isnan(value):
            return False
        return bool(value > threshold)

    def severity(self, location_id: str, predicted: float) -> Severity:
        """Severity band for a predicted volume.

        Monotonic in ``predicted`` by construction: the ratio rises with the
        prediction and the bands ascend, so a larger prediction can never land
        in a lower band.

        An unknown ratio -- no threshold for the zone, or no prediction --
        returns ``LOW``, which is the only option the configured band set
        offers. Callers that need to tell "quiet" from "unknown" apart must
        check :meth:`ratio` for ``nan``; the prediction record does, and reports
        the zone as unscored rather than calm.
        """
        ratio = self.ratio(location_id, predicted)
        if np.isnan(ratio):
            return Severity.LOW
        return self.config.severity_for(ratio)

    def congested_mask(self, table: FeatureTable) -> NDArray[np.bool_]:
        """Which of a table's *observed* targets exceed their zone threshold.

        The labels an evaluation scores probabilities against.
        """
        return np.array(
            [self.is_congested(table.locations[i], float(table.y[i])) for i in range(len(table))],
            dtype=bool,
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the registry."""
        return {
            "method": self.config.threshold.method,
            "percentile": self.config.threshold.percentile,
            "metric": self.config.metric,
            "fitted_on": self.fitted_on,
            "rows": self.rows,
            "period": {"first_sample": self.first_sample, "last_sample": self.last_sample},
            "by_zone": {zone: round(value, 3) for zone, value in sorted(self.by_zone.items())},
            "severity_bands": [
                {"name": band.name.value, "min_ratio": band.min_ratio}
                for band in self.config.severity_bands
            ],
            "note": (
                "fitted on the training segment only; the congested rate rises in "
                "later periods because the definition is anchored to the past"
            ),
        }

    def describe(self) -> str:
        """Human-readable summary."""
        zones = ", ".join(f"{zone} {value:,.0f}" for zone, value in sorted(self.by_zone.items()))
        return (
            f"congestion = {self.config.metric} above the zone's "
            f"{self.config.threshold.percentile:g}th percentile, fitted on "
            f"{self.fitted_on} ({self.rows:,} rows, {self.first_sample} to "
            f"{self.last_sample}): {zones}"
        )


def fit_thresholds(
    train: FeatureTable,
    config: CongestionConfig,
    *,
    segment_name: str = "train",
) -> CongestionThresholds:
    """Fit per-zone congestion thresholds on the training segment.

    Args:
        train: The **training** feature table, and nothing else. There is no
            parameter here for validation or test data; that absence is the
            leakage control.
        config: The congestion definition to follow.
        segment_name: What to record as the fitting segment, so a report states
            which data set the threshold rather than implying it.

    Returns:
        A :class:`CongestionThresholds`.

    Raises:
        ValueError: if the table is empty. A threshold from no data would be a
            number with no meaning, and every downstream probability would
            inherit it.
    """
    if len(train) == 0:
        raise ValueError("cannot fit congestion thresholds on an empty table")

    percentile = config.threshold.percentile
    by_zone: dict[str, float] = {}
    locations = np.asarray(train.locations)
    for zone in sorted(set(train.locations)):
        values = train.y[locations == zone]
        # Targets are never NaN by construction, but a zone present only in
        # dropped rows would arrive empty; it gets no threshold rather than one
        # borrowed from a neighbouring zone.
        if values.size == 0:
            continue
        by_zone[zone] = float(np.percentile(values, percentile))

    return CongestionThresholds(
        by_zone=by_zone,
        config=config,
        fitted_on=segment_name,
        rows=len(train),
        first_sample=min(train.sample_times).isoformat(),
        last_sample=max(train.sample_times).isoformat(),
    )


def congested_rate(thresholds: CongestionThresholds, table: FeatureTable) -> float:
    """Share of a table's observed targets that count as congested."""
    if len(table) == 0:
        return float("nan")
    return float(thresholds.congested_mask(table).mean())

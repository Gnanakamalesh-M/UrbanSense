"""Building the feature table over the unified layer (SPEC sections 15, 19, 33).

The input is :attr:`ReconciledResult.unified` **and nothing else**. The
duplicate-linked and conflicted buckets are not parameters of
:func:`build_feature_table`, so a quarantined, duplicated or disputed record
cannot reach training even by mistake -- the signature enforces the rule rather
than a comment asking people to remember it.

Two rules shape the output:

**Missing stays missing.** An unknown feature is ``NaN``, never zero and never
filled. ``HistGradientBoostingRegressor`` learns a split direction for ``NaN``,
so the absence is information the model can use rather than a gap to patch
(SPEC section 15).

**A row needs a target.** Rows whose target value is unknown are *excluded*, not
imputed: you cannot learn from an absent label. That is the one exclusion this
module performs, and it is counted and reported rather than done quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from urbansense.data_integrity.unified import ReconciledResult
from urbansense.features.calendar import compute_calendar, to_local
from urbansense.features.series import SeriesIndex
from urbansense.features.specs import FeatureKind, FeatureSet, FeatureSpec

#: Minimum known hours before a rolling window reports a mean. Below this the
#: window is too sparse to summarize and the feature is NaN instead.
DEFAULT_MIN_WINDOW_OBSERVATIONS = 2


@dataclass(frozen=True)
class FeatureTable:
    """A design matrix with the metadata needed to audit it.

    Attributes:
        feature_set: The features, in column order.
        x: Design matrix, ``(n_samples, n_features)``, ``NaN`` where unknown.
        y: Targets, ``(n_samples,)``. Never ``NaN`` -- rows without a target
            are excluded during construction.
        sample_times: The prediction time ``t`` of each row. The target sits at
            ``t + horizon``.
        target_times: The instant each target was observed.
        locations: Zone of each row.
        record_ids: Source record id behind each target, so a row can be traced
            back to the feed line it came from.
        rows_considered: Candidate rows before exclusions.
        rows_without_target: Rows dropped for an absent target.
    """

    feature_set: FeatureSet
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sample_times: tuple[datetime, ...]
    target_times: tuple[datetime, ...]
    locations: tuple[str, ...]
    record_ids: tuple[str, ...]
    rows_considered: int = 0
    rows_without_target: int = 0

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.x.shape[0])

    @property
    def names(self) -> tuple[str, ...]:
        """Feature column names, in order."""
        return self.feature_set.names

    def column(self, name: str) -> NDArray[np.float64]:
        """One feature column by name."""
        return self.x[:, self.names.index(name)]

    @property
    def missing_counts(self) -> dict[str, int]:
        """Unknown values per feature, so sparsity is visible not hidden."""
        return {
            name: int(np.isnan(self.x[:, index]).sum()) for index, name in enumerate(self.names)
        }

    def select(self, mask: NDArray[np.bool_]) -> FeatureTable:
        """Return the subset of rows where ``mask`` is true.

        Used by the temporal splitter. Metadata is sliced alongside the matrix so
        a split can never lose track of which row is which.
        """
        chosen = np.flatnonzero(mask)
        return FeatureTable(
            feature_set=self.feature_set,
            x=self.x[chosen],
            y=self.y[chosen],
            sample_times=tuple(self.sample_times[i] for i in chosen),
            target_times=tuple(self.target_times[i] for i in chosen),
            locations=tuple(self.locations[i] for i in chosen),
            record_ids=tuple(self.record_ids[i] for i in chosen),
            rows_considered=len(chosen),
            rows_without_target=0,
        )


def _zone_codes(locations: tuple[str, ...]) -> dict[str, float]:
    """Map zone ids to stable numeric codes.

    Sorted so the code for a zone does not depend on the order rows happened to
    arrive in. A code that shifted between training and prediction would feed
    the model one zone's history under another zone's label.
    """
    return {name: float(index) for index, name in enumerate(sorted(locations))}


def compute_features(
    spec: FeatureSpec,
    *,
    index: SeriesIndex,
    location_id: str,
    sample_time: datetime,
    target_metric: str,
    timezone_offset_hours: float,
    zone_codes: dict[str, float],
    min_window_observations: int = DEFAULT_MIN_WINDOW_OBSERVATIONS,
) -> float:
    """Compute one feature for one sample.

    Every branch reads at or before ``sample_time``. The offsets come from the
    spec, which has already rejected negative values, and the leakage probe
    verifies the result behaviourally.

    Returns:
        The feature value, or ``float("nan")`` when unknown.
    """
    if spec.kind is FeatureKind.CALENDAR:
        return compute_calendar(spec.name, to_local(sample_time, timezone_offset_hours))

    if spec.kind is FeatureKind.SPATIAL:
        return zone_codes.get(location_id, float("nan"))

    metric = spec.metric if spec.metric is not None else target_metric
    where = spec.location if spec.location is not None else location_id
    end = sample_time - timedelta(hours=spec.offset_hours)

    if spec.kind in (FeatureKind.LAG, FeatureKind.COVARIATE) and spec.window_hours == 0:
        value = index.at(where, metric, end)
        return float("nan") if value is None else value

    mean = index.window_mean(
        where,
        metric,
        end=end,
        hours=max(spec.window_hours, 1),
        min_observations=min_window_observations,
    )
    return float("nan") if mean is None else mean


def build_feature_table(
    result: ReconciledResult,
    feature_set: FeatureSet,
    *,
    timezone_offset_hours: float = 0.0,
    min_window_observations: int = DEFAULT_MIN_WINDOW_OBSERVATIONS,
    index: SeriesIndex | None = None,
) -> FeatureTable:
    """Build the design matrix from the unified layer.

    Args:
        result: Reconciliation output. Only ``result.unified`` is read -- the
            duplicate-linked and conflicted buckets are never touched, which is
            how the "only unified rows" rule is enforced structurally.
        feature_set: Features to compute and the target horizon.
        timezone_offset_hours: Source's local offset, for calendar features.
            "Friday evening" is a local-clock fact.
        min_window_observations: Minimum known hours before a rolling mean is
            reported rather than left unknown.
        index: Pre-built index, used by the leakage probe to substitute a
            perturbed copy. Built from ``result.unified`` when omitted.

    Returns:
        A :class:`FeatureTable` whose rows each have a known target.
    """
    source = index if index is not None else SeriesIndex.from_observations(result.unified)
    target_metric = feature_set.target_metric
    horizon = timedelta(hours=feature_set.horizon_hours)

    target_rows = [
        observation for observation in result.unified if observation.metric == target_metric
    ]
    zone_codes = _zone_codes(tuple({row.location_id for row in target_rows}))

    # Candidate sample times are the instants where a *target* exists. Iterating
    # targets rather than all hours means a row is only built when there is
    # something to learn, and the horizon arithmetic runs in one direction only.
    candidates = sorted(
        {(row.location_id, row.event_time) for row in target_rows},
        key=lambda pair: (pair[1], pair[0]),
    )

    rows: list[list[float]] = []
    targets: list[float] = []
    sample_times: list[datetime] = []
    target_times: list[datetime] = []
    locations: list[str] = []
    record_ids: list[str] = []
    without_target = 0

    record_id_by_key = {
        (row.location_id, row.event_time): str(row.attributes.get("record_id", row.observation_id))
        for row in target_rows
    }

    for location_id, target_time in candidates:
        sample_time = target_time - horizon

        target = source.at(location_id, target_metric, target_time)
        if target is None:
            # No label: excluded, not imputed. Counted so the drop is visible.
            without_target += 1
            continue

        rows.append(
            [
                compute_features(
                    spec,
                    index=source,
                    location_id=location_id,
                    sample_time=sample_time,
                    target_metric=target_metric,
                    timezone_offset_hours=timezone_offset_hours,
                    zone_codes=zone_codes,
                    min_window_observations=min_window_observations,
                )
                for spec in feature_set.specs
            ]
        )
        targets.append(target)
        sample_times.append(sample_time)
        target_times.append(target_time)
        locations.append(location_id)
        record_ids.append(record_id_by_key[(location_id, target_time)])

    matrix = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, len(feature_set.specs)), dtype=np.float64)
    )

    return FeatureTable(
        feature_set=feature_set,
        x=matrix,
        y=np.asarray(targets, dtype=np.float64),
        sample_times=tuple(sample_times),
        target_times=tuple(target_times),
        locations=tuple(locations),
        record_ids=tuple(record_ids),
        rows_considered=len(candidates),
        rows_without_target=without_target,
    )

"""Validated configuration models (SPEC sections 3, 5, 11, 14).

Configuration is what makes the platform problem-agnostic. Traffic is the only
problem type implemented in v1, but it is not special-cased anywhere in the
code: it is one YAML file in ``configs/problems/``. Adding flood or waste later
means adding a file and enabling it, not editing modules.

Config is validated on load for the same reason data is: a typo in a unit or a
metric name must fail at startup, not three stages downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbansense.schemas.enums import (
    AggregationLevel,
    ProblemType,
    ReconciliationPolicy,
    Severity,
)


class MetricConfig(BaseModel):
    """Definition of one measurable metric within a problem domain.

    Attributes:
        name: Canonical metric name used in ``Observation.metric``.
        canonical_unit: Unit every value of this metric is normalized to.
        accepted_units: Units that may arrive and be converted to the canonical
            one. Anything else is an unknown unit and is quarantined rather than
            assumed (SPEC section 11).
        aliases: Source-side terminology mapped to ``name``, e.g. "vehicle
            count" and "vehicles observed" both mean ``traffic_volume``
            (SPEC section 5).
        non_negative: Whether negative values are physically impossible. Such
            values are quarantined, never clipped or dropped (SPEC section 14).
        plausible_min: Lower bound of the plausible range, for outlier flagging.
        plausible_max: Upper bound. Exceeding it flags an outlier for
            investigation -- it does not mark the value an error (SPEC section 16).
        default_aggregation: Aggregation level assumed when a source is silent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    accepted_units: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    non_negative: bool = True
    plausible_min: float | None = None
    plausible_max: float | None = None
    default_aggregation: AggregationLevel = AggregationLevel.HOURLY

    @model_validator(mode="after")
    def _canonical_unit_is_accepted(self) -> Self:
        """The canonical unit must itself be an accepted input unit."""
        if self.accepted_units and self.canonical_unit not in self.accepted_units:
            raise ValueError(
                f"metric {self.name!r}: canonical_unit {self.canonical_unit!r} is not "
                f"listed in accepted_units {self.accepted_units}."
            )
        return self

    @model_validator(mode="after")
    def _plausible_range_ordered(self) -> Self:
        """The plausible range must not be inverted."""
        if (
            self.plausible_min is not None
            and self.plausible_max is not None
            and self.plausible_max < self.plausible_min
        ):
            raise ValueError(
                f"metric {self.name!r}: plausible_max ({self.plausible_max}) is below "
                f"plausible_min ({self.plausible_min})."
            )
        return self


class ThresholdConfig(BaseModel):
    """How a zone's congestion threshold is computed (SPEC section 25).

    Attributes:
        method: Only ``percentile`` is implemented. Declared rather than assumed
            so a future absolute-value method is a config change, not a code
            branch added under time pressure.
        percentile: Which percentile of the zone's observed volume counts as
            congested.

    The threshold is always fitted on the **training period only**. That is not
    expressible in this file -- it is enforced by the fitting function's
    signature, which accepts one feature table and has no parameter through
    which validation or test data could arrive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["percentile"] = "percentile"
    percentile: float = Field(default=90.0, gt=0.0, lt=100.0)


class SeverityBand(BaseModel):
    """One severity band, keyed on the ratio of prediction to threshold.

    Attributes:
        name: The severity this band assigns.
        min_ratio: Lowest predicted-volume-to-threshold ratio in the band.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Severity
    min_ratio: float = Field(ge=0.0)


class CongestionConfig(BaseModel):
    """What counts as congestion, and how bad it is (SPEC section 25).

    Config-driven because "congested" is a policy decision, not a fact about
    the data: a six-lane arterial and a residential collector do not become
    problems at the same vehicle count, and a city that revises its definition
    should not need a code change.

    Attributes:
        metric: Metric the threshold applies to.
        threshold: How the per-zone threshold is computed.
        severity_bands: Bands in ascending order of ratio.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(min_length=1)
    threshold: ThresholdConfig = ThresholdConfig()
    severity_bands: tuple[SeverityBand, ...] = ()

    @model_validator(mode="after")
    def _bands_are_ordered_and_complete(self) -> Self:
        """Bands must start at zero and strictly increase.

        Monotonic severity is a property this phase asserts, so it is enforced
        where it can be guaranteed -- at load -- rather than left to the
        classifier to preserve. Overlapping or unordered bands would make the
        band assigned to a volume depend on iteration order.
        """
        if not self.severity_bands:
            raise ValueError("congestion: 'severity_bands' must list at least one band.")

        ratios = [band.min_ratio for band in self.severity_bands]
        if ratios != sorted(ratios) or len(set(ratios)) != len(ratios):
            raise ValueError(
                f"congestion: severity_bands must be in strictly ascending min_ratio "
                f"order, got {ratios}."
            )
        if ratios[0] != 0.0:
            raise ValueError(
                f"congestion: the first severity band must start at min_ratio 0.0 so "
                f"every ratio falls in a band, got {ratios[0]}."
            )

        ranks = [band.name.rank for band in self.severity_bands]
        if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
            raise ValueError(
                f"congestion: severity names must ascend with min_ratio, got "
                f"{[band.name.value for band in self.severity_bands]}."
            )
        return self

    def severity_for(self, ratio: float) -> Severity:
        """The band a prediction-to-threshold ratio falls in.

        Monotonic in ``ratio`` by construction: bands ascend, and the last one
        whose floor is reached wins.
        """
        chosen = self.severity_bands[0].name
        for band in self.severity_bands:
            if ratio >= band.min_ratio:
                chosen = band.name
            else:
                break
        return chosen


class ProblemConfig(BaseModel):
    """One urban problem domain and the metrics it is measured by.

    Attributes:
        problem_type: Which domain this file configures.
        enabled: Whether the problem participates in this run. Flood and waste
            ship disabled; v1 runs traffic only.
        description: What the domain covers.
        metrics: Metrics belonging to the domain, keyed by canonical name.
        congestion: What counts as congestion and how severe it is. Optional:
            only traffic defines it in v1, and a domain without the block simply
            has no congestion concept rather than an invented default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    problem_type: ProblemType
    enabled: bool = False
    description: str = ""
    metrics: tuple[MetricConfig, ...] = ()
    congestion: CongestionConfig | None = None

    @model_validator(mode="after")
    def _congestion_metric_is_declared(self) -> Self:
        """A threshold on an undeclared metric would never be computable."""
        if self.congestion is not None and self.metric(self.congestion.metric) is None:
            raise ValueError(
                f"problem {self.problem_type.value!r}: congestion.metric "
                f"{self.congestion.metric!r} is not one of this problem's metrics "
                f"{sorted(m.name for m in self.metrics)}."
            )
        return self

    @model_validator(mode="after")
    def _metric_names_unique(self) -> Self:
        """Duplicate metric names would make alias resolution ambiguous."""
        names = [metric.name for metric in self.metrics]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"problem {self.problem_type.value!r}: duplicate metric names {sorted(duplicates)}."
            )
        return self

    def metric(self, name: str) -> MetricConfig | None:
        """Return the metric definition for ``name``, or ``None``."""
        return next((metric for metric in self.metrics if metric.name == name), None)


class PathsConfig(BaseModel):
    """Filesystem layout.

    ``raw`` is append-only by project rule: nothing in the codebase may
    overwrite or delete inside it (SPEC section 4A).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_root: Path = Path("data")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    external: Path = Path("data/external")
    sample: Path = Path("data/sample")
    quarantine: Path = Path("quarantine")
    models: Path = Path("models")
    reports: Path = Path("reports")


class TemporalConfig(BaseModel):
    """Time handling and leakage guards (SPEC sections 12, 33).

    Attributes:
        display_timezone: Timezone used for human-facing output only. Stored
            instants are always UTC.
        future_tolerance_minutes: How far ahead of now an ``event_time`` may sit
            before it is treated as a clock error or leakage.
        min_prediction_gap_minutes: Minimum gap between the last feature
            observation and the prediction target, enforced by the leakage
            checks in later phases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_timezone: str = "Asia/Kolkata"
    future_tolerance_minutes: Annotated[int, Field(ge=0)] = 5
    min_prediction_gap_minutes: Annotated[int, Field(ge=0)] = 0


class ValidationConfig(BaseModel):
    """Thresholds for the integrity workflow (SPEC sections 14-18, 34, 35).

    Every threshold a detector uses lives here rather than in the detector, so
    tuning is a configuration change and a reviewer can see the whole policy in
    one place.

    Attributes:
        min_document_confidence: Below this, a document extraction is routed to
            human review instead of becoming a record.
        quarantine_on_unknown_unit: Whether an unrecognized unit quarantines the
            record. Never silently coerce.
        quarantine_on_unknown_location: Likewise for an unmapped location.
        min_frozen_run: Identical consecutive readings before a run counts as a
            frozen sensor. Six hours of a byte-identical vehicle count is not a
            quiet road (SPEC section 17).
        outlier_mad_threshold: Deviations from the group median, in MAD units,
            before a value is flagged suspicious. Deliberately conservative: a
            flag is only useful if it is rare.
        outlier_min_group_size: Minimum comparison-group size before a group may
            judge anything. Below it, median and MAD are noise.
        current_within_hours: Arrival lag at or under which data still describes
            current conditions.
        recent_within_hours: Arrival lag at or under which data is late but
            still informs the present. Beyond it, data is HISTORICAL and must
            never be presented as current (SPEC section 12).
        source_warning_gap_hours: Reporting gap beyond which a source is
            flagged for attention.
        source_failed_gap_hours: Reporting gap beyond which a source is
            considered failed.
        source_warning_missing_rate: Missing-value rate above which a source is
            flagged even with no long gap. Missing is counted, never zeroed.
        frozen_ignored_values: Values whose repetition is physically normal, so
            a run of them is not a frozen sensor. Zero by default: a rain gauge
            reporting 0 mm through a dry month is working correctly, and a
            detector on a closed road legitimately counts nothing for hours.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_document_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    quarantine_on_unknown_unit: bool = True
    quarantine_on_unknown_location: bool = True

    # --- sensor failure (SPEC section 17) -----------------------------
    min_frozen_run: Annotated[int, Field(ge=2)] = 6
    frozen_ignored_values: tuple[float, ...] = (0.0,)
    source_warning_gap_hours: Annotated[float, Field(gt=0.0)] = 2.0
    source_failed_gap_hours: Annotated[float, Field(gt=0.0)] = 6.0
    source_warning_missing_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10

    # --- outliers (SPEC section 16) -----------------------------------
    outlier_mad_threshold: Annotated[float, Field(gt=0.0)] = 6.0
    outlier_min_group_size: Annotated[int, Field(ge=2)] = 12

    # --- recency (SPEC section 12) ------------------------------------
    current_within_hours: Annotated[float, Field(ge=0.0)] = 2.0
    recent_within_hours: Annotated[float, Field(ge=0.0)] = 48.0

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> Self:
        """Recency bands and gap thresholds must not be inverted."""
        if self.recent_within_hours < self.current_within_hours:
            raise ValueError(
                f"recent_within_hours ({self.recent_within_hours}) must be at least "
                f"current_within_hours ({self.current_within_hours})"
            )
        if self.source_failed_gap_hours < self.source_warning_gap_hours:
            raise ValueError(
                f"source_failed_gap_hours ({self.source_failed_gap_hours}) must be at "
                f"least source_warning_gap_hours ({self.source_warning_gap_hours})"
            )
        return self


class ReconciliationConfig(BaseModel):
    """Thresholds and policy for duplicate and conflict handling (SPEC 9, 10).

    Attributes:
        conflict_relative_tolerance: Relative difference at or under which two
            values are taken to be the same measurement. Above it they are a
            material disagreement and the group becomes a conflict.
        conflict_absolute_tolerance: Absolute difference that always counts as
            agreement, as a floor. Without it, two near-zero readings differing
            by one rounding step would look like a 100% disagreement and
            manufacture conflicts out of quiet hours.
        conflict_policy: How conflicts are settled. Only manual review exists
            in v1 -- deliberately, since any automatic winner is a policy
            choice, and a policy that discards the values it rejects destroys
            the evidence that it was the wrong policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    conflict_relative_tolerance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    conflict_absolute_tolerance: Annotated[float, Field(ge=0.0)] = 0.5
    conflict_policy: ReconciliationPolicy = ReconciliationPolicy.MANUAL_REVIEW


class Settings(BaseModel):
    """Fully resolved configuration for one run.

    Attributes:
        env: Environment name, e.g. ``"local"`` or ``"ci"``.
        project_name: Display name.
        paths: Filesystem layout.
        temporal: Time handling and leakage guards.
        validation: Integrity thresholds.
        reconciliation: Duplicate and conflict handling policy.
        locations_file: Path to the canonical location registry, relative to the
            config directory (SPEC section 13).
        problems: Problem configurations, keyed by problem type.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    env: str = "local"
    project_name: str = "UrbanSense"
    paths: PathsConfig = PathsConfig()
    temporal: TemporalConfig = TemporalConfig()
    validation: ValidationConfig = ValidationConfig()
    reconciliation: ReconciliationConfig = ReconciliationConfig()
    locations_file: str = "locations/chennai_zones.yaml"
    problems: dict[ProblemType, ProblemConfig] = Field(default_factory=dict)

    @property
    def enabled_problems(self) -> tuple[ProblemType, ...]:
        """Problem types active in this run, in declaration order."""
        return tuple(key for key, problem in self.problems.items() if problem.enabled)

    def resolve_metric(self, problem_type: ProblemType, name: str) -> MetricConfig | None:
        """Resolve a canonical name *or alias* to its metric definition.

        This is the terminology mapping from SPEC section 5: a source saying
        "vehicle count" and one saying "traffic_volume" land on the same metric.
        Matching is case-insensitive and treats spaces and underscores alike,
        since source spellings vary in exactly that way.
        """
        problem = self.problems.get(problem_type)
        if problem is None:
            return None

        def normalize(text: str) -> str:
            return text.strip().lower().replace("-", " ").replace("_", " ")

        wanted = normalize(name)
        for metric in problem.metrics:
            candidates = {normalize(metric.name), *(normalize(a) for a in metric.aliases)}
            if wanted in candidates:
                return metric
        return None

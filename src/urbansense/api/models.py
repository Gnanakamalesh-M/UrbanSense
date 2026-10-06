"""Response models for the API (SPEC section 44).

    Keep the API modular and documented.

Typed responses rather than bare dicts, so ``/docs`` and ``/openapi.json``
describe the real shape of every payload. A reader can then see what an endpoint
returns without issuing a request, which is most of what "documented" means for
an API.

Two shapes carry most of the traffic:

:class:`Page`
    A paging envelope: the items, plus ``total``, ``limit`` and ``offset``. The
    total is the count **after filtering and before paging**, so a client can
    tell "there are 12 more" from "that is all of them".

:class:`UnifiedObservation`
    One unified reading **with its provenance**. The provenance is not optional
    and not a separate lookup: the question this project exists to answer is
    "where did this value come from?", and an endpoint that returned a number
    without its origin would be the wrong answer at the wrong layer.

Several payloads are passed through as mappings rather than fully re-typed --
pattern evidence, drift signals, experiment summaries. Those are written by
earlier phases with their own nested shapes, and re-declaring them here would
create a second schema to keep in step with the first. The envelope is typed,
the artifact's own structure is preserved.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

#: Items a page can hold at once. A request above this is a 422 rather than a
#: silent full scan: the unified export is tens of thousands of rows, and an
#: unbounded page would serialize all of them.
MAX_PAGE_LIMIT = 500
DEFAULT_PAGE_LIMIT = 50

ItemT = TypeVar("ItemT")


class Page(BaseModel, Generic[ItemT]):
    """A page of results.

    Attributes:
        items: This page's items.
        total: Matching items before paging, so a client knows whether more
            exist.
        limit: Items requested.
        offset: Items skipped.
        note: Why the page is empty, when it is and the reason is knowable --
            typically that the producing job has not been run yet. An empty page
            with a note is a better answer than a 404 or a 500, because "nothing
            has been computed" is true and actionable.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[ItemT]
    total: int
    limit: int
    offset: int
    note: str | None = None

    @property
    def has_more(self) -> bool:
        """Whether more items follow this page."""
        return self.offset + len(self.items) < self.total


class Health(BaseModel):
    """Service status and what it can actually serve.

    Attributes:
        status: ``ok`` when the process is up. It says nothing about the data.
        version: Package version.
        read_only: Always true. No endpoint mutates anything.
        authentication: Always ``none``. Stated in the payload rather than only
            in a README, because an unauthenticated service that does not say so
            is the more dangerous kind.
        bind_hint: Where the bundled script binds by default.
        artifacts: Which artifact families have something to serve, so an empty
            endpoint can be told from a broken one.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    read_only: bool
    authentication: str
    bind_hint: str
    artifacts: dict[str, bool]


class Provenance(BaseModel):
    """Where a value came from.

    Permissive by design: Phase 0 defines a direct and a document provenance
    shape, and this model carries whichever one the record has rather than
    forcing both into one schema.
    """

    model_config = ConfigDict(extra="allow")

    origin: str | None = None
    ingested_at: str | None = None
    pipeline_version: str | None = None


class UnifiedObservation(BaseModel):
    """One reading from the unified layer, with its origin.

    Attributes:
        observation_id: Canonical identifier.
        record_id: The source's own line id, for tracing back to the feed.
        source_id: Which source reported it.
        source_type: Which stream it arrived on.
        location_id: Zone.
        metric: What was measured.
        value: The measurement. ``None`` means missing, never zero.
        unit: Canonical unit.
        event_time: When the measurement happened.
        source_time: When the source stamped it.
        received_time: When it reached the system.
        validation_status: What validation concluded.
        is_imputed: Whether the value was filled. Always false in this project.
        duplicate_of: The record this duplicates, if any.
        conflicts_with: Records this disagrees with.
        provenance: Where the value came from.
    """

    model_config = ConfigDict(extra="forbid")

    observation_id: str
    record_id: str | None = None
    source_id: str
    source_type: str
    problem_type: str | None = None
    location_id: str
    metric: str
    value: float | None = None
    unit: str | None = None
    event_time: str
    source_time: str | None = None
    received_time: str
    aggregation_level: str | None = None
    validation_status: str
    is_imputed: bool = False
    duplicate_of: str | None = None
    conflicts_with: list[str] = Field(default_factory=list)
    provenance: Provenance


class PatternSummary(BaseModel):
    """One discovered pattern.

    The evidence block is passed through whole. A pattern is a claim, and its
    strength without its interval, p-value and controls cannot be judged -- so
    the endpoint serves the claim and its support together rather than a
    convenient summary of the claim alone.
    """

    model_config = ConfigDict(extra="allow")

    pattern_id: str
    kind: str
    description: str
    status: str | None = None
    confidence: float | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class PatternDetail(BaseModel):
    """One pattern plus the run that found it.

    Attributes:
        pattern: The pattern.
        run: The discovery run header -- hypotheses tested, FDR level, seed.
            Served with the pattern because one finding out of 672 tests is a
            different claim from one out of ten.
    """

    model_config = ConfigDict(extra="forbid")

    pattern: PatternSummary
    run: dict[str, Any] | None = None


class PredictionOutcome(BaseModel):
    """A recorded prediction and what actually happened."""

    model_config = ConfigDict(extra="allow")

    prediction_id: str
    model_version: str
    location_id: str
    target_time: str
    predicted: float
    actual: float | None = None
    error: float | None = None
    probability: float | None = None
    severity: str | None = None
    congested_predicted: bool | None = None
    congested_actual: bool | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)


class AnomalyRecord(BaseModel):
    """A flagged reading and its candidate explanations.

    ``is_error`` is always ``None``. A flagged reading may be a sensor fault, a
    festival or an accident, and this system does not pretend to know which --
    the field is carried through so the API cannot imply a verdict the pipeline
    declined to make.
    """

    model_config = ConfigDict(extra="allow")

    record_id: str
    location_id: str
    metric: str
    event_time: str
    value: float
    expected: float | None = None
    deviation_mad_sigma: float | None = None
    status: str | None = None
    is_error: None = None
    explanations: list[dict[str, Any]] = Field(default_factory=list)


class DriftEventRecord(BaseModel):
    """One drift evaluation.

    Every detector's verdict is kept, firing or not: knowing that feature drift
    was quiet while performance drift fired is what distinguishes a changed city
    from a model that is merely out of season.
    """

    model_config = ConfigDict(extra="allow")

    detected_at: str
    is_drift: bool
    kinds: list[str] = Field(default_factory=list)
    episode_id: str | None = None
    magnitude: float | None = None
    signals: list[dict[str, Any]] = Field(default_factory=list)


class DriftSummary(BaseModel):
    """Drift events, episodes and candidate decisions together.

    Attributes:
        events: Every evaluation.
        episodes: Runs of consecutive firings.
        candidates: Candidate updates and what was decided.
        note: Why the summary is empty, when it is.
    """

    model_config = ConfigDict(extra="forbid")

    events: list[DriftEventRecord]
    episodes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    note: str | None = None


class ModelSummary(BaseModel):
    """One model version's headline metadata."""

    model_config = ConfigDict(extra="allow")

    version: str
    status: str
    problem_type: str | None = None
    target: str | None = None
    horizon_hours: int | None = None
    algorithm: str | None = None
    created_at: str | None = None


class DataQuality(BaseModel):
    """Counts and per-source health.

    Attributes:
        rows_in: Input rows.
        valid: Usable observations.
        quarantined: Records held back.
        missing_values: Usable observations carrying no measurement. Reported
            separately because "58,000 valid rows" means something different
            when some of them have no value -- and they stay missing.
        source_health: Per-source status.
        note: Why the summary is unavailable, when it is.
    """

    model_config = ConfigDict(extra="allow")

    rows_in: int | None = None
    valid: int | None = None
    suspicious: int | None = None
    quarantined: int | None = None
    missing_values: int | None = None
    unified_rows: int | None = None
    duplicate_linked: int | None = None
    conflicted_observations: int | None = None
    quarantine_by_reason: dict[str, int] = Field(default_factory=dict)
    source_health: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class ExperimentSummary(BaseModel):
    """One experiment report.

    The pre-registered configuration and the verdict are both served, because a
    result without the parameters that were fixed in advance cannot be read as
    evidence.
    """

    model_config = ConfigDict(extra="allow")

    report_file: str
    generated_at: str | None = None
    pre_registered_config: dict[str, Any] = Field(default_factory=dict)
    verdict: dict[str, Any] = Field(default_factory=dict)
    arms: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

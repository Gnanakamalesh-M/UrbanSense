"""Shaping artifacts into what each page displays (SPEC section 43).

Every function here takes a :class:`~urbansense.dashboard.loaders.DashboardData`
and returns plain dicts and lists. **Nothing in this module imports streamlit**,
which is what makes the dashboard's data layer testable without launching a
server -- the tests call these functions directly and assert on the numbers.

The division is deliberate: :mod:`urbansense.dashboard.app` decides how
something looks, this module decides what it says. A page that computed its own
figures inline would be a second implementation of the pipeline's arithmetic,
drifting quietly from the reports.

Three rules the shaping obeys, inherited from the rest of the project:

**Missing stays missing.** A row with no measurement arrives as ``None`` and
leaves as ``None``. Nothing here substitutes a zero to make a chart tidier, so
"no data" and "a reading of 0" stay distinguishable on screen.

**Nothing is asserted that the artifact does not say.** The experiment view
reads the stored report's own verdict rather than recomputing a ranking, so the
dashboard cannot disagree with the report it is displaying -- including when the
finding is unflattering.

**Absence is reported, not hidden.** When a family has no artifact, the view
returns an empty payload carrying the loader's note, which names the command
that would produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from urbansense.dashboard.loaders import DashboardData

#: Shown on every page. The single most important caveat in the project: every
#: number on screen comes from a generator, not a city.
SYNTHETIC_BANNER = (
    "All data here is **synthetic**, produced by this repository's own generator "
    "with known planted ground truth. No real city, sensor feed or person is "
    "involved, and every figure is a simulation result rather than a measurement "
    "or a forecast."
)

#: Severity bands, worst first, for ordering a display.
SEVERITY_ORDER = ("severe", "high", "moderate", "low", "none")


@dataclass(frozen=True)
class Panel:
    """One page's data, plus why it might be empty.

    Attributes:
        rows: The records to display.
        note: The loader's explanation when there is nothing, naming the command
            that would produce the artifact. ``None`` when data is present.
        extra: Page-specific headline figures.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to show."""
        return not self.rows

    def __len__(self) -> int:
        """Rows available."""
        return len(self.rows)


def _mean(values: list[float]) -> float | None:
    """Mean, or ``None`` for an empty list.

    ``None`` rather than 0.0: an average of nothing is not zero, and a chart
    that drew it as zero would invent a reading.
    """
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# City Overview
# ---------------------------------------------------------------------------


def zone_risk(data: DashboardData) -> Panel:
    """Per-zone risk, from the recorded predictions.

    "Risk" is the mean calibrated congestion probability over the stored
    outcomes for that zone -- not a fresh model call. These are the predictions
    the system actually made, which is the only version of a prediction that can
    be checked against what happened.

    Zones with no recorded prediction keep their row with ``risk=None``. Dropping
    them would make a zone silently disappear from the overview, which reads as
    a working dashboard showing three zones rather than a missing artifact.
    """
    rows = data.predictions()
    coordinates = {zone["location_id"]: zone for zone in data.zones() if zone.get("location_id")}

    by_zone: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        zone = row.get("location_id")
        if isinstance(zone, str):
            by_zone.setdefault(zone, []).append(row)

    names = sorted(set(by_zone) | set(coordinates))
    shaped: list[dict[str, Any]] = []
    for zone in names:
        found = by_zone.get(zone, [])
        probabilities = [
            float(row["probability"])
            for row in found
            if isinstance(row.get("probability"), (int, float))
        ]
        errors = [
            abs(float(row["error"])) for row in found if isinstance(row.get("error"), (int, float))
        ]
        geometry = coordinates.get(zone, {})
        shaped.append(
            {
                "location_id": zone,
                "description": geometry.get("description", ""),
                "latitude": geometry.get("latitude"),
                "longitude": geometry.get("longitude"),
                "risk": _mean(probabilities),
                "mae": _mean(errors),
                "predictions": len(found),
                "congested_hours": sum(1 for row in found if row.get("congested_actual") is True),
            }
        )

    return Panel(
        rows=shaped,
        note=data.missing_note("predictions") if not rows else None,
        extra={"zones": len(shaped), "scored_predictions": len(rows)},
    )


def map_points(data: DashboardData) -> list[dict[str, float]]:
    """Zone points for the map, in the column names ``st.map`` expects.

    Only zones that have coordinates. A zone without them is still listed in
    :func:`zone_risk`; it simply cannot be placed, which is why the map is a
    companion to that table rather than a replacement for it.
    """
    points: list[dict[str, float]] = []
    for row in zone_risk(data).rows:
        latitude, longitude = row.get("latitude"), row.get("longitude")
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
            risk = row.get("risk")
            # Marker area reads as risk. A zone with no prediction gets the floor
            # rather than vanishing from the map.
            scale = float(risk) if isinstance(risk, (int, float)) else 0.0
            points.append(
                {
                    "lat": float(latitude),
                    "lon": float(longitude),
                    "size": 120.0 + 600.0 * scale,
                }
            )
    return points


def overview_headline(data: DashboardData) -> dict[str, Any]:
    """The figures across the top of City Overview.

    Each one is read from an artifact, and each is ``None`` when its artifact is
    absent -- so a fresh clone shows blanks with explanations rather than a row
    of confident zeros.
    """
    quality = data.data_quality()
    champion = next(
        (model for model in data.models() if model.get("status") == "champion"),
        None,
    )
    events = data.drift().get("events", [])
    firing = [event for event in events if isinstance(event, dict) and event.get("is_drift")]

    return {
        "champion": champion.get("version") if champion else None,
        "champion_created": champion.get("created_at") if champion else None,
        "model_versions": len(data.models()),
        "anomalies": len(data.anomalies()),
        "unexplained_anomalies": sum(
            1
            for item in data.anomalies()
            if any(
                candidate.get("kind") == "unexplained"
                for candidate in item.get("explanations", [])
                if isinstance(candidate, dict)
            )
        ),
        "drift_evaluations": len(events),
        "drift_firings": len(firing),
        "quarantined": quality.get("quarantined") if quality else None,
        "missing_values": quality.get("missing_values") if quality else None,
        "unified_rows": quality.get("unified_rows") if quality else None,
        "conflicts": len(data.conflicts()),
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def prediction_rows(
    data: DashboardData,
    *,
    zone: str | None = None,
    date: str | None = None,
) -> Panel:
    """Recorded predictions for one zone and date.

    These are **logged predictions, not fresh inference**: each was made by
    whichever model was champion at the time, and is shown with what actually
    happened where that is known. An unresolved prediction keeps ``actual=None``
    and ``error=None``.
    """
    rows = [
        row
        for row in data.predictions(zone=zone)
        if date is None or str(row.get("target_time", "")).startswith(date)
    ]
    rows.sort(key=lambda row: str(row.get("target_time", "")))

    shaped = [
        {
            "target_time": row.get("target_time"),
            "location_id": row.get("location_id"),
            "model_version": row.get("model_version"),
            "predicted": row.get("predicted"),
            "actual": row.get("actual"),
            "error": row.get("error"),
            "probability": row.get("probability"),
            "severity": row.get("severity"),
            "congested_predicted": row.get("congested_predicted"),
            "congested_actual": row.get("congested_actual"),
            "conditions": row.get("conditions", {}),
        }
        for row in rows
    ]

    resolved = [row for row in shaped if isinstance(row["error"], (int, float))]
    return Panel(
        rows=shaped,
        note=data.missing_note("predictions") if not shaped else None,
        extra={
            "total": len(shaped),
            "resolved": len(resolved),
            "pending": len(shaped) - len(resolved),
            "mae": _mean([abs(float(row["error"])) for row in resolved]),
            "mean_probability": _mean(
                [
                    float(row["probability"])
                    for row in shaped
                    if isinstance(row["probability"], (int, float))
                ]
            ),
        },
    )


def prediction_zones(data: DashboardData) -> list[str]:
    """Zones that have recorded predictions, for the picker."""
    found = {
        row["location_id"] for row in data.predictions() if isinstance(row.get("location_id"), str)
    }
    return sorted(found)


def prediction_dates(data: DashboardData, *, zone: str | None = None) -> list[str]:
    """Target dates that have recorded predictions, for the picker."""
    found = {
        str(row["target_time"])[:10]
        for row in data.predictions(zone=zone)
        if isinstance(row.get("target_time"), str)
    }
    return sorted(found)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def pattern_rows(data: DashboardData) -> Panel:
    """Discovered patterns with the multiple-testing context attached.

    The run header travels with the findings because a list of patterns cannot
    be judged without the number of hypotheses behind it -- one finding out of
    several hundred tests is a different claim from one out of ten. It is in
    ``extra`` so the page can print it above the table.
    """
    found = data.patterns()
    run = data.pattern_run() or {}

    shaped = []
    for item in found:
        evidence = item.get("evidence", {})
        effect = evidence.get("effect", {}) if isinstance(evidence, dict) else {}
        shaped.append(
            {
                "pattern_id": item.get("pattern_id"),
                "kind": item.get("kind"),
                "status": item.get("status"),
                "description": item.get("description"),
                "strength": item.get("strength"),
                "confidence": item.get("confidence"),
                "effect": effect.get("value") if isinstance(effect, dict) else None,
                "lower": effect.get("lower") if isinstance(effect, dict) else None,
                "upper": effect.get("upper") if isinstance(effect, dict) else None,
                "first_detected": item.get("first_detected"),
                "last_observed": item.get("last_observed"),
            }
        )

    by_status: dict[str, int] = {}
    for item in shaped:
        status = str(item.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    return Panel(
        rows=shaped,
        note=data.missing_note("patterns") if not shaped else None,
        extra={
            "tests_run": run.get("tests_run"),
            "fdr_level": run.get("fdr_level"),
            "discoveries": run.get("discoveries"),
            "cells_too_sparse": run.get("cells_too_sparse"),
            "by_status": by_status,
        },
    )


def anomaly_rows(data: DashboardData) -> Panel:
    """Flagged readings with their ranked candidate explanations.

    ``is_error`` is carried through as ``None`` on every record. A flagged
    reading may be a sensor fault, a festival or a genuine extreme, and the
    pipeline declines to decide -- so the dashboard must not imply a verdict
    either.
    """
    found = data.anomalies()
    shaped = [
        {
            "record_id": item.get("record_id"),
            "location_id": item.get("location_id"),
            "event_time": item.get("event_time"),
            "metric": item.get("metric"),
            "value": item.get("value"),
            "expected": item.get("expected"),
            "deviation_mad_sigma": item.get("deviation_mad_sigma"),
            "is_error": None,
            "leading_explanation": _leading_explanation(item),
            "candidates": len(item.get("explanations", []) or []),
        }
        for item in found
    ]
    return Panel(
        rows=shaped,
        note=data.missing_note("anomalies") if not shaped else None,
        extra={"total": len(shaped), "verdict": "none: is_error is null on every record"},
    )


def _leading_explanation(item: dict[str, Any]) -> str | None:
    """The best-supported candidate's kind. Not a conclusion."""
    explanations = item.get("explanations") or []
    for candidate in explanations:
        if isinstance(candidate, dict) and candidate.get("kind"):
            return str(candidate["kind"])
    return None


# ---------------------------------------------------------------------------
# Data Health
# ---------------------------------------------------------------------------


def source_health(data: DashboardData) -> Panel:
    """Per-source health, **before** any unified figure.

    Source-specific views precede the unified one (SPEC section 20) so a reader
    can always see which stream a number came from. A healthy total hides an
    unhealthy feed, and the point of this page is to stop that happening.
    """
    quality = data.data_quality()
    if quality is None:
        return Panel(note=data.missing_note("data_quality"))

    health = quality.get("source_health", {})
    rows = [
        {
            "source_id": source_id,
            "status": detail.get("status"),
            "observations": detail.get("observations"),
            "missing": detail.get("missing"),
            "frozen_runs": detail.get("frozen_runs"),
            "frozen_observations": detail.get("frozen_observations"),
            "last_observation_time": detail.get("last_observation_time"),
            "max_gap_hours": detail.get("max_gap_hours"),
            "detail": detail.get("detail"),
        }
        for source_id, detail in sorted(health.items())
        if isinstance(detail, dict)
    ]
    return Panel(
        rows=rows,
        extra={"unhealthy": [row["source_id"] for row in rows if row["status"] != "healthy"]},
    )


def data_health(data: DashboardData) -> dict[str, Any]:
    """The unified counts, shown after the per-source table.

    ``missing_values`` is reported beside ``valid`` rather than folded into it:
    a usable observation may carry no measurement, and it stays missing.
    """
    quality = data.data_quality()
    quarantine = data.quarantine()

    by_reason: dict[str, int] = {}
    for record in quarantine:
        reason = str(record.get("reason") or "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1
    if quality and not by_reason:
        declared = quality.get("quarantine_by_reason", {})
        if isinstance(declared, dict):
            by_reason = {str(key): int(value) for key, value in declared.items()}

    return {
        "note": data.missing_note("data_quality") if quality is None else None,
        "rows_in": quality.get("rows_in") if quality else None,
        "valid": quality.get("valid") if quality else None,
        "suspicious": quality.get("suspicious") if quality else None,
        "quarantined": quality.get("quarantined") if quality else len(quarantine) or None,
        "missing_values": quality.get("missing_values") if quality else None,
        "unified_rows": quality.get("unified_rows") if quality else None,
        "duplicate_linked": quality.get("duplicate_linked") if quality else None,
        "conflicted_observations": quality.get("conflicted_observations") if quality else None,
        "conflict_groups": len(data.conflicts()),
        "quarantine_by_reason": by_reason,
        "missing_policy": "Missing values are counted, never filled. No value becomes a zero.",
    }


def conflict_rows(data: DashboardData) -> Panel:
    """Recorded disagreements, every candidate value kept.

    A conflict is not resolved by overwriting: the disputed slot is a hole in
    the unified layer plus this record of what each source claimed.
    """
    found = data.conflicts()
    shaped = [
        {
            "location_id": group.get("location_id"),
            "metric": group.get("metric"),
            "event_time": group.get("event_time"),
            "sources": ", ".join(
                str(candidate.get("source_id"))
                for candidate in group.get("candidates", []) or []
                if isinstance(candidate, dict)
            ),
            "values": ", ".join(
                str(candidate.get("value"))
                for candidate in group.get("candidates", []) or []
                if isinstance(candidate, dict)
            ),
            "candidates": len(group.get("candidates", []) or []),
        }
        for group in found
    ]
    return Panel(rows=shaped, note=data.missing_note("conflicts") if not shaped else None)


def unified_sample(data: DashboardData, *, zone: str | None = None, limit: int = 200) -> Panel:
    """A slice of the unified layer, provenance included.

    Shown after the source-specific views, and every row carries where its value
    came from -- the question the project exists to answer.
    """
    rows = data.unified(zone=zone, limit=limit)
    shaped = [
        {
            "event_time": row.get("event_time"),
            "location_id": row.get("location_id"),
            "metric": row.get("metric"),
            # Stays None when the measurement is missing.
            "value": row.get("value"),
            "unit": row.get("unit"),
            "source_id": row.get("source_id"),
            "source_type": row.get("source_type"),
            "is_imputed": row.get("is_imputed", False),
            "origin": (row.get("provenance") or {}).get("origin"),
            "pipeline_version": (row.get("provenance") or {}).get("pipeline_version"),
            "received_time": row.get("received_time"),
        }
        for row in rows
    ]
    return Panel(
        rows=shaped,
        note=data.missing_note("unified") if not shaped else None,
        extra={"missing_values": sum(1 for row in shaped if row["value"] is None)},
    )


# ---------------------------------------------------------------------------
# Model Evolution
# ---------------------------------------------------------------------------


def model_timeline(data: DashboardData) -> Panel:
    """Model versions oldest first, with exactly one champion.

    Rejected challengers keep their rows. Knowing which updates failed, and by
    how much, is the experimental record -- a timeline of successes only would
    be a sales pitch.
    """
    found = data.models()
    shaped = [
        {
            "version": model.get("version"),
            "status": model.get("status"),
            "created_at": model.get("created_at"),
            "algorithm": model.get("algorithm"),
            "horizon_hours": model.get("horizon_hours"),
            "test_mae": _registry_metric(model, "mae"),
            "test_rmse": _registry_metric(model, "rmse"),
            "gap_hours": model.get("gap_hours"),
        }
        for model in found
    ]
    champions = [row for row in shaped if row["status"] == "champion"]
    return Panel(
        rows=shaped,
        note=data.missing_note("models") if not shaped else None,
        extra={
            "champion": champions[0]["version"] if champions else None,
            "champion_count": len(champions),
            "versions": len(shaped),
        },
    )


def _registry_metric(model: dict[str, Any], name: str) -> float | None:
    """One metric from a registry entry's test block, when present."""
    metrics = model.get("metrics")
    if not isinstance(metrics, dict):
        return None
    segment = metrics.get("test")
    if isinstance(segment, dict) and isinstance(segment.get(name), (int, float)):
        return float(segment[name])
    if isinstance(metrics.get(name), (int, float)):
        return float(metrics[name])
    return None


def drift_rows(data: DashboardData) -> Panel:
    """Every drift evaluation, firing or quiet.

    The quiet ones are kept on purpose: a false-alarm rate cannot be read from
    firings alone, and a page that showed only detections would make a calm
    detector look idle.
    """
    payload = data.drift()
    events = [event for event in payload.get("events", []) if isinstance(event, dict)]
    shaped = [
        {
            "detected_at": event.get("detected_at"),
            "is_drift": bool(event.get("is_drift")),
            "kinds": ", ".join(str(kind) for kind in event.get("kinds", []) or []) or "-",
            "magnitude": event.get("magnitude"),
            "episode_id": event.get("episode_id"),
            "subject": event.get("strongest_subject"),
        }
        for event in events
    ]
    return Panel(
        rows=shaped,
        note=payload.get("note") or (data.missing_note("drift") if not shaped else None),
        extra={
            "evaluations": len(shaped),
            "firings": sum(1 for row in shaped if row["is_drift"]),
            "episodes": len(payload.get("episodes", []) or []),
        },
    )


def candidate_rows(data: DashboardData) -> Panel:
    """Candidate updates and what was decided about each.

    A drift event never retrains on its own: it raises a candidate, which is
    then judged against the config-declared promotion rules. Both outcomes are
    shown, because a rejected candidate is the system working.
    """
    payload = data.drift()
    found = [item for item in payload.get("candidates", []) if isinstance(item, dict)]
    shaped = [
        {
            "candidate_id": item.get("candidate_id"),
            "raised_at": item.get("raised_at"),
            "decision_time": item.get("decision_time"),
            "status": item.get("status"),
            "champion_version": item.get("champion_version"),
            "promoted_version": item.get("promoted_version"),
            "challengers": len(item.get("evaluations", []) or []),
            "notes": "; ".join(str(note) for note in item.get("notes", []) or []),
        }
        for item in found
    ]
    promoted = [row for row in shaped if row["promoted_version"]]
    return Panel(
        rows=shaped,
        note=data.missing_note("drift") if not shaped else None,
        extra={
            "candidates": len(shaped),
            "promoted": len(promoted),
            "rejected": len(shaped) - len(promoted),
        },
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment_summary(data: DashboardData) -> Panel:
    """The three-arm comparison, as the stored report states it.

    **The verdict and the ranking are read from the report, never recomputed
    here.** That is what stops the dashboard flattering the project: the Phase 8
    result is that a 28-day calendar retrain matched or beat the adaptive
    system, and a page that derived its own ordering could quietly present a
    different story from the one the experiment recorded.

    The cost columns sit beside the accuracy columns for the same reason. An
    accuracy table on its own lets the most expensive arm look best for free.
    """
    reports = data.experiments()
    if not reports:
        return Panel(note=data.missing_note("experiments"))

    report = reports[0]
    arms = report.get("arms", {})
    config = report.get("pre_registered_config", {})
    verdict = report.get("verdict", {})

    ranking = verdict.get("ranking_by_mae") or []
    ordered = [name for name in ranking if name in arms] + [
        name for name in arms if name not in ranking
    ]

    rows = []
    for name in ordered:
        arm = arms.get(name)
        if not isinstance(arm, dict):
            continue
        congestion = arm.get("congestion", {})
        cost = arm.get("cost", {})
        rows.append(
            {
                "arm": name,
                "mae": _aggregate(arm.get("mae")),
                "mae_sd": _aggregate(arm.get("mae"), "sd"),
                "rmse": _aggregate(arm.get("rmse")),
                "f1": _aggregate(congestion.get("f1") if isinstance(congestion, dict) else None),
                "brier": _aggregate(arm.get("brier")),
                "retrains": _aggregate(cost.get("retrains") if isinstance(cost, dict) else None),
                "fits": _aggregate(cost.get("fits") if isinstance(cost, dict) else None),
                "training_rows": _aggregate(
                    cost.get("training_rows") if isinstance(cost, dict) else None
                ),
            }
        )

    return Panel(
        rows=rows,
        extra={
            "report_file": report.get("report_file"),
            "generated_at": report.get("generated_at"),
            "pre_registered": config.get("pre_registered"),
            "seeds": config.get("seeds", []),
            "retrain_every_days": config.get("retrain_every_days"),
            # Straight from the report. Not re-derived, not softened.
            "conclusion": verdict.get("conclusion"),
            "best": verdict.get("best"),
            "ranking": list(ranking),
            "ties": [list(pair) for pair in verdict.get("ties", []) or []],
            "regressions": report.get("regressions", {}),
            "limitations": list(report.get("limitations", []) or []),
        },
    )


def _aggregate(block: object, key: str = "mean") -> float | None:
    """One figure out of a report's ``{mean, sd, seeds, values}`` block."""
    if isinstance(block, dict) and isinstance(block.get(key), (int, float)):
        return float(block[key])
    if key == "mean" and isinstance(block, (int, float)):
        return float(block)
    return None


def experiment_slices(data: DashboardData) -> Panel:
    """Per-slice MAE for every arm, including where the best arm lost.

    The counterweight to the headline table: an arm can win overall and still
    be the worst choice for the hours somebody actually cares about.
    """
    reports = data.experiments()
    if not reports:
        return Panel(note=data.missing_note("experiments"))

    arms = reports[0].get("arms", {})
    names = [name for name in arms if isinstance(arms[name], dict)]

    slice_names: list[str] = []
    for name in names:
        for slice_name in arms[name].get("slice_mae", {}):
            if slice_name not in slice_names:
                slice_names.append(slice_name)

    rows = []
    for slice_name in sorted(slice_names):
        row: dict[str, Any] = {"slice": slice_name}
        for name in names:
            row[name] = _aggregate(arms[name].get("slice_mae", {}).get(slice_name))
        rows.append(row)

    return Panel(
        rows=rows,
        extra={"arms": names, "regressions": reports[0].get("regressions", {})},
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def readiness(data: DashboardData) -> list[dict[str, Any]]:
    """Every artifact family, whether it is present, and what would fill it.

    Drives the sidebar, so an empty page is always distinguishable from a broken
    one without reading a log.
    """
    available = data.availability()
    return [
        {
            "family": family,
            "ready": bool(present),
            "note": None if present else data.missing_note(family),
        }
        for family, present in sorted(available.items())
    ]

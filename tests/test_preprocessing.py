"""Tests for normalization: units, recency and the location registry.

These pin the three things normalization must never do: coerce an unknown unit,
discard the original value, or guess at a location that merely resembles a known
one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from urbansense.config import load_settings
from urbansense.preprocessing import (
    LocationRegistry,
    LocationRegistryError,
    MatchKind,
    UnitError,
    ZoneEntry,
    are_compatible,
    arrival_lag,
    canonicalize_unit,
    classify_recency,
    convert,
    dimension_of,
    is_historical,
    is_known_unit,
    known_units,
    load_location_registry,
    normalize_name,
    normalize_value,
)
from urbansense.schemas.enums import ProblemType, Recency

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Unit conversion (SPEC section 11)
# ---------------------------------------------------------------------------


def test_flow_conversion_scales_by_interval():
    assert convert(2050, "vehicles/15min", "vehicles/hour") == pytest.approx(8200.0)
    assert convert(100, "vehicles/5min", "vehicles/hour") == pytest.approx(1200.0)
    assert convert(24000, "vehicles/day", "vehicles/hour") == pytest.approx(1000.0)


def test_speed_conversion():
    assert convert(36, "km/hour", "m/s") == pytest.approx(10.0)
    assert convert(10, "m/s", "km/hour") == pytest.approx(36.0)
    assert convert(60, "mph", "km/hour") == pytest.approx(96.56064)


def test_ratio_and_length_and_depth_conversions():
    assert convert(50, "percent", "ratio") == pytest.approx(0.5)
    assert convert(1, "kilometre", "metre") == pytest.approx(1000.0)
    assert convert(2, "inch", "mm") == pytest.approx(50.8)


@pytest.mark.parametrize(
    ("from_unit", "to_unit", "value"),
    [
        ("vehicles/15min", "vehicles/hour", 2050.0),
        ("m/s", "km/hour", 13.75),
        ("percent", "ratio", 42.0),
        ("foot", "metre", 120.0),
        ("inch", "mm", 1.5),
    ],
)
def test_conversion_round_trips(from_unit, to_unit, value):
    """Converting there and back must return the original value."""
    there = convert(value, from_unit, to_unit)
    back = convert(there, to_unit, from_unit)
    assert back == pytest.approx(value)


def test_same_unit_conversion_is_identity():
    assert convert(8200.0, "vehicles/hour", "vehicles/hour") == 8200.0


def test_unit_spelling_is_canonicalized():
    assert canonicalize_unit(" KM/H ") == "km/hour"
    assert canonicalize_unit("vph") == "vehicles/hour"
    assert canonicalize_unit("%") == "percent"
    assert convert(36, "km/h", "m/s") == pytest.approx(10.0)


def test_unknown_unit_raises_rather_than_coercing():
    """There is no "probably vehicles per hour" fallback."""
    with pytest.raises(UnitError, match="unknown unit"):
        convert(1.0, "vehicles/fortnight", "vehicles/hour")


def test_incompatible_dimensions_raise():
    """A queue length is not a speed; converting one to the other is a bug."""
    with pytest.raises(UnitError, match="different dimensions"):
        convert(1.0, "metre", "km/hour")


def test_dimension_and_compatibility_helpers():
    assert dimension_of("vehicles/15min") == "flow"
    assert dimension_of("nonsense") is None
    assert are_compatible("m/s", "mph")
    assert not are_compatible("m/s", "metre")
    assert is_known_unit("km/h")
    assert not is_known_unit("furlongs/fortnight")
    assert "vehicles/hour" in known_units()


def test_normalize_value_keeps_the_original(generator_config=None):
    """Both representations survive, so the conversion stays auditable."""
    result = normalize_value(2050.0, "vehicles/15min", "vehicles/hour")
    assert result.value == pytest.approx(8200.0)
    assert result.unit == "vehicles/hour"
    assert result.original_value == 2050.0
    assert result.original_unit == "vehicles/15min"
    assert result.converted is True


def test_normalize_value_marks_no_op_conversions():
    result = normalize_value(8200.0, "vehicles/hour", "vehicles/hour")
    assert result.converted is False
    assert result.value == 8200.0


def test_normalize_value_keeps_missing_missing():
    """A missing reading is converted in name only; it never becomes a number."""
    result = normalize_value(None, "vehicles/15min", "vehicles/hour")
    assert result.value is None
    assert result.value != 0
    assert result.original_value is None
    assert result.unit == "vehicles/hour"
    assert result.original_unit == "vehicles/15min"


def test_normalize_value_rejects_unknown_unit_even_when_missing():
    """An unknown unit on an absent reading is still a feed we do not grasp."""
    with pytest.raises(UnitError):
        normalize_value(None, "vehicles/fortnight", "vehicles/hour")


def test_every_configured_accepted_unit_is_convertible():
    """Config and the conversion registry must not drift apart.

    If a problem config accepted a unit this module cannot convert, every row
    using it would be quarantined and the cause would look like bad data rather
    than a missing conversion.
    """
    settings = load_settings()
    for problem in settings.problems.values():
        for metric in problem.metrics:
            for unit in metric.accepted_units:
                assert is_known_unit(unit), (
                    f"{problem.problem_type.value}/{metric.name} accepts {unit!r} "
                    "but the conversion registry does not know it"
                )
                assert are_compatible(unit, metric.canonical_unit), (
                    f"{problem.problem_type.value}/{metric.name}: {unit!r} is not "
                    f"convertible to the canonical {metric.canonical_unit!r}"
                )


def test_generated_rainfall_unit_is_convertible():
    """The rainfall covariate is not in the traffic config but must still parse."""
    assert is_known_unit("mm")
    assert convert(10, "mm", "cm") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Recency (SPEC section 12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lag", "expected"),
    [
        (timedelta(0), Recency.CURRENT),
        (timedelta(minutes=30), Recency.CURRENT),
        (timedelta(hours=2), Recency.CURRENT),
        (timedelta(hours=3), Recency.RECENT),
        (timedelta(hours=48), Recency.RECENT),
        (timedelta(hours=49), Recency.HISTORICAL),
        (timedelta(days=400), Recency.HISTORICAL),
    ],
)
def test_recency_bands(lag, expected):
    assert classify_recency(NOW - lag, NOW) is expected


def test_old_report_received_today_is_historical():
    """A 2022 report uploaded in 2026 must never read as current data."""
    event = datetime(2022, 6, 1, 18, 0, tzinfo=UTC)
    assert classify_recency(event, NOW) is Recency.HISTORICAL
    assert is_historical(event, NOW)


def test_fresh_reading_is_current():
    event = NOW - timedelta(minutes=4)
    assert classify_recency(event, NOW) is Recency.CURRENT
    assert not is_historical(event, NOW)


def test_bands_are_configurable():
    event = NOW - timedelta(hours=5)
    assert classify_recency(event, NOW, current_within=timedelta(hours=6)) is Recency.CURRENT


def test_arrival_lag_is_signed():
    assert arrival_lag(NOW - timedelta(hours=3), NOW) == timedelta(hours=3)
    assert arrival_lag(NOW, NOW - timedelta(hours=3)) == timedelta(hours=-3)


def test_negative_lag_does_not_read_as_stale():
    """A clock error is not evidence of age.

    Receipt before the event is handled by the timestamp validators on its own
    terms; it must not reappear here as a freshness claim.
    """
    assert classify_recency(NOW + timedelta(hours=1), NOW) is Recency.CURRENT


# ---------------------------------------------------------------------------
# Location registry (SPEC section 13)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    return load_location_registry()


def test_shipped_registry_loads(registry):
    assert len(registry) >= 5
    assert registry.get("ZONE-B") is not None


def test_canonical_id_resolves(registry):
    match = registry.resolve("ZONE-B")
    assert match.resolved
    assert match.location_id == "ZONE-B"
    assert match.kind is MatchKind.CANONICAL_ID
    assert match.confidence == 1.0


@pytest.mark.parametrize("spelling", ["zone-b", "ZONE_B", "  Zone B  ", "zone b"])
def test_separator_and_case_differences_resolve(registry, spelling):
    """Case and separator choice are spelling noise, not meaning."""
    assert registry.resolve(spelling).location_id == "ZONE-B"


def test_display_name_resolves(registry):
    match = registry.resolve("Anna Nagar")
    assert match.resolved
    assert match.location_id == "ZONE-B"
    assert match.kind is MatchKind.NAME


def test_curated_alias_resolves(registry):
    match = registry.resolve("zone-17")
    assert match.resolved
    assert match.location_id == "ZONE-B"
    assert match.kind is MatchKind.ALIAS


def test_similar_but_different_name_is_not_auto_matched(registry):
    """The core rule: similar is not identical (SPEC section 13).

    "Anna Nagar East" scores high string similarity against "Anna Nagar" but is
    a different area. Auto-matching them would merge two zones' traffic
    permanently and invisibly, so it must go to review instead.
    """
    match = registry.resolve("Anna Nagar East")
    assert not match.resolved
    assert match.location_id is None
    assert match.kind is MatchKind.UNRESOLVED
    assert match.confidence == 0.0


def test_a_high_similarity_suggestion_is_advisory_only(registry):
    """The similarity score helps a reviewer; it never causes acceptance."""
    match = registry.resolve("Anna Nagar East")
    assert match.suggestions, "expected a candidate to be offered to the reviewer"
    best_id, best_score = match.suggestions[0]
    assert best_id == "ZONE-B"
    assert best_score > 0.75, "similarity really is high, which is the point"
    assert not match.resolved, "high similarity must still not resolve"
    assert "similar is not identical" in match.detail


@pytest.mark.parametrize(
    "unknown",
    ["Anna Nagar West", "Zone-99", "Velachery", "ZONE-E", ""],
)
def test_unknown_locations_do_not_resolve(registry, unknown):
    assert not registry.resolve(unknown).resolved


def test_unknown_location_with_no_lookalike_says_so(registry):
    match = registry.resolve("Reykjavik")
    assert not match.resolved
    assert match.suggestions == ()
    assert "resembles no known zone" in match.detail


def test_resolved_match_carries_coordinates(registry):
    match = registry.resolve("Anna Nagar")
    assert match.zone is not None
    assert match.zone.latitude == pytest.approx(13.0604)
    assert match.zone.longitude == pytest.approx(80.2496)


def test_registry_rejects_an_alias_claimed_by_two_zones():
    """An ambiguous alias is a config error, not a runtime tie to break.

    Picking one would silently assign a road's traffic to the wrong zone.
    """
    with pytest.raises(ValueError, match="claimed by both"):
        LocationRegistry(
            zones=(
                ZoneEntry(location_id="ZONE-A", name="Alpha", aliases=("shared",)),
                ZoneEntry(location_id="ZONE-B", name="Beta", aliases=("shared",)),
            )
        )


def test_registry_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate location_ids"):
        LocationRegistry(
            zones=(
                ZoneEntry(location_id="ZONE-A", name="Alpha"),
                ZoneEntry(location_id="ZONE-A", name="Also Alpha"),
            )
        )


def test_registry_covers_every_zone_the_demo_data_uses():
    """A zone in the data but not the registry would quarantine the whole feed."""
    import csv
    from pathlib import Path

    registry = load_location_registry()
    path = Path("data/sample/traffic_history.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        seen = {row["location_id"] for row in csv.DictReader(handle)}
    for location_id in sorted(seen):
        assert registry.resolve(location_id).resolved, (
            f"{location_id} appears in the demo data but is not in the registry"
        )


def test_normalize_name_folds_only_spelling():
    assert normalize_name("  ZONE_B ") == "zone b"
    assert normalize_name("Anna-Nagar") == "anna nagar"
    # The distinguishing word survives, which is what keeps the zones separate.
    assert normalize_name("Anna Nagar East") != normalize_name("Anna Nagar")


def test_missing_registry_file_is_an_error(tmp_path):
    with pytest.raises(LocationRegistryError, match="not found"):
        load_location_registry(tmp_path / "nope.yaml")


def test_malformed_registry_is_an_error(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text("zones: [\n", encoding="utf-8")
    with pytest.raises(LocationRegistryError, match="not valid YAML"):
        load_location_registry(path)


def test_empty_registry_is_an_error(tmp_path):
    """An empty registry would send every row to review."""
    path = tmp_path / "zones.yaml"
    path.write_text("zones: []\n", encoding="utf-8")
    with pytest.raises(LocationRegistryError, match="declares no zones"):
        load_location_registry(path)


def test_registry_path_is_declared_in_settings():
    settings = load_settings()
    assert settings.locations_file.endswith(".yaml")
    assert ProblemType.TRAFFIC in settings.problems

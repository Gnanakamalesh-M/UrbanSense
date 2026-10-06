"""Unit conversion (SPEC section 11).

Two values are not comparable until they mean the same thing, and
``2050 vehicles/15min`` versus ``8200 vehicles/hour`` is the same road. So every
value is converted to its metric's canonical unit before anything compares,
aggregates or range-checks it.

**The original is always kept.** Normalization records the converted value *and*
the value as it arrived, because a conversion that cannot be re-checked is one
that will eventually be wrong without anyone noticing.

**An unknown unit is never coerced.** There is no "probably vehicles per hour"
fallback: a unit this module does not recognize produces a failure the caller
turns into a quarantine record. Guessing is exactly how silent corruption gets
in (SPEC section 14).

Where the knowledge lives is a deliberate split. Which units a metric may
*arrive* in is policy and lives in ``configs/problems/*.yaml``. How to convert
between two known units is physics, does not vary by deployment, and lives
here. A test asserts every unit the config accepts is convertible to that
metric's canonical unit, so the two cannot drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Conversion factors to each dimension's reference unit.
#:
#: A value in unit ``u`` becomes ``value * _TO_REFERENCE[u]`` in that
#: dimension's reference unit, so converting between any two units of the same
#: dimension is one multiply and one divide. Dimensions are kept separate so a
#: length can never be converted into a speed.
_TO_REFERENCE: dict[str, tuple[str, float]] = {
    # --- vehicle flow, reference vehicles/hour -------------------------
    "vehicles/hour": ("flow", 1.0),
    "vehicles/minute": ("flow", 60.0),
    "vehicles/5min": ("flow", 12.0),
    "vehicles/15min": ("flow", 4.0),
    "vehicles/30min": ("flow", 2.0),
    "vehicles/day": ("flow", 1.0 / 24.0),
    # --- speed, reference km/hour --------------------------------------
    "km/hour": ("speed", 1.0),
    "m/s": ("speed", 3.6),
    "mph": ("speed", 1.609344),
    # --- dimensionless ratio, reference ratio --------------------------
    "ratio": ("ratio", 1.0),
    "percent": ("ratio", 0.01),
    # --- length, reference metre ---------------------------------------
    "metre": ("length", 1.0),
    "kilometre": ("length", 1000.0),
    "foot": ("length", 0.3048),
    # --- rainfall depth, reference mm ----------------------------------
    # Rain belongs to the flood domain, which v1 does not enable, but the
    # covariate is ingested alongside traffic so its units must be known.
    "mm": ("depth", 1.0),
    "cm": ("depth", 10.0),
    "inch": ("depth", 25.4),
}

#: Alternative spellings a source might use, mapped to the canonical spelling.
#: Only unambiguous synonyms belong here -- this is spelling, not guessing.
_UNIT_ALIASES: dict[str, str] = {
    "vehicles/h": "vehicles/hour",
    "veh/hour": "vehicles/hour",
    "veh/h": "vehicles/hour",
    "vph": "vehicles/hour",
    "vehicles per hour": "vehicles/hour",
    "vehicles/min": "vehicles/minute",
    "km/h": "km/hour",
    "kmph": "km/hour",
    "kph": "km/hour",
    "m/sec": "m/s",
    "metres/second": "m/s",
    "miles/hour": "mph",
    "%": "percent",
    "fraction": "ratio",
    "m": "metre",
    "meter": "metre",
    "meters": "metre",
    "metres": "metre",
    "km": "kilometre",
    "kilometer": "kilometre",
    "ft": "foot",
    "feet": "foot",
    "millimetre": "mm",
    "millimeter": "mm",
    "centimetre": "cm",
    "centimeter": "cm",
    "in": "inch",
    "inches": "inch",
}


class UnitError(ValueError):
    """Raised when a unit is unknown or a conversion is not defined.

    Deliberately an exception rather than a silent passthrough: the caller must
    decide what to do (in this project, quarantine the record), and an
    unnoticed failure here would mean comparing incompatible numbers.
    """


def canonicalize_unit(unit: str) -> str:
    """Normalize a unit's spelling without converting anything.

    Trims whitespace, lowercases, and resolves known synonyms (``km/h`` ->
    ``km/hour``). Unknown spellings are returned as given, lowercased, so the
    caller sees what the source actually said in the error message.
    """
    cleaned = unit.strip().lower()
    return _UNIT_ALIASES.get(cleaned, cleaned)


def is_known_unit(unit: str) -> bool:
    """Whether this module can convert the given unit."""
    return canonicalize_unit(unit) in _TO_REFERENCE


def dimension_of(unit: str) -> str | None:
    """Return the physical dimension of a unit, or ``None`` if unknown."""
    found = _TO_REFERENCE.get(canonicalize_unit(unit))
    return found[0] if found is not None else None


def are_compatible(from_unit: str, to_unit: str) -> bool:
    """Whether two units measure the same dimension."""
    left, right = dimension_of(from_unit), dimension_of(to_unit)
    return left is not None and left == right


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` from one unit to another.

    Args:
        value: The measurement.
        from_unit: Unit the value is in.
        to_unit: Unit to convert to.

    Returns:
        The converted value. Returned unchanged when the units are the same.

    Raises:
        UnitError: if either unit is unknown, or if they measure different
            dimensions -- converting a queue length into a speed is a bug, not
            a conversion, so it fails loudly.
    """
    source = canonicalize_unit(from_unit)
    target = canonicalize_unit(to_unit)
    if source == target:
        return value

    source_entry = _TO_REFERENCE.get(source)
    target_entry = _TO_REFERENCE.get(target)
    if source_entry is None:
        raise UnitError(f"unknown unit {from_unit!r}; known units are {sorted(_TO_REFERENCE)}")
    if target_entry is None:
        raise UnitError(f"unknown unit {to_unit!r}; known units are {sorted(_TO_REFERENCE)}")

    source_dimension, source_factor = source_entry
    target_dimension, target_factor = target_entry
    if source_dimension != target_dimension:
        raise UnitError(
            f"cannot convert {from_unit!r} ({source_dimension}) to {to_unit!r} "
            f"({target_dimension}): different dimensions"
        )

    return value * source_factor / target_factor


@dataclass(frozen=True, slots=True)
class Normalized:
    """The result of normalizing one value's unit.

    Both representations are carried so the conversion stays auditable
    (SPEC section 11).

    Attributes:
        value: The value in the canonical unit.
        unit: The canonical unit.
        original_value: The value as it arrived, or ``None`` when the reading
            was missing. Missing stays missing through conversion.
        original_unit: The unit as it arrived.
        converted: Whether a conversion actually happened. ``False`` when the
            source already used the canonical unit, in which case there is no
            second representation worth recording.
    """

    value: float | None
    unit: str
    original_value: float | None
    original_unit: str
    converted: bool


def normalize_value(value: float | None, from_unit: str, canonical_unit: str) -> Normalized:
    """Convert a possibly-missing value into its canonical unit.

    A ``None`` value stays ``None``: it is converted in name only, because a
    sensor that reported nothing did not report zero of anything
    (SPEC section 15). The unit is still normalized, so a missing reading still
    records what it would have been measured in.

    Raises:
        UnitError: if the source unit is unknown or incompatible.
    """
    source = canonicalize_unit(from_unit)
    target = canonicalize_unit(canonical_unit)

    if value is None:
        # Validate the unit anyway -- an unknown unit on a missing reading is
        # still a feed we do not understand.
        if source != target and not are_compatible(source, target):
            raise UnitError(
                f"cannot convert {from_unit!r} to {canonical_unit!r}: unknown or incompatible units"
            )
        return Normalized(
            value=None,
            unit=target,
            original_value=None,
            original_unit=source,
            converted=source != target,
        )

    converted_value = convert(value, source, target)
    return Normalized(
        value=converted_value,
        unit=target,
        original_value=value,
        original_unit=source,
        converted=source != target,
    )


def known_units() -> tuple[str, ...]:
    """Every unit this module can convert, sorted."""
    return tuple(sorted(_TO_REFERENCE))

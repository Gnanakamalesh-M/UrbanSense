"""Range and plausibility checks (SPEC section 14).

Detects values that cannot be measurements: negative vehicle counts, negative
rainfall, a car at 480 km/h. The bounds are not hardcoded -- they come from each
metric's ``MetricConfig`` in ``configs/problems/*.yaml``, so adding flood or
waste means adding YAML.

**Nothing is deleted and nothing is clipped.** A negative count is not silenced
by `max(0, value)`; that would turn a sensor fault into a plausible-looking
reading and destroy the evidence that something is wrong. Impossible values go
to quarantine with their payload (SPEC section 18).

Note the distinction this module maintains against
:mod:`urbansense.data_integrity.outliers`:

* **impossible** -- violates physics or the metric's declared bounds. Not a
  measurement of anything. Quarantined.
* **unusual** -- surprising but physically possible. Could be a festival, an
  accident, a diversion, or a genuine record. Flagged for investigation and
  kept (SPEC section 16).

Checking against *canonical-unit* bounds is why normalization runs first: a
`2050 vehicles/15min` reading compared against a per-hour ceiling would look
fine while actually being four times the limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from urbansense.config.models import MetricConfig
from urbansense.schemas.enums import QuarantineReason


@dataclass(frozen=True, slots=True)
class RangeViolation:
    """One failed range check.

    Attributes:
        reason: Quarantine category for the failure.
        detail: Human-readable explanation naming the bound that was violated.
    """

    reason: QuarantineReason
    detail: str


def check_range(value: float | None, metric: MetricConfig, *, unit: str) -> RangeViolation | None:
    """Check one value against its metric's declared bounds.

    Args:
        value: The value in the metric's canonical unit, or ``None``.
        metric: The metric's configuration, carrying the bounds.
        unit: Unit the value is in, used only in the message.

    Returns:
        A :class:`RangeViolation` if the value is impossible, otherwise
        ``None``. A ``None`` value always returns ``None``: missing data is not
        an invalid value, and treating absence as a range failure would
        quarantine every gap in the feed (SPEC section 15).
    """
    if value is None:
        return None

    if metric.non_negative and value < 0:
        return RangeViolation(
            reason=QuarantineReason.INVALID_VALUE,
            detail=(
                f"{metric.name} is {value:g} {unit}, but the metric is declared "
                "non-negative; a negative count is not a measurement. Preserved "
                "rather than clipped to zero, which would hide the fault."
            ),
        )

    if metric.plausible_min is not None and value < metric.plausible_min:
        return RangeViolation(
            reason=QuarantineReason.INVALID_VALUE,
            detail=(
                f"{metric.name} is {value:g} {unit}, below the configured "
                f"plausible minimum of {metric.plausible_min:g}"
            ),
        )

    if metric.plausible_max is not None and value > metric.plausible_max:
        return RangeViolation(
            reason=QuarantineReason.INVALID_VALUE,
            detail=(
                f"{metric.name} is {value:g} {unit}, above the configured "
                f"plausible maximum of {metric.plausible_max:g}; physically "
                "impossible rather than merely unusual"
            ),
        )

    return None

"""Leakage detection (SPEC section 33).

    This is mandatory. The system must prevent future information from entering
    historical training data during evaluation.

A model that scores well because it saw the answer is worse than no model,
because it is convincing. So this module does not trust declarations -- it
*tests* behaviour.

**The probe.** For a sample at time ``t``, every observation after ``t`` is
perturbed (multiplied *and* shifted, so the change cannot cancel at zero), the
features for ``t`` are rebuilt against the perturbed data, and each feature is
compared with its original value. A feature that changed read the future. This
catches leakage no matter how the feature was written: a hand-rolled builder
ignoring the declared offsets is caught exactly as readily as a mis-declared
spec, because the evidence is behaviour rather than intent.

The target is excluded by construction. Mutating the future obviously changes
``y`` -- that is the label, not a feature, and conflating the two would make every
honest setup look leaky.

**Two clauses, honestly reported.**

``event_time <= t`` is always probed; it is the constraint section 33 describes.

``received_time <= t`` matters only when the receipts actually say something
about availability. After a batch load every record shares the instant the file
was read, so enforcing the clause would reject every row and make backtesting
impossible -- the receipt instant is then an artifact of when we happened to
open the file, not of when the information was available. The clause is reported
as *not applicable* rather than silently skipped, so a reader can see exactly
what was verified.

**Applicability is decided per sample time, not by counting receipt instants.**
An earlier version asked only whether more than one distinct receipt existed,
and that was wrong in a way that produced false accusations rather than missed
ones. Concatenating two CSVs gives two load instants seconds apart, both months
after the last event: more than one, so the clause was probed -- and then
*every* record counted as a late arrival, every feature moved, and the probe
reported 237 violations against feature code that was entirely correct. The same
challenger came out clean on one file and leaky on two.

So the question is whether the receipts **straddle** the sample time: some
records already arrived, some had not. Only then does "was this record available
yet?" have two possible answers, and only then can perturbing the late arrivals
tell a leaky feature from a clean one. When nothing had arrived, the clause is
unprobeable at that instant and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from urbansense.data_integrity.unified import ReconciledResult
from urbansense.features.builder import compute_features
from urbansense.features.series import SeriesIndex
from urbansense.features.specs import FeatureSet

#: How many sample times to probe by default. Each probe rebuilds every feature
#: against a perturbed copy of the data, so running it on every row would be far
#: too slow; a spread across the series catches a systematically leaky feature
#: immediately, and leakage is systematic by nature.
DEFAULT_PROBE_SAMPLES = 24


class ClauseStatus(StrEnum):
    """Whether a leakage clause was actually checked."""

    #: Probed, and no violation found.
    ENFORCED = "enforced"
    #: Probed, and a violation found.
    VIOLATED = "violated"
    #: Could not be probed meaningfully on this data. Recorded, never hidden.
    NOT_APPLICABLE = "not_applicable"


class LeakageKind(StrEnum):
    """Which clause a finding violates."""

    #: The feature read data whose event is after the sample time.
    FUTURE_EVENT = "future_event"
    #: The feature read data that had not yet arrived at the sample time.
    LATE_ARRIVAL = "late_arrival"


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    """One feature caught reading data it should not have.

    Attributes:
        feature_name: The offending feature.
        kind: Which clause it violated.
        sample_time: The sample time at which it was caught.
        original_value: Value built against the real data.
        perturbed_value: Value built after the unavailable data was perturbed.
            That these differ *is* the proof.
        location_id: Zone the sample belonged to.
    """

    feature_name: str
    kind: LeakageKind
    sample_time: datetime
    original_value: float
    perturbed_value: float
    location_id: str

    def describe(self) -> str:
        """Human-readable explanation."""
        what = (
            "data after that instant"
            if self.kind is LeakageKind.FUTURE_EVENT
            else "data that had not yet arrived"
        )
        return (
            f"{self.feature_name} changed from {self.original_value:g} to "
            f"{self.perturbed_value:g} at {self.location_id} "
            f"{self.sample_time.isoformat()} when {what} was perturbed"
        )


@dataclass(frozen=True)
class LeakageReport:
    """The outcome of a leakage check.

    Attributes:
        event_time_clause: Whether ``event_time <= t`` held.
        received_time_clause: Whether ``received_time <= t`` was checkable, and
            if so whether it held.
        received_time_note: Why the receipt clause was or was not applicable.
        findings: Features caught reading unavailable data.
        probed_samples: How many sample times were probed.
        probed_features: How many features were probed per sample.
        distinct_received_times: Distinct receipt instants in the data, which is
            what decides whether the receipt clause means anything.
    """

    event_time_clause: ClauseStatus
    received_time_clause: ClauseStatus
    received_time_note: str
    findings: tuple[LeakageFinding, ...] = ()
    probed_samples: int = 0
    probed_features: int = 0
    distinct_received_times: int = 0

    @property
    def is_clean(self) -> bool:
        """Whether every applicable clause was satisfied."""
        return not self.findings and ClauseStatus.VIOLATED not in (
            self.event_time_clause,
            self.received_time_clause,
        )

    @property
    def leaky_features(self) -> tuple[str, ...]:
        """Names of the features that leaked, sorted and deduplicated."""
        return tuple(sorted({finding.feature_name for finding in self.findings}))

    def findings_of(self, kind: LeakageKind) -> tuple[LeakageFinding, ...]:
        """Findings for one clause."""
        return tuple(finding for finding in self.findings if finding.kind is kind)

    def as_dict(self) -> dict[str, object]:
        """Serializable form, for the model registry."""
        return {
            "clean": self.is_clean,
            "event_time_clause": self.event_time_clause.value,
            "received_time_clause": self.received_time_clause.value,
            "received_time_note": self.received_time_note,
            "probed_samples": self.probed_samples,
            "probed_features": self.probed_features,
            "distinct_received_times": self.distinct_received_times,
            "leaky_features": list(self.leaky_features),
            "findings": [
                {
                    "feature": finding.feature_name,
                    "kind": finding.kind.value,
                    "sample_time": finding.sample_time.isoformat(),
                    "location_id": finding.location_id,
                    "original": finding.original_value,
                    "perturbed": finding.perturbed_value,
                }
                for finding in self.findings
            ],
        }

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"leakage check: {'CLEAN' if self.is_clean else 'LEAKAGE DETECTED'}",
            f"  event_time <= t      {self.event_time_clause.value}"
            f"  ({self.probed_features} features x {self.probed_samples} sample times)",
            f"  received_time <= t   {self.received_time_clause.value}"
            f"  ({self.received_time_note})",
        ]
        lines.extend(f"  LEAK: {finding.describe()}" for finding in self.findings)
        return "\n".join(lines)


class LeakageError(RuntimeError):
    """Raised when training would proceed on leaky features."""


def values_match(left: float, right: float) -> bool:
    """Whether two feature values are the same.

    Two ``NaN``s count as matching: a feature unknown before and unknown after
    learned nothing from the perturbation.
    """
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isnan(left) or math.isnan(right):
        return False
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def detect_leakage(
    result: ReconciledResult,
    feature_set: FeatureSet,
    *,
    timezone_offset_hours: float = 0.0,
    probe_samples: int = DEFAULT_PROBE_SAMPLES,
    perturbation_factor: float = 10.0,
    perturbation_shift: float = 1000.0,
) -> LeakageReport:
    """Probe every feature for reads of unavailable data.

    Args:
        result: Reconciliation output; only ``unified`` is read.
        feature_set: Features to probe.
        timezone_offset_hours: Local offset for calendar features.
        probe_samples: How many sample times to probe, spread evenly across the
            series so a leaky feature is caught wherever it sits.
        perturbation_factor: Multiplier applied to unavailable values.
        perturbation_shift: Offset added as well, so the perturbation is not a
            no-op for a value of zero.

    Returns:
        A :class:`LeakageReport`. Empty findings mean every probed feature was
        unmoved by data it should not have been able to see.
    """
    observations = list(result.unified)
    index = SeriesIndex.from_observations(observations)

    target_rows = [
        observation
        for observation in observations
        if observation.metric == feature_set.target_metric
    ]
    zone_codes = {
        name: float(position)
        for position, name in enumerate(sorted({row.location_id for row in target_rows}))
    }

    candidates = sorted(
        {(row.event_time, row.location_id) for row in target_rows},
        key=lambda pair: (pair[0], pair[1]),
    )
    # Skip the warm-up period: a sample whose lags all fall before the data
    # begins is NaN everywhere and would prove nothing either way.
    warmup = feature_set.max_reach_hours
    usable = candidates[warmup:] if len(candidates) > warmup else candidates
    if not usable:
        return LeakageReport(
            event_time_clause=ClauseStatus.NOT_APPLICABLE,
            received_time_clause=ClauseStatus.NOT_APPLICABLE,
            received_time_note="no usable sample time: the series is shorter than its warm-up",
            distinct_received_times=index.distinct_received_times,
        )

    step = max(1, len(usable) // max(probe_samples, 1))
    probes = usable[::step][:probe_samples]
    horizon = timedelta(hours=feature_set.horizon_hours)

    findings: list[LeakageFinding] = []
    receipt_probes = 0
    for target_time, location_id in probes:
        sample_time = target_time - horizon
        # Decided per sample time: the clause only means something where some
        # records had arrived and some had not.
        receipts_are_meaningful = index.receipts_straddle(sample_time)
        receipt_probes += int(receipts_are_meaningful)

        variants = [
            (
                LeakageKind.FUTURE_EVENT,
                index.with_future_perturbed(
                    sample_time, factor=perturbation_factor, shift=perturbation_shift
                ),
            )
        ]
        if receipts_are_meaningful:
            variants.append(
                (
                    LeakageKind.LATE_ARRIVAL,
                    index.with_late_arrivals_perturbed(
                        sample_time, factor=perturbation_factor, shift=perturbation_shift
                    ),
                )
            )

        for spec in feature_set.specs:
            original = compute_features(
                spec,
                index=index,
                location_id=location_id,
                sample_time=sample_time,
                target_metric=feature_set.target_metric,
                timezone_offset_hours=timezone_offset_hours,
                zone_codes=zone_codes,
            )
            for kind, perturbed_index in variants:
                probed = compute_features(
                    spec,
                    index=perturbed_index,
                    location_id=location_id,
                    sample_time=sample_time,
                    target_metric=feature_set.target_metric,
                    timezone_offset_hours=timezone_offset_hours,
                    zone_codes=zone_codes,
                )
                if not values_match(original, probed):
                    findings.append(
                        LeakageFinding(
                            feature_name=spec.name,
                            kind=kind,
                            sample_time=sample_time,
                            original_value=original,
                            perturbed_value=probed,
                            location_id=location_id,
                        )
                    )

    future_findings = [f for f in findings if f.kind is LeakageKind.FUTURE_EVENT]
    late_findings = [f for f in findings if f.kind is LeakageKind.LATE_ARRIVAL]

    if receipt_probes:
        receipt_note = (
            f"probed at {receipt_probes} of {len(probes)} sample time(s), where "
            f"receipts straddled the instant; {index.distinct_received_times} "
            "distinct receipt instant(s) in the data"
        )
    else:
        receipt_note = (
            f"not probed: every record arrived after every sample time "
            f"({index.distinct_received_times} distinct receipt instant(s), all "
            "after the events), so receipt order carries no information about "
            "what was available and perturbing late arrivals would perturb the "
            "whole dataset"
        )

    return LeakageReport(
        event_time_clause=(ClauseStatus.VIOLATED if future_findings else ClauseStatus.ENFORCED),
        received_time_clause=(
            ClauseStatus.NOT_APPLICABLE
            if not receipt_probes
            else (ClauseStatus.VIOLATED if late_findings else ClauseStatus.ENFORCED)
        ),
        received_time_note=receipt_note,
        findings=tuple(findings),
        probed_samples=len(probes),
        probed_features=len(feature_set.specs),
        distinct_received_times=index.distinct_received_times,
    )


def assert_no_leakage(report: LeakageReport) -> None:
    """Raise if a leakage check found anything.

    Deliberately loud and fatal. Training on leaky features produces a model that
    looks good and is useless, and a warning on stderr is exactly the kind of
    thing that gets scrolled past.

    Raises:
        LeakageError: when the report is not clean.
    """
    if report.is_clean:
        return
    raise LeakageError(
        f"refusing to train: {len(report.findings)} leakage finding(s) across "
        f"features {list(report.leaky_features)}.\n{report.describe()}"
    )

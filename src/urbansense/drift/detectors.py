"""Drift statistics and the three detector families (SPEC section 29).

    Monitor: feature drift, prediction drift, performance drift.

Three things about this module are deliberate and worth stating before the code.

**Bin edges come from the reference, never from the window being judged.** A
quantile binning recomputed on each recent window would adapt to the very shift
it is supposed to notice, and PSI would stay near zero through any amount of
drift.

**The KS gate is the effect size, not the p-value.** On the 650-row weekly
windows this project replays, a KS test calls almost any difference significant;
the D statistic is what says whether the difference matters. Both are reported,
and the gate is D.

**The performance reference must be out-of-sample.** Measured on the real data,
the champion's in-sample MAE is 104 while its honest out-of-sample level is
~174. Using the in-sample figure as the reference puts every window at a 1.67x
ratio from the first week and every detector screams immediately. The reference
error therefore comes from a held-back window, and :func:`performance_drift`
takes it as an argument rather than computing it from training rows.

A detector with too thin a reference reports ``INSUFFICIENT_REFERENCE`` rather
than firing or staying quiet. A three-sigma gate estimated from two numbers is
not a gate, and silence would read as "no drift".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy import stats

#: Bins for the population stability index. Ten is the convention, and it keeps
#: roughly 65 observations per bin on a weekly window of this dataset.
DEFAULT_PSI_BINS = 10

#: Published PSI convention: below 0.1 is no material change, 0.1 to 0.25 is a
#: minor shift worth watching, above 0.25 is a major shift. These are taken as
#: given rather than fitted -- on this dataset the stable stretch reaches 0.117,
#: so a 0.1 gate would false-alarm and 0.25 is what may create a candidate.
PSI_MINOR = 0.1
PSI_MAJOR = 0.25

#: KS D-statistic at or above which a distribution change is called material.
DEFAULT_KS_EFFECT = 0.15

#: How many standard deviations above the reference mean a performance ratio
#: must sit to fire. Three keeps the stable stretch quiet: its ratios average
#: 1.668 with a standard deviation of 0.053, so the gate lands at 1.83 against a
#: stable maximum of 1.76 and a first-drifted-week value of 2.56.
DEFAULT_SIGMA_MULTIPLIER = 3.0

#: Reference windows needed before a sigma-based gate means anything.
DEFAULT_MIN_REFERENCE_WINDOWS = 4

#: Smallest sample a statistic will be computed on. Below this the number is
#: noise wearing a decimal point.
MIN_SAMPLE = 30


class DriftKind(StrEnum):
    """Which of SPEC section 29's three families a finding belongs to."""

    #: An input distribution moved.
    FEATURE = "feature"
    #: The model's output distribution moved.
    PREDICTION = "prediction"
    #: The model's error got worse.
    PERFORMANCE = "performance"


class DriftStatus(StrEnum):
    """What a detector concluded.

    ``INSUFFICIENT_REFERENCE`` and ``INSUFFICIENT_DATA`` are outcomes, not
    errors. Reporting them as "no drift" would turn an absence of evidence into
    evidence of absence, which is the failure mode this whole phase exists to
    avoid.
    """

    #: Compared, and nothing material moved.
    STABLE = "stable"
    #: Compared, and the gate was crossed.
    DRIFTED = "drifted"
    #: Moved, but not past the gate. Reported so a trend is visible.
    MINOR = "minor"
    #: Too few reference windows to establish a gate.
    INSUFFICIENT_REFERENCE = "insufficient_reference"
    #: Too few observations in the window being judged.
    INSUFFICIENT_DATA = "insufficient_data"


def _clean(values: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Drop unknown values rather than filling them.

    Missing stays missing: an absent reading contributes nothing to a
    distribution, and substituting a mean would make a sparse window look like
    the reference.
    """
    array = np.asarray(values, dtype=np.float64)
    return array[~np.isnan(array)]


def psi(
    reference: Sequence[float] | NDArray[np.float64],
    recent: Sequence[float] | NDArray[np.float64],
    *,
    bins: int = DEFAULT_PSI_BINS,
) -> float:
    """Population stability index between a reference and a recent sample.

    Bin edges are the reference's quantiles, extended to infinity at both ends
    so a recent value outside the reference range lands in the end bin instead
    of being dropped. Zero-count bins are floored at a small epsilon, because
    the index is a sum of logs and an empty bin would otherwise make it
    infinite.

    Returns:
        The index, or ``nan`` when either sample is too small to bin.
    """
    left, right = _clean(reference), _clean(recent)
    if left.size < MIN_SAMPLE or right.size < MIN_SAMPLE:
        return float("nan")

    edges = np.quantile(left, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # A reference with too few distinct quantiles cannot be binned. A constant
    # collapses every interior edge onto one value; a binary feature onto two,
    # leaving half the bins empty so the sum of logs reflects the epsilon floor
    # rather than any change in the data. Requiring nearly all edges to be
    # distinct rejects both, and rejecting is right: PSI is not the tool for a
    # discrete feature. One tie is tolerated, since an integer-valued series can
    # legitimately repeat a quantile.
    if len(np.unique(edges)) < bins:
        return float("nan")

    reference_share = np.histogram(left, edges)[0] / left.size
    recent_share = np.histogram(right, edges)[0] / right.size
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    recent_share = np.clip(recent_share, epsilon, None)
    return float(np.sum((recent_share - reference_share) * np.log(recent_share / reference_share)))


def psi_severity(value: float) -> DriftStatus:
    """Map a PSI onto the published convention."""
    if np.isnan(value):
        return DriftStatus.INSUFFICIENT_DATA
    if value >= PSI_MAJOR:
        return DriftStatus.DRIFTED
    return DriftStatus.MINOR if value >= PSI_MINOR else DriftStatus.STABLE


@dataclass(frozen=True)
class KSResult:
    """A two-sample Kolmogorov-Smirnov comparison.

    Attributes:
        statistic: The D statistic -- the largest gap between the two empirical
            distribution functions. This is the effect size, and it is what the
            gate uses.
        p_value: Significance. Reported for completeness and deliberately not
            used as the gate: on a 650-row window almost any difference is
            "significant", so a p-value gate would fire on noise.
        tested: Whether there was enough data to compare at all.
    """

    statistic: float
    p_value: float
    tested: bool

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 8),
            "tested": self.tested,
            "note": "the D statistic is the gate; p is reported but not used as one",
        }


def ks_test(
    reference: Sequence[float] | NDArray[np.float64],
    recent: Sequence[float] | NDArray[np.float64],
) -> KSResult:
    """Two-sample KS test, reporting the effect size first."""
    left, right = _clean(reference), _clean(recent)
    if left.size < MIN_SAMPLE or right.size < MIN_SAMPLE:
        return KSResult(statistic=float("nan"), p_value=float("nan"), tested=False)
    outcome = stats.ks_2samp(left, right)
    return KSResult(statistic=float(outcome.statistic), p_value=float(outcome.pvalue), tested=True)


@dataclass(frozen=True)
class DriftSignal:
    """One detector's verdict on one window.

    Attributes:
        kind: Which family this belongs to.
        subject: What was compared -- a feature name, the prediction series or
            the error metric.
        status: What the detector concluded.
        statistic: The headline number.
        threshold: The gate it was compared against.
        reference_rows: Observations in the reference sample.
        recent_rows: Observations in the window judged.
        detail: Supporting numbers, kept so a finding can be re-read later.
    """

    kind: DriftKind
    subject: str
    status: DriftStatus
    statistic: float
    threshold: float
    reference_rows: int
    recent_rows: int
    detail: dict[str, object] | None = None

    @property
    def fired(self) -> bool:
        """Whether this signal crosses the gate."""
        return self.status is DriftStatus.DRIFTED

    @property
    def conclusive(self) -> bool:
        """Whether the detector could reach a verdict at all."""
        return self.status not in (
            DriftStatus.INSUFFICIENT_REFERENCE,
            DriftStatus.INSUFFICIENT_DATA,
        )

    @property
    def magnitude(self) -> float:
        """How far past the gate the statistic sits, as a ratio.

        ``nan`` when the gate is zero or the statistic unknown, rather than a
        fabricated multiple.
        """
        if np.isnan(self.statistic) or self.threshold == 0 or np.isnan(self.threshold):
            return float("nan")
        return float(self.statistic / self.threshold)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "kind": self.kind.value,
            "subject": self.subject,
            "status": self.status.value,
            "statistic": None if np.isnan(self.statistic) else round(self.statistic, 6),
            "threshold": None if np.isnan(self.threshold) else round(self.threshold, 6),
            "magnitude_vs_threshold": (
                None if np.isnan(self.magnitude) else round(self.magnitude, 4)
            ),
            "reference_rows": self.reference_rows,
            "recent_rows": self.recent_rows,
            "detail": self.detail or {},
        }

    def describe(self) -> str:
        """One line."""
        if not self.conclusive:
            return f"{self.kind.value}/{self.subject}: {self.status.value}"
        return (
            f"{self.kind.value}/{self.subject}: {self.status.value} "
            f"(statistic {self.statistic:.4f} against gate {self.threshold:.4f}, "
            f"n={self.recent_rows} vs reference {self.reference_rows})"
        )


def feature_drift(
    subject: str,
    reference: Sequence[float] | NDArray[np.float64],
    recent: Sequence[float] | NDArray[np.float64],
    *,
    psi_gate: float = PSI_MAJOR,
    ks_effect_gate: float = DEFAULT_KS_EFFECT,
    bins: int = DEFAULT_PSI_BINS,
) -> DriftSignal:
    """Compare one input's reference distribution against a recent window.

    Both PSI and KS are computed. PSI is the gate, because its published
    thresholds let the result be stated without reference to this dataset; the
    KS effect size travels alongside as a second opinion, and a case where the
    two disagree is exactly the case worth a human looking at.
    """
    left, right = _clean(reference), _clean(recent)
    index = psi(left, right, bins=bins)
    ks = ks_test(left, right)

    if np.isnan(index):
        status = DriftStatus.INSUFFICIENT_DATA
    elif index >= psi_gate:
        status = DriftStatus.DRIFTED
    elif index >= PSI_MINOR:
        status = DriftStatus.MINOR
    else:
        status = DriftStatus.STABLE

    return DriftSignal(
        kind=DriftKind.FEATURE,
        subject=subject,
        status=status,
        statistic=index,
        threshold=psi_gate,
        reference_rows=int(left.size),
        recent_rows=int(right.size),
        detail={
            "psi": None if np.isnan(index) else round(index, 6),
            "psi_minor_gate": PSI_MINOR,
            "ks": ks.as_dict(),
            "ks_effect_gate": ks_effect_gate,
            "ks_agrees": bool(
                ks.tested and (ks.statistic >= ks_effect_gate) == (index >= psi_gate)
            ),
            "reference_median": float(np.median(left)) if left.size else None,
            "recent_median": float(np.median(right)) if right.size else None,
        },
    )


def prediction_drift(
    reference: Sequence[float] | NDArray[np.float64],
    recent: Sequence[float] | NDArray[np.float64],
    *,
    gate: float,
    bins: int = DEFAULT_PSI_BINS,
    subject: str = "prediction",
) -> DriftSignal:
    """Compare the model's own output distribution across two windows.

    Worth a separate family from feature drift because the two can disagree: a
    small move in several inputs can combine into a large move in the output,
    and a model can also absorb an input shift without changing what it says.

    The gate is passed in rather than taken from the PSI convention, because
    prediction PSI runs an order of magnitude smaller than feature PSI on this
    data -- 0.004 to 0.018 on the stable stretch against 0.03 to 0.14 after the
    shift. :func:`sigma_gate` derives it from the reference period.
    """
    left, right = _clean(reference), _clean(recent)
    index = psi(left, right, bins=bins)
    ks = ks_test(left, right)

    if np.isnan(index):
        status = DriftStatus.INSUFFICIENT_DATA
    elif index >= gate:
        status = DriftStatus.DRIFTED
    elif index >= gate * 0.5:
        status = DriftStatus.MINOR
    else:
        status = DriftStatus.STABLE

    return DriftSignal(
        kind=DriftKind.PREDICTION,
        subject=subject,
        status=status,
        statistic=index,
        threshold=gate,
        reference_rows=int(left.size),
        recent_rows=int(right.size),
        detail={
            "psi": None if np.isnan(index) else round(index, 6),
            "ks": ks.as_dict(),
            "reference_mean": float(np.mean(left)) if left.size else None,
            "recent_mean": float(np.mean(right)) if right.size else None,
            "gate_source": "reference-period variability, not the PSI convention",
        },
    )


def performance_drift(
    recent_error: float,
    *,
    reference_error: float,
    gate_ratio: float,
    recent_rows: int,
    reference_rows: int,
    subject: str = "mae",
) -> DriftSignal:
    """Compare a recent error against the reference error level.

    Args:
        recent_error: Error over the window being judged.
        reference_error: The **out-of-sample** reference error. Passing an
            in-sample figure here is the single easiest way to make this
            detector useless: on the real data it would start every window at a
            1.67x ratio and fire immediately and permanently.
        gate_ratio: Ratio at or above which drift is called. See
            :func:`sigma_gate`.
        recent_rows: Observations in the recent window.
        reference_rows: Observations behind the reference figure.
        subject: Error metric being compared.
    """
    if reference_error <= 0 or np.isnan(reference_error) or np.isnan(recent_error):
        return DriftSignal(
            kind=DriftKind.PERFORMANCE,
            subject=subject,
            status=DriftStatus.INSUFFICIENT_REFERENCE,
            statistic=float("nan"),
            threshold=gate_ratio,
            reference_rows=reference_rows,
            recent_rows=recent_rows,
            detail={"reason": "no usable reference error"},
        )

    ratio = recent_error / reference_error
    if recent_rows < MIN_SAMPLE:
        status = DriftStatus.INSUFFICIENT_DATA
    elif ratio >= gate_ratio:
        status = DriftStatus.DRIFTED
    elif ratio >= 1.0 + (gate_ratio - 1.0) * 0.5:
        status = DriftStatus.MINOR
    else:
        status = DriftStatus.STABLE

    return DriftSignal(
        kind=DriftKind.PERFORMANCE,
        subject=subject,
        status=status,
        statistic=ratio,
        threshold=gate_ratio,
        reference_rows=reference_rows,
        recent_rows=recent_rows,
        detail={
            "recent_error": round(recent_error, 4),
            "reference_error": round(reference_error, 4),
            "reference_is_out_of_sample": True,
        },
    )


def sigma_gate(
    reference_values: Sequence[float],
    *,
    multiplier: float = DEFAULT_SIGMA_MULTIPLIER,
    minimum_windows: int = DEFAULT_MIN_REFERENCE_WINDOWS,
    floor: float = 0.0,
) -> float:
    """A gate at ``mean + multiplier * sd`` of the reference period's own values.

    Calibrated on the reference period only -- data that was available before
    any decision was taken. Choosing a gate by looking at the drifted stretch
    would make every subsequent detection circular.

    Returns:
        The gate, or ``nan`` when there are too few reference windows to
        estimate a spread. Callers turn that into
        ``INSUFFICIENT_REFERENCE`` rather than guessing a number.
    """
    values = _clean(reference_values)
    if values.size < minimum_windows:
        return float("nan")
    gate = float(np.mean(values) + multiplier * np.std(values, ddof=1))
    return max(gate, floor)

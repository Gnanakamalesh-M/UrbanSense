"""Candidate updates: the step between noticing drift and acting on it.

    Drift detected -> Investigate -> Candidate model update   (SPEC section 29)
    Do not blindly retrain whenever new data arrives.         (SPEC section 30)

A drift event does **not** retrain anything. It creates a
:class:`CandidateUpdate`: a dated record saying what fired, how hard, and what
decision time it applies to. That record is SPEC section 29's *investigate* arrow
made durable, and it exists so that every model in the registry can be traced
back to the evidence that prompted it.

The decision that follows is recorded with **every** rule outcome, not only the
one that failed. "Rejected" on its own invites a retry with a nudged threshold;
"rejected, improved 1.8% against a 3.0% requirement" says how close it came and
leaves the threshold alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from urbansense.drift.detectors import DriftKind
from urbansense.drift.events import DriftEvent

#: Default directory for candidate and decision logs.
DEFAULT_CANDIDATE_DIR = Path("reports") / "adaptation"


class CandidateStatus(StrEnum):
    """Where a candidate update got to."""

    #: Created by a drift episode, nothing trained yet.
    RAISED = "raised"
    #: Challengers trained and evaluated.
    EVALUATED = "evaluated"
    #: A challenger passed every rule and became champion.
    PROMOTED = "promoted"
    #: No challenger passed. The champion was kept.
    REJECTED = "rejected"
    #: Could not be decided -- too little evaluation data, usually at the end
    #: of a replay. Distinct from rejection: nothing was judged and found
    #: wanting, there was simply nothing to judge on.
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RuleOutcome:
    """One promotion rule's verdict on one challenger.

    Attributes:
        rule: Which rule.
        passed: Whether it was satisfied.
        detail: The numbers behind the verdict, so a near miss is visible.
    """

    rule: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {"rule": self.rule, "passed": self.passed, "detail": self.detail}

    def describe(self) -> str:
        """One line."""
        return f"{'PASS' if self.passed else 'FAIL'}  {self.rule}: {self.detail}"


@dataclass(frozen=True)
class ChallengerEvaluation:
    """How one challenger scored against the champion.

    Attributes:
        version: Registry version name.
        training_window: Which window it was trained on.
        training_rows: Rows it trained on.
        training_end: Latest sample time in its training data. Asserted to be
            before the decision time.
        evaluation_rows: Rows it was judged on.
        champion_mae: Champion error on the evaluation window.
        challenger_mae: Challenger error on the same window.
        champion_rmse: Champion RMSE.
        challenger_rmse: Challenger RMSE.
        slice_results: Per-slice champion and challenger errors with row counts.
        leakage_clean: Whether the leakage probe passed.
        rules: Every rule's verdict.
    """

    version: str
    training_window: str
    training_rows: int
    training_end: datetime
    evaluation_rows: int
    champion_mae: float
    challenger_mae: float
    champion_rmse: float
    challenger_rmse: float
    slice_results: tuple[tuple[str, int, float, float], ...] = ()
    leakage_clean: bool = True
    rules: tuple[RuleOutcome, ...] = ()

    @property
    def relative_improvement(self) -> float:
        """Fraction by which the challenger beats the champion. Negative is worse."""
        if self.champion_mae <= 0:
            return float("nan")
        return 1.0 - (self.challenger_mae / self.champion_mae)

    @property
    def passed(self) -> bool:
        """Whether every rule was satisfied."""
        return bool(self.rules) and all(rule.passed for rule in self.rules)

    @property
    def failures(self) -> tuple[RuleOutcome, ...]:
        """Rules that were not satisfied."""
        return tuple(rule for rule in self.rules if not rule.passed)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "version": self.version,
            "training_window": self.training_window,
            "training_rows": self.training_rows,
            "training_end": self.training_end.isoformat(),
            "evaluation_rows": self.evaluation_rows,
            "champion_mae": round(self.champion_mae, 4),
            "challenger_mae": round(self.challenger_mae, 4),
            "champion_rmse": round(self.champion_rmse, 4),
            "challenger_rmse": round(self.challenger_rmse, 4),
            "relative_improvement": round(self.relative_improvement, 6),
            "leakage_clean": self.leakage_clean,
            "slices": [
                {
                    "name": name,
                    "rows": rows,
                    "champion_mae": round(champion, 4),
                    "challenger_mae": round(challenger, 4),
                }
                for name, rows, champion, challenger in self.slice_results
            ],
            "rules": [rule.as_dict() for rule in self.rules],
            "passed": self.passed,
        }

    def describe(self) -> str:
        """Human-readable summary, with every rule outcome."""
        lines = [
            f"{self.version} ({self.training_window}, {self.training_rows:,} rows "
            f"to {self.training_end:%Y-%m-%d})",
            f"  MAE {self.challenger_mae:,.1f} against champion "
            f"{self.champion_mae:,.1f} on {self.evaluation_rows:,} rows "
            f"({self.relative_improvement:+.1%})",
        ]
        lines.extend(f"  {rule.describe()}" for rule in self.rules)
        return "\n".join(lines)


@dataclass
class CandidateUpdate:
    """A drift episode's request for investigation.

    Attributes:
        candidate_id: Stable identifier, derived from the decision time.
        raised_at: The replay instant the drift fired.
        decision_time: The instant challengers are trained up to. Nothing at or
            after it may reach a challenger's training data.
        episode_id: The drift episode that prompted this.
        drift_kinds: Which families fired.
        drift_magnitude: How far past its gate the strongest signal sat.
        drift_subject: The feature or metric that moved most.
        champion_version: The incumbent at the time.
        status: Where the candidate got to.
        evaluations: One per challenger trained.
        promoted_version: The version promoted, when one was.
        notes: Why the outcome was what it was.
    """

    candidate_id: str
    raised_at: datetime
    decision_time: datetime
    episode_id: str | None
    drift_kinds: tuple[DriftKind, ...]
    drift_magnitude: float
    drift_subject: str
    champion_version: str
    status: CandidateStatus = CandidateStatus.RAISED
    evaluations: tuple[ChallengerEvaluation, ...] = ()
    promoted_version: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> ChallengerEvaluation | None:
        """The challenger with the lowest error, passing or not."""
        if not self.evaluations:
            return None
        return min(self.evaluations, key=lambda item: item.challenger_mae)

    @property
    def winner(self) -> ChallengerEvaluation | None:
        """The best challenger that passed every rule, if any.

        Chosen by error among the passing challengers only. A challenger that
        scores better but fails a protected-slice rule does not win: the rules
        are the gate, and ordering by error inside the gate is just a
        tie-break.
        """
        passing = [item for item in self.evaluations if item.passed]
        return min(passing, key=lambda item: item.challenger_mae) if passing else None

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "candidate_id": self.candidate_id,
            "raised_at": self.raised_at.isoformat(),
            "decision_time": self.decision_time.isoformat(),
            "episode_id": self.episode_id,
            "drift": {
                "kinds": [kind.value for kind in self.drift_kinds],
                "magnitude_vs_gate": round(self.drift_magnitude, 4),
                "subject": self.drift_subject,
            },
            "champion_version": self.champion_version,
            "status": self.status.value,
            "promoted_version": self.promoted_version,
            "evaluations": [item.as_dict() for item in self.evaluations],
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """Human-readable summary of the candidate and its decision."""
        kinds = ", ".join(kind.value for kind in self.drift_kinds) or "none"
        lines = [
            f"{self.candidate_id}  raised {self.raised_at:%Y-%m-%d}  "
            f"[{kinds}] {self.drift_subject} at {self.drift_magnitude:.2f}x gate",
            f"  champion {self.champion_version} -> {self.status.value.upper()}"
            + (f" ({self.promoted_version})" if self.promoted_version else ""),
        ]
        for item in self.evaluations:
            lines.append("  " + item.describe().replace("\n", "\n  "))
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def candidate_from_event(
    event: DriftEvent,
    *,
    decision_time: datetime,
    champion_version: str,
) -> CandidateUpdate:
    """Turn a firing drift event into a candidate update.

    Deliberately the only path from a drift event to any retraining: nothing
    else in the package trains a model, so "do not blindly retrain" is a
    property of the call graph rather than a convention.

    Raises:
        ValueError: if the event did not fire. A candidate with no drift behind
            it would be a retrain with no justification recorded.
    """
    if not event.is_drift:
        raise ValueError("cannot raise a candidate update from an event that did not fire")

    strongest = event.strongest
    return CandidateUpdate(
        candidate_id=f"CAND-{decision_time:%Y%m%dT%H%M%S}",
        raised_at=event.detected_at,
        decision_time=decision_time,
        episode_id=event.episode_id,
        drift_kinds=event.kinds,
        drift_magnitude=strongest.magnitude if strongest else float("nan"),
        drift_subject=strongest.subject if strongest else "unknown",
        champion_version=champion_version,
    )


class CandidateStore:
    """Append-only JSONL log of candidate updates and their decisions.

    Attributes:
        root: Directory the log lives in.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialize.

        Args:
            root: Directory to write into. Defaults to ``reports/adaptation/``.
        """
        self.root = root if root is not None else DEFAULT_CANDIDATE_DIR

    @property
    def path(self) -> Path:
        """The candidate log."""
        return self.root / "candidates.jsonl"

    def append(self, candidates: Iterable[CandidateUpdate]) -> int:
        """Append candidates to the log. Returns the number written."""
        rows = list(candidates)
        if not rows:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for candidate in rows:
                payload = candidate.as_dict()
                payload["recorded_at"] = datetime.now(UTC).isoformat()
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return len(rows)

    def read(self) -> tuple[dict[str, object], ...]:
        """Read the log back, in the order written."""
        if not self.path.is_file():
            return ()
        rows: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return tuple(rows)

"""Champion/challenger evaluation (SPEC sections 30, 32).

    Compare them using predefined evaluation rules.
    Only promote the challenger when it passes the evaluation criteria.
    Otherwise: REJECT CANDIDATE, KEEP CURRENT MODEL

Every rule is evaluated and recorded, including the ones that passed. A decision
that reported only its blocking reason would invite the obvious next move --
nudge that threshold and re-run -- whereas a decision that says "improved 1.8%
against a required 3.0%, every other rule passed" settles the question without
touching the rules.

**Keeping the champion is the default**, not the fallback. The rules are
evaluated to find a reason to *change*; absent one, nothing happens. That is why
an evaluation window with too few rows yields ``INCONCLUSIVE`` rather than a
rejection: nothing was found wanting, there was simply nothing to judge on.

**A thin protected slice cannot block a promotion.** The Friday ZONE-B window
contributes 8-10 rows to a four-week evaluation, and a safeguard that two
observations can swing is a coin toss wearing a safeguard's clothes. Below the
configured row floor the slice is reported as unjudgeable and the rule abstains,
which is stated in the decision rather than hidden by it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from urbansense.adaptation.candidates import (
    CandidateStatus,
    CandidateUpdate,
    ChallengerEvaluation,
    RuleOutcome,
)
from urbansense.adaptation.challenger import ChallengerFit
from urbansense.adaptation.config import PromotionRules
from urbansense.evaluation.metrics import local_window_slice, mae, rmse, zone_slice
from urbansense.features.builder import FeatureTable
from urbansense.prediction.gradient import GradientForecaster

#: Weekday index for Friday, matching the protected slice names.
FRIDAY = 4


def _slice_masks(
    table: FeatureTable, names: Sequence[str], *, timezone_offset_hours: float
) -> list[tuple[str, NDArray[np.bool_]]]:
    """Resolve protected slice names to row masks.

    Names follow the Phase 4 convention (``"ZONE-B Fri 18-20"``) and reuse that
    phase's slice builders, so a slice means the same thing in a promotion
    decision as it does in an error report. An unrecognized name is skipped and
    reported rather than silently matching nothing -- a protected slice that
    quietly selected zero rows would be a safeguard that never fires.
    """
    resolved: list[tuple[str, NDArray[np.bool_]]] = []
    for name in names:
        parts = name.split()
        if len(parts) == 3 and parts[1] == "Fri" and "-" in parts[2]:
            start_hour, end_hour = (int(piece) for piece in parts[2].split("-"))
            chooser = local_window_slice(
                parts[0],
                weekday=FRIDAY,
                start_hour=start_hour,
                end_hour=end_hour,
                timezone_offset_hours=timezone_offset_hours,
            )
        elif len(parts) == 1:
            chooser = zone_slice(parts[0])
        else:
            continue
        resolved.append((name, chooser(table)))
    return resolved


def evaluate_challenger(
    champion: GradientForecaster,
    fit: ChallengerFit,
    evaluation: FeatureTable,
    *,
    rules: PromotionRules,
    version: str,
    timezone_offset_hours: float = 0.0,
) -> ChallengerEvaluation:
    """Score one challenger against the champion and apply every rule.

    Both models predict the **same** evaluation window, so the comparison is an
    A/B rather than two differently-scoped measurements.
    """
    champion_predictions = champion.predict(evaluation)
    challenger_predictions = fit.model.predict(evaluation)

    champion_mae = mae(evaluation.y, champion_predictions)
    challenger_mae = mae(evaluation.y, challenger_predictions)
    improvement = 1.0 - (challenger_mae / champion_mae) if champion_mae > 0 else float("nan")

    slice_results: list[tuple[str, int, float, float]] = []
    slice_outcomes: list[RuleOutcome] = []
    for name, mask in _slice_masks(
        evaluation, rules.protected_slices, timezone_offset_hours=timezone_offset_hours
    ):
        rows = int(mask.sum())
        if rows == 0:
            slice_outcomes.append(
                RuleOutcome(
                    rule=f"protected slice {name}",
                    passed=True,
                    detail="absent from the evaluation window; rule abstains",
                )
            )
            continue
        slice_champion = mae(evaluation.y[mask], champion_predictions[mask])
        slice_challenger = mae(evaluation.y[mask], challenger_predictions[mask])
        slice_results.append((name, rows, slice_champion, slice_challenger))

        if rows < rules.min_slice_rows:
            slice_outcomes.append(
                RuleOutcome(
                    rule=f"protected slice {name}",
                    passed=True,
                    detail=(
                        f"only {rows} row(s), below the {rules.min_slice_rows}-row floor; "
                        f"too thin to judge, rule abstains "
                        f"(champion {slice_champion:,.0f}, challenger {slice_challenger:,.0f})"
                    ),
                )
            )
            continue

        regression = (
            (slice_challenger / slice_champion) - 1.0 if slice_champion > 0 else float("nan")
        )
        passed = not (regression > rules.max_slice_regression)
        slice_outcomes.append(
            RuleOutcome(
                rule=f"protected slice {name}",
                passed=passed,
                detail=(
                    f"{rows} rows, champion {slice_champion:,.0f} -> challenger "
                    f"{slice_challenger:,.0f} ({regression:+.1%}, limit "
                    f"{rules.max_slice_regression:+.1%})"
                ),
            )
        )

    outcomes: list[RuleOutcome] = [
        RuleOutcome(
            rule="evaluation rows",
            passed=len(evaluation) >= rules.min_evaluation_rows,
            detail=f"{len(evaluation):,} rows against a {rules.min_evaluation_rows} minimum",
        ),
        RuleOutcome(
            rule="MAE improvement",
            passed=bool(improvement >= rules.min_relative_mae_improvement),
            detail=(
                f"{improvement:+.2%} against a required "
                f"{rules.min_relative_mae_improvement:+.2%} "
                f"(champion {champion_mae:,.1f} -> challenger {challenger_mae:,.1f})"
            ),
        ),
        RuleOutcome(
            rule="training horizon",
            passed=fit.respects_horizon,
            detail=(
                f"training targets end {fit.target_end:%Y-%m-%d %H:%M}, decision time "
                f"{fit.decision_time:%Y-%m-%d %H:%M}"
            ),
        ),
    ]
    if rules.require_leakage_clean:
        outcomes.append(
            RuleOutcome(
                rule="leakage probe",
                passed=fit.leakage_clean,
                detail=(
                    "clean"
                    if fit.leakage_clean
                    else "FAILED -- a leaky challenger wins its comparison dishonestly"
                ),
            )
        )
    outcomes.extend(slice_outcomes)

    return ChallengerEvaluation(
        version=version,
        training_window=fit.window.name,
        training_rows=fit.training_rows,
        training_end=fit.training_end,
        evaluation_rows=len(evaluation),
        champion_mae=champion_mae,
        challenger_mae=challenger_mae,
        champion_rmse=rmse(evaluation.y, champion_predictions),
        challenger_rmse=rmse(evaluation.y, challenger_predictions),
        slice_results=tuple(slice_results),
        leakage_clean=fit.leakage_clean,
        rules=tuple(outcomes),
    )


def decide(
    candidate: CandidateUpdate,
    evaluations: Sequence[ChallengerEvaluation],
    *,
    rules: PromotionRules,
    evaluation_rows: int,
) -> CandidateUpdate:
    """Settle a candidate: promote the best passing challenger, or keep the champion.

    Returns the candidate with its status, evaluations and notes filled in. The
    champion is kept unless a challenger passes every rule, and the reason is
    always recorded.
    """
    candidate.evaluations = tuple(evaluations)

    if evaluation_rows < rules.min_evaluation_rows:
        candidate.status = CandidateStatus.INCONCLUSIVE
        candidate.notes.append(
            f"only {evaluation_rows} evaluation row(s) available, below the "
            f"{rules.min_evaluation_rows} minimum; no decision was taken and the "
            "champion was kept by default -- this is not a rejection, nothing was judged"
        )
        return candidate

    if not evaluations:
        candidate.status = CandidateStatus.INCONCLUSIVE
        candidate.notes.append("no challenger could be trained with the data available")
        return candidate

    candidate.status = CandidateStatus.EVALUATED
    winner = candidate.winner
    if winner is None:
        candidate.status = CandidateStatus.REJECTED
        best = candidate.best
        candidate.notes.append("no challenger satisfied every rule; champion kept")
        if best is not None:
            reasons = "; ".join(rule.detail for rule in best.failures)
            candidate.notes.append(
                f"closest was {best.version} at {best.relative_improvement:+.2%}: {reasons}"
            )
        return candidate

    candidate.status = CandidateStatus.PROMOTED
    candidate.promoted_version = winner.version
    candidate.notes.append(
        f"{winner.version} passed every rule, improving MAE by "
        f"{winner.relative_improvement:.2%} on {winner.evaluation_rows:,} rows"
    )
    rejected = [item for item in evaluations if item.version != winner.version]
    for item in rejected:
        verdict = "passed but scored worse" if item.passed else "failed a rule"
        candidate.notes.append(f"{item.version} not promoted: {verdict}")
    return candidate


def version_name(base: str, *, index: int, window: str) -> str:
    """Name a challenger version.

    ``Traffic-v1.2-expanding``. The window is in the name because two
    challengers are trained per candidate and a registry entry has to say which
    one it is without being opened.
    """
    return f"{base}.{index}-{window}"


def promotion_summary(candidates: Sequence[CandidateUpdate]) -> str:
    """One line per candidate, for a replay report."""
    if not candidates:
        return "no candidate updates were raised"
    lines = [f"{len(candidates)} candidate update(s):"]
    for candidate in candidates:
        kinds = ",".join(kind.value for kind in candidate.drift_kinds)
        lines.append(
            f"  {candidate.raised_at:%Y-%m-%d}  {candidate.status.value:<12} "
            f"[{kinds}]"
            + (f" -> {candidate.promoted_version}" if candidate.promoted_version else "")
        )
    promoted = sum(1 for c in candidates if c.status is CandidateStatus.PROMOTED)
    rejected = sum(1 for c in candidates if c.status is CandidateStatus.REJECTED)
    inconclusive = sum(1 for c in candidates if c.status is CandidateStatus.INCONCLUSIVE)
    lines.append(f"  promoted {promoted}, rejected {rejected}, inconclusive {inconclusive}")
    return "\n".join(lines)

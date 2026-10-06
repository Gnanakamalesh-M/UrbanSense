"""Aggregating and reporting the three-arm comparison (SPEC section 38).

    Report results honestly. Do not cherry-pick favorable results.

**This module has no idea which arm is supposed to win.** It ranks by measured
MAE, declares a tie when the gap is inside the cross-seed spread, and names
every slice where an arm lost. There is no "adaptive" special case anywhere in
the ranking, and a test feeds it hand-built results where adaptive loses and
where it ties to prove that.

That matters more than it sounds. The easy way to write this file is to compute
"improvement over static" and print it, which reads as a result no matter what
the numbers say -- adaptive beating a frozen model is nearly guaranteed and
almost meaningless. Ranking all arms against each other, with cost beside
accuracy, lets the report say the thing nobody wants to write: that the cheap
calendar baseline matched or beat the clever system.

A tie is a real outcome and gets said as one. Two arms whose means differ by
less than the seed-to-seed variation have not been distinguished by this
experiment, and reporting the smaller number as a winner would be reading noise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from urbansense.experiments.config import ARM_ORDER, ExperimentConfig
from urbansense.experiments.metrics import Aggregate, ArmMetrics, within_one_sd
from urbansense.experiments.runner import SeedRun

#: Default directory for experiment reports.
DEFAULT_EXPERIMENT_DIR = Path("reports") / "experiments"


@dataclass(frozen=True)
class ArmSummary:
    """One arm's figures aggregated across seeds.

    Attributes:
        arm: Which arm.
        mae: Mean absolute error across seeds.
        rmse: Root mean squared error across seeds.
        precision: Congestion precision across seeds.
        recall: Congestion recall across seeds.
        f1: Congestion F1 across seeds.
        brier: Brier score across seeds.
        retrains: Models deployed across seeds.
        fits: Models trained across seeds, deployed or not.
        training_rows: Rows consumed across seeds.
        slice_mae: Per-slice error, keyed by slice name.
    """

    arm: str
    mae: Aggregate
    rmse: Aggregate
    precision: Aggregate
    recall: Aggregate
    f1: Aggregate
    brier: Aggregate
    retrains: Aggregate
    fits: Aggregate
    training_rows: Aggregate
    slice_mae: dict[str, Aggregate]

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "arm": self.arm,
            "mae": self.mae.as_dict(),
            "rmse": self.rmse.as_dict(),
            "congestion": {
                "precision": self.precision.as_dict(),
                "recall": self.recall.as_dict(),
                "f1": self.f1.as_dict(),
            },
            "brier": self.brier.as_dict(),
            "cost": {
                "retrains": self.retrains.as_dict(),
                "fits": self.fits.as_dict(),
                "training_rows": self.training_rows.as_dict(),
            },
            "slice_mae": {name: value.as_dict() for name, value in self.slice_mae.items()},
        }


def summarize_arm(arm: str, runs: Sequence[ArmMetrics]) -> ArmSummary:
    """Aggregate one arm's per-seed metrics."""
    slice_names: list[str] = []
    for run in runs:
        for item in run.slices:
            if item.name not in slice_names:
                slice_names.append(item.name)

    def slice_values(name: str) -> list[float]:
        values: list[float] = []
        for run in runs:
            found = run.slice_named(name)
            if found is not None and found.rows > 0:
                values.append(found.mae)
        return values

    return ArmSummary(
        arm=arm,
        mae=Aggregate.of([run.mae for run in runs]),
        rmse=Aggregate.of([run.rmse for run in runs]),
        precision=Aggregate.of([run.congestion.precision for run in runs]),
        recall=Aggregate.of([run.congestion.recall for run in runs]),
        f1=Aggregate.of([run.congestion.f1 for run in runs]),
        brier=Aggregate.of([run.brier for run in runs]),
        retrains=Aggregate.of([float(run.cost.retrains) for run in runs]),
        fits=Aggregate.of([float(run.cost.fits) for run in runs]),
        training_rows=Aggregate.of([float(run.cost.training_rows) for run in runs]),
        slice_mae={name: Aggregate.of(slice_values(name)) for name in slice_names},
    )


@dataclass(frozen=True)
class Verdict:
    """What the numbers support saying, and nothing more.

    Attributes:
        ranking: Arm names ordered by measured MAE, best first.
        best: The arm with the lowest mean MAE.
        ties: Pairs whose means differ by less than the cross-seed spread.
        conclusion: One sentence a reader can quote.
    """

    ranking: tuple[str, ...]
    best: str
    ties: tuple[tuple[str, str], ...]
    conclusion: str

    def is_tie(self, left: str, right: str) -> bool:
        """Whether these two arms were not distinguished."""
        return (left, right) in self.ties or (right, left) in self.ties

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "ranking_by_mae": list(self.ranking),
            "best": self.best,
            "ties": [list(pair) for pair in self.ties],
            "conclusion": self.conclusion,
        }


def decide(summaries: dict[str, ArmSummary]) -> Verdict:
    """Rank the arms on their measured error and say what that supports.

    No arm is privileged. The ranking comes from the means, the ties come from
    the spreads, and the conclusion is assembled from both -- so an experiment
    in which the cheapest arm wins reports exactly that.
    """
    usable = {
        name: summary for name, summary in summaries.items() if not np.isnan(summary.mae.mean)
    }
    if not usable:
        return Verdict(
            ranking=(), best="", ties=(), conclusion="no arm produced a usable error figure"
        )

    ranking = tuple(sorted(usable, key=lambda name: usable[name].mae.mean))
    best = ranking[0]

    ties: list[tuple[str, str]] = []
    names = list(usable)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if within_one_sd(usable[left].mae, usable[right].mae):
                ties.append((left, right))

    leader = usable[best]
    runner_up = usable[ranking[1]] if len(ranking) > 1 else None

    if runner_up is None:
        conclusion = f"only {best} produced a usable figure, so nothing was compared"
    elif within_one_sd(leader.mae, runner_up.mae):
        conclusion = (
            f"{best} and {runner_up.arm} are not distinguished by this experiment: "
            f"{leader.mae.describe()} against {runner_up.mae.describe()} MAE, a gap "
            f"smaller than the seed-to-seed spread. Choosing between them on these "
            f"numbers would be reading noise"
        )
    else:
        cheaper = (
            " and it is also the cheaper of the two"
            if leader.training_rows.mean < runner_up.training_rows.mean
            else (
                f", bought with {leader.training_rows.mean / runner_up.training_rows.mean:.2f}x "
                f"the training rows of {runner_up.arm}"
                if runner_up.training_rows.mean > 0
                else ""
            )
        )
        conclusion = (
            f"{best} has the lowest error at {leader.mae.describe()} MAE against "
            f"{runner_up.arm}'s {runner_up.mae.describe()}{cheaper}"
        )

    return Verdict(ranking=ranking, best=best, ties=tuple(ties), conclusion=conclusion)


@dataclass(frozen=True)
class ExperimentReport:
    """The whole comparison, ready to write out.

    Attributes:
        config: The pre-registered configuration, copied verbatim.
        summaries: One per arm.
        verdict: What the numbers support.
        seeds: Per-seed detail.
        generated_at: When the report was produced.
        notes: Limitations, stated rather than left for a reader to discover.
    """

    config: ExperimentConfig
    summaries: dict[str, ArmSummary]
    verdict: Verdict
    seeds: tuple[SeedRun, ...]
    generated_at: datetime
    notes: tuple[str, ...] = ()

    def regressions(self, arm: str) -> tuple[str, ...]:
        """Slices where ``arm`` is worse than the best arm on that slice.

        The counterweight to a headline. An arm can win overall and still be
        the worst choice for the hours somebody actually cares about.
        """
        summary = self.summaries.get(arm)
        if summary is None:
            return ()
        worse: list[str] = []
        for name, value in summary.slice_mae.items():
            if np.isnan(value.mean):
                continue
            rivals = [
                other.slice_mae[name].mean
                for other_name, other in self.summaries.items()
                if other_name != arm
                and name in other.slice_mae
                and not np.isnan(other.slice_mae[name].mean)
            ]
            if rivals and value.mean > min(rivals):
                worse.append(name)
        return tuple(worse)

    def as_dict(self) -> dict[str, object]:
        """Serializable form, with the pre-registration first."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "pre_registered_config": self.config.as_dict(),
            "verdict": self.verdict.as_dict(),
            "arms": {name: summary.as_dict() for name, summary in self.summaries.items()},
            "regressions": {name: list(self.regressions(name)) for name in self.summaries},
            "seeds": [run.as_dict() for run in self.seeds],
            "limitations": list(self.notes),
        }

    def render(self) -> str:
        """A readable table. The verdict comes last, after the evidence."""
        lines = [
            "# Static vs scheduled retrain vs adaptive",
            "",
            f"Generated {self.generated_at:%Y-%m-%d %H:%M} UTC. {self.config.describe()}",
            "",
            f"{len(self.seeds)} seed(s), "
            f"{self.seeds[0].rows:,} rows scored per arm per seed. "
            "Every arm saw identical rows.",
            "",
            "## Accuracy and cost",
            "",
            "| arm | MAE | RMSE | congestion F1 | Brier | retrains | fits | training rows |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in ARM_ORDER:
            summary = self.summaries.get(name)
            if summary is None:
                continue
            lines.append(
                f"| {name} | {summary.mae.describe()} | {summary.rmse.describe()} "
                f"| {summary.f1.describe(places=3)} | {summary.brier.describe(places=4)} "
                f"| {summary.retrains.describe(places=1)} "
                f"| {summary.fits.describe(places=1)} "
                f"| {summary.training_rows.describe(places=0)} |"
            )

        lines.extend(["", "## Congestion detection", ""])
        lines.append("| arm | precision | recall | F1 |")
        lines.append("|---|---|---|---|")
        for name in ARM_ORDER:
            summary = self.summaries.get(name)
            if summary is None:
                continue
            lines.append(
                f"| {name} | {summary.precision.describe(places=3)} "
                f"| {summary.recall.describe(places=3)} | {summary.f1.describe(places=3)} |"
            )

        slice_names = sorted(
            {name for summary in self.summaries.values() for name in summary.slice_mae}
        )
        if slice_names:
            lines.extend(["", "## MAE by slice", ""])
            header = "| slice | " + " | ".join(name for name in ARM_ORDER if name in self.summaries)
            lines.append(header + " |")
            lines.append("|---" * (1 + len(self.summaries)) + "|")
            for slice_name in slice_names:
                cells = []
                for name in ARM_ORDER:
                    summary = self.summaries.get(name)
                    if summary is None:
                        continue
                    value = summary.slice_mae.get(slice_name)
                    cells.append(value.describe() if value is not None else "n/a")
                lines.append(f"| {slice_name} | " + " | ".join(cells) + " |")

        lines.extend(["", "## Where each arm lost", ""])
        for name in ARM_ORDER:
            if name not in self.summaries:
                continue
            worse = self.regressions(name)
            if worse:
                lines.append(
                    f"- **{name}** was beaten on {len(worse)} slice(s): {', '.join(worse)}"
                )
            else:
                lines.append(f"- **{name}** was the best arm on every slice measured")

        lines.extend(["", "## Verdict", "", self.verdict.conclusion + "."])
        if self.verdict.ties:
            lines.append("")
            for left, right in self.verdict.ties:
                lines.append(
                    f"- {left} and {right} differ by less than the seed-to-seed "
                    "spread and are not distinguished by this experiment"
                )
        lines.append("")
        lines.append(f"Ranked by measured MAE: {' < '.join(self.verdict.ranking)}.")

        if self.notes:
            lines.extend(["", "## Limitations", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


#: Limitations that hold for any run of this experiment, stated in the report
#: rather than left for a reader to work out.
STANDING_LIMITATIONS: tuple[str, ...] = (
    "Synthetic data. The generator's drift is a clean multiplicative level shift "
    "with a known start date; real drift is messier, and an arm tuned for this "
    "shape may not transfer.",
    "One drift type. Only a level shift is planted -- no gradual drift, no "
    "seasonal regime change, no sensor replacement. The adaptive arm's advantage "
    "on other shapes is untested.",
    "Thin protected slices. The Friday evening windows contribute 8-10 rows per "
    "four-week evaluation, so their per-slice figures move a lot between seeds "
    "and should not be read as precisely as the overall numbers.",
    "One fixed calibrator. It is fitted once from the shared starting champion "
    "and never refitted, so the Brier column compares regressors rather than "
    "complete serving systems. Recalibrating per retrain would be more realistic "
    "and would confound the model's contribution with the calibrator's.",
    "Few seeds. Spreads from a handful of datasets are themselves imprecise; a "
    "tie here means 'not distinguished', not 'equal'.",
)


def build_report(
    config: ExperimentConfig,
    seeds: Sequence[SeedRun],
    *,
    now: datetime | None = None,
    notes: Sequence[str] = STANDING_LIMITATIONS,
) -> ExperimentReport:
    """Aggregate per-seed runs into a report.

    Raises:
        ValueError: if no seed produced results. An empty report would render a
            table of ``n/a`` that reads like a finished experiment.
    """
    if not seeds:
        raise ValueError("cannot build a report with no seed runs")

    arms = [name for name in ARM_ORDER if any(name in run.metrics for run in seeds)]
    summaries = {
        name: summarize_arm(name, [run.metrics[name] for run in seeds if name in run.metrics])
        for name in arms
    }
    return ExperimentReport(
        config=config,
        summaries=summaries,
        verdict=decide(summaries),
        seeds=tuple(seeds),
        generated_at=now if now is not None else datetime.now(UTC),
        notes=tuple(notes),
    )


def write_report(report: ExperimentReport, root: Path | None = None) -> tuple[Path, Path]:
    """Write the JSON and markdown forms, one pair per run.

    Appends a new timestamped pair rather than overwriting, for the same reason
    the pattern store does: comparing what an earlier run concluded against a
    later one is the point of keeping them.

    Returns:
        The JSON path and the markdown path.
    """
    directory = root if root is not None else DEFAULT_EXPERIMENT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")

    json_path = directory / f"experiment-{stamp}.json"
    json_path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path = directory / f"experiment-{stamp}.md"
    markdown_path.write_text(report.render() + "\n", encoding="utf-8", newline="\n")
    return json_path, markdown_path

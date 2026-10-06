"""Replaying a stream to compare a static model against an adaptive system.

    New Data -> Validation -> Drift Detection -> Candidate Update
    -> Train Candidate -> Evaluate -> Compare with Champion   (SPEC section 30)

The whole loop, run forward over a stream in time order, with up to three arms
scored on **the same rows**:

- **static** -- the original champion, frozen. Never retrained.
- **scheduled** -- retrained every ``scheduled_every`` and deployed
  unconditionally. The fair baseline: "adaptive beats a model that never
  retrains" is a weak claim, because the cheap alternative is a calendar.
- **adaptive** -- starts as the same model, and may be replaced when a candidate
  update passes the promotion rules.

All three predict the *same* ``window`` slice inside one loop, so "every arm saw
identical evaluation rows" is true by construction. Two parallel harnesses could
not guarantee it: any difference in how each sliced the stream would show up as
a difference in accuracy.

Scoring every arm on identical rows is what makes the comparison an A/B rather
than two differently-scoped runs. The adaptive arm predicts each window with
whichever model was champion *at that point in the replay*, never with the final
one -- using the end-state model to score the whole stream would credit it with
months it did not serve.

**The static arm is the pre-drift champion, deliberately.** Comparing against
Phase 6's ``Traffic-v1.1`` would be meaningless: that model trained across the
planted shift, so there would be nothing for the adaptive system to adapt to.

Nothing here reads ``ground_truth.json``. The replay discovers the change point
the same way a deployment would -- by noticing its error rise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from urbansense.adaptation.candidates import (
    CandidateStatus,
    CandidateUpdate,
    candidate_from_event,
)
from urbansense.adaptation.challenger import (
    ChallengerError,
    ChallengerFit,
    evaluation_window,
    train_challenger,
)
from urbansense.adaptation.config import AdaptationConfig
from urbansense.adaptation.lineage import (
    assert_single_champion,
    promote,
    register_challenger,
    reject,
    replay_run_root,
)
from urbansense.adaptation.promotion import decide, evaluate_challenger, version_name
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.drift.events import DriftEpisode, DriftEvent
from urbansense.drift.monitor import DriftMonitor, MonitorConfig
from urbansense.evaluation.metrics import SliceFn, SliceMetrics, evaluate_slice
from urbansense.features.builder import FeatureTable
from urbansense.features.leakage import LeakageReport, detect_leakage
from urbansense.prediction.gradient import GradientForecaster
from urbansense.prediction.registry import ModelRegistry, ModelVersion
from urbansense.schemas.enums import ModelStatus


@dataclass(frozen=True)
class RetrainCost:
    """What an arm spent to reach its accuracy.

    Attributes:
        retrains: Models actually deployed.
        fits: Models trained, deployed or not. Larger than ``retrains`` for the
            adaptive arm, which trains several challengers and keeps one.
        training_rows: Rows consumed across every fit.
    """

    retrains: int = 0
    fits: int = 0
    training_rows: int = 0

    def plus(self, *, retrains: int = 0, fits: int = 0, rows: int = 0) -> RetrainCost:
        """A copy with the given amounts added."""
        return RetrainCost(
            retrains=self.retrains + retrains,
            fits=self.fits + fits,
            training_rows=self.training_rows + rows,
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "retrains": self.retrains,
            "fits": self.fits,
            "training_rows": self.training_rows,
        }

    def describe(self) -> str:
        """One line."""
        return (
            f"{self.retrains} retrain(s), {self.fits} fit(s), "
            f"{self.training_rows:,} training row(s)"
        )


@dataclass(frozen=True)
class ArmResult:
    """One arm's error over the replayed stream.

    Attributes:
        name: Which arm.
        overall: Metrics across every scored row.
        slices: Per-slice metrics.
        rows: Rows scored.
        models_used: Distinct models that served during the replay.
        cost: What the arm spent to get its accuracy. Reported beside the error
            because an accuracy table on its own lets the most expensive arm
            look best for free.
        predictions: This arm's prediction for every scored row, in scoring
            order. Kept because the adaptive and scheduled arms are
            time-varying: their predictions cannot be reproduced after the run
            by re-predicting, since the model that made each one has already
            been replaced.
    """

    name: str
    overall: SliceMetrics
    slices: tuple[SliceMetrics, ...]
    rows: int
    models_used: tuple[str, ...]
    cost: RetrainCost = field(default_factory=lambda: RetrainCost())
    predictions: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def slice_named(self, name: str) -> SliceMetrics | None:
        """One slice by name."""
        return next((item for item in self.slices if item.name == name), None)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "arm": self.name,
            "rows": self.rows,
            "models_used": list(self.models_used),
            "cost": self.cost.as_dict(),
            "overall": self.overall.as_dict(),
            "slices": [item.as_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class ReplayResult:
    """Everything one replay produced.

    Attributes:
        scenario: Which scenario ran.
        events: Every drift evaluation, in time order.
        candidates: Candidate updates and their decisions.
        static: The frozen champion's result.
        adaptive: The adaptive system's result.
        scheduled: The calendar-retrained arm's result, when that arm ran.
        scored_from: First instant scored.
        scored_to: Last instant scored.
        gates: The drift gates in force, as a dict.
        actual: The observed target for every scored row, in scoring order.
        locations: Zone per scored row.
        record_ids: Source record id per scored row. Every arm shares these --
            that is what "scored on identical rows" means, and comparing ids
            rather than row counts is what makes the claim checkable.
        episodes: Drift episodes the monitor accumulated, for the log.
        notes: Honest caveats about the run.
    """

    scenario: str
    events: tuple[DriftEvent, ...]
    candidates: tuple[CandidateUpdate, ...]
    static: ArmResult
    adaptive: ArmResult
    scored_from: datetime
    scored_to: datetime
    scheduled: ArmResult | None = None
    gates: dict[str, object] = field(default_factory=dict)
    actual: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    locations: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    episodes: tuple[DriftEpisode, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def relative_improvement(self) -> float:
        """Fraction by which adaptive beats static. Negative means it lost."""
        if self.static.overall.mae <= 0:
            return float("nan")
        return 1.0 - (self.adaptive.overall.mae / self.static.overall.mae)

    @property
    def adaptation_helped(self) -> bool:
        """Whether the adaptive arm actually came out ahead overall."""
        return self.adaptive.overall.mae < self.static.overall.mae

    @property
    def arms(self) -> dict[str, ArmResult]:
        """Every arm that ran, in report order.

        Keyed by name so a report can iterate without knowing which arms a
        particular run included.
        """
        found = {"static": self.static, "adaptive": self.adaptive}
        if self.scheduled is not None:
            found["scheduled"] = self.scheduled
        return {name: found[name] for name in ("static", "scheduled", "adaptive") if name in found}

    @property
    def promotions(self) -> tuple[CandidateUpdate, ...]:
        """Candidates that resulted in a promotion."""
        return tuple(item for item in self.candidates if item.status is CandidateStatus.PROMOTED)

    @property
    def regressed_slices(self) -> tuple[str, ...]:
        """Slices where adaptation made things worse.

        Reported because an overall improvement can hide a slice getting worse,
        and a report that printed only the headline would be the overselling
        this phase is meant to avoid.
        """
        worse: list[str] = []
        for item in self.adaptive.slices:
            baseline = self.static.slice_named(item.name)
            if baseline is not None and item.mae > baseline.mae:
                worse.append(item.name)
        return tuple(worse)

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "scenario": self.scenario,
            "scored": {
                "from": self.scored_from.isoformat(),
                "to": self.scored_to.isoformat(),
                "rows": self.static.rows,
            },
            "gates": self.gates,
            "drift_events": len(self.events),
            "drift_fired": sum(1 for event in self.events if event.is_drift),
            "candidates": [item.as_dict() for item in self.candidates],
            "static": self.static.as_dict(),
            "adaptive": self.adaptive.as_dict(),
            "relative_improvement": round(self.relative_improvement, 6),
            "adaptation_helped": self.adaptation_helped,
            "regressed_slices": list(self.regressed_slices),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """The static-versus-adaptive table, slices included.

        Prints the slices where adaptation did not help in their own section,
        so a headline improvement cannot stand alone.
        """
        lines = [
            f"static vs adaptive -- {self.scenario} "
            f"({self.static.rows:,} rows, {self.scored_from:%Y-%m-%d} to "
            f"{self.scored_to:%Y-%m-%d})",
            "",
            f"  {'slice':<24}{'static MAE':>13}{'adaptive MAE':>15}{'change':>10}",
            f"  {'overall':<24}{self.static.overall.mae:>13,.1f}"
            f"{self.adaptive.overall.mae:>15,.1f}"
            f"{-self.relative_improvement:>9.1%}",
        ]
        for item in self.adaptive.slices:
            baseline = self.static.slice_named(item.name)
            if baseline is None or baseline.rows == 0:
                continue
            change = (item.mae / baseline.mae) - 1.0 if baseline.mae > 0 else float("nan")
            thin = "  (thin)" if item.rows < 30 else ""
            lines.append(
                f"  {item.name:<24}{baseline.mae:>13,.1f}{item.mae:>15,.1f}{change:>9.1%}{thin}"
            )

        lines.append("")
        lines.append(
            f"  RMSE overall: static {self.static.overall.rmse:,.1f} -> adaptive "
            f"{self.adaptive.overall.rmse:,.1f}"
        )
        lines.append(f"  models served in the adaptive arm: {', '.join(self.adaptive.models_used)}")

        if self.adaptation_helped:
            lines.append(f"  adaptation reduced overall MAE by {self.relative_improvement:.1%}")
        else:
            lines.append(
                f"  adaptation did NOT help: overall MAE moved {-self.relative_improvement:+.1%}"
            )
        regressed = self.regressed_slices
        if regressed:
            lines.append(f"  but it was worse on {len(regressed)} slice(s): {', '.join(regressed)}")
        else:
            lines.append("  no slice got worse")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


#: A named row filter over a feature table, matching the Phase 4 convention in
#: urbansense.evaluation.metrics.
SliceSpec = Sequence[tuple[str, SliceFn]]


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    """Rows whose sample time falls in ``[start, end)``."""
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)


@dataclass
class AdaptiveReplay:
    """Runs the detect-investigate-train-compare loop over a stream.

    Attributes:
        table: The full feature table to replay.
        reconciled: The unified layer, for the leakage probe.
        champion: The starting champion, trained before the replay begins.
        reference: The out-of-sample reference window for drift gates.
        config: The adaptation rules.
        training_end: Exclusive upper bound of the champion's training period.
            Passed to the monitor, which refuses a reference the champion has
            already seen.
        base_version: Version name the challengers are numbered from.
        timezone_offset_hours: Local offset, for slice definitions.
    """

    table: FeatureTable
    reconciled: ReconciledResult
    champion: GradientForecaster
    reference: FeatureTable
    config: AdaptationConfig
    training_end: datetime | None = None
    base_version: str = "Traffic-v1"
    timezone_offset_hours: float = 0.0

    def _monitor_config(self) -> MonitorConfig:
        """Translate the adaptation rules into a monitor configuration."""
        rules = self.config.monitoring
        return MonitorConfig(
            step=rules.step,
            window=rules.window,
            watched_features=rules.watched_features,
            sigma_multiplier=rules.sigma_multiplier,
            min_reference_windows=rules.min_reference_windows,
        )

    def run(
        self,
        *,
        scenario: str,
        start: datetime,
        end: datetime,
        slices: SliceSpec = (),
        scheduled_every: timedelta | None = None,
        scheduled_window: str = "expanding",
    ) -> ReplayResult:
        """Replay the stream, adapting where the rules allow.

        Args:
            scenario: Name for the report.
            start: First evaluation instant.
            end: Last evaluation instant.
            slices: Named row filters for the comparison table.
            scheduled_every: Cadence for the scheduled-retrain arm. ``None``
                leaves that arm out, which is what Phase 7's two-arm replay did.
            scheduled_window: Which configured training window the scheduled arm
                deploys. It cannot pick by evaluating both -- that would be
                peeking at the evaluation window, which the promotion rules are
                allowed to do and an unconditional baseline is not.

        Returns:
            A :class:`ReplayResult` with every arm scored on identical rows.
        """
        monitor = DriftMonitor(
            reference=self.reference,
            predict=self.champion.predict,
            config=self._monitor_config(),
            training_end=self.training_end,
        )
        monitor.tracker.cooldown = self.config.monitoring.cooldown
        monitor.tracker.required_kinds = self.config.monitoring.required_kinds
        gates = monitor.calibrate()

        serving: GradientForecaster = self.champion
        serving_version = f"{self.base_version}.1-static"
        models_used: list[str] = [serving_version]

        # The scheduled arm. It starts from the same frozen champion, so before
        # its first retrain the three arms agree exactly -- which is a useful
        # sanity property: any early divergence would be a bug.
        scheduled_window_spec = next(
            (
                window
                for window in self.config.candidates.training_windows
                if window.name == scheduled_window
            ),
            None,
        )
        if scheduled_every is not None and scheduled_window_spec is None:
            raise ValueError(
                f"scheduled_window {scheduled_window!r} is not one of the configured "
                f"training windows "
                f"{[w.name for w in self.config.candidates.training_windows]}; the "
                "scheduled arm must use the same training machinery as the adaptive "
                "one or the comparison is not like for like"
            )
        scheduled_serving: GradientForecaster = self.champion
        scheduled_models: list[str] = [f"{self.base_version}.1-static"]
        scheduled_cost = RetrainCost()
        next_retrain = start + scheduled_every if scheduled_every is not None else None

        events: list[DriftEvent] = []
        candidates: list[CandidateUpdate] = []
        notes: list[str] = []
        promotion_index = 1

        # Scored rows, accumulated per window so every arm sees the same rows.
        scored_actual: list[NDArray[np.float64]] = []
        static_predictions: list[NDArray[np.float64]] = []
        adaptive_predictions: list[NDArray[np.float64]] = []
        scheduled_predictions: list[NDArray[np.float64]] = []
        scored_tables: list[FeatureTable] = []
        adaptive_cost = RetrainCost()

        moment = start
        step = self.config.monitoring.step
        while moment <= end:
            # --- score the window that is now in the past ------------------
            window = _window(self.table, moment, moment + step)
            if len(window) > 0:
                scored_tables.append(window)
                scored_actual.append(window.y)
                # One slice, three predictions. This is what makes "identical
                # evaluation rows" structural rather than something to compare
                # afterwards and hope about.
                static_predictions.append(self.champion.predict(window))
                adaptive_predictions.append(serving.predict(window))
                if scheduled_every is not None:
                    scheduled_predictions.append(scheduled_serving.predict(window))

            # --- the scheduled arm's calendar ------------------------------
            # Checked before the drift look-up, so a retrain and a candidate
            # falling on the same instant are both evaluated against the same
            # decision time rather than one seeing a model the other does not.
            if (
                scheduled_every is not None
                and next_retrain is not None
                and scheduled_window_spec is not None
                and moment >= next_retrain
            ):
                try:
                    scheduled_fit = replace(
                        train_challenger(
                            self.table,
                            decision_time=moment,
                            window=scheduled_window_spec,
                            probe_leakage=False,
                        ),
                        leakage=self._shared_leakage(),
                    )
                except ChallengerError as exc:
                    notes.append(f"scheduled retrain at {moment:%Y-%m-%d} skipped: {exc}")
                else:
                    # Deployed unconditionally. That is the arm's definition: no
                    # rules, no margin, no veto -- which is exactly what makes it
                    # the baseline the adaptive arm has to beat.
                    scheduled_serving = scheduled_fit.model
                    scheduled_cost = scheduled_cost.plus(
                        retrains=1, fits=1, rows=scheduled_fit.training_rows
                    )
                    scheduled_models.append(
                        f"{self.base_version}.s{scheduled_cost.retrains}-{scheduled_window}"
                    )
                next_retrain = next_retrain + scheduled_every

            # --- look for drift in the window just closed ------------------
            event = monitor.evaluate(self.table, at=moment)
            stamped, raise_candidate = monitor.tracker.observe(event)
            events.append(stamped)

            if raise_candidate:
                candidate = candidate_from_event(
                    stamped, decision_time=moment, champion_version=serving_version
                )
                promotion_index += 1
                candidate = self._investigate(candidate, index=promotion_index, slices=slices)
                candidates.append(candidate)
                for evaluation in candidate.evaluations:
                    adaptive_cost = adaptive_cost.plus(fits=1, rows=evaluation.training_rows)

                winner = candidate.winner
                if winner is not None and candidate.status is CandidateStatus.PROMOTED:
                    fit = self._fits.get(winner.version)
                    if fit is not None:
                        serving = fit.model
                        serving_version = winner.version
                        models_used.append(serving_version)
                        adaptive_cost = adaptive_cost.plus(retrains=1)

            moment = moment + step

        monitor.tracker.close(end)

        if not scored_tables:
            raise ValueError(f"nothing to score between {start.isoformat()} and {end.isoformat()}")

        actual = np.concatenate(scored_actual)
        static = np.concatenate(static_predictions)
        adaptive = np.concatenate(adaptive_predictions)
        scheduled = np.concatenate(scheduled_predictions) if scheduled_predictions else None

        if len(models_used) == 1:
            notes.append(
                "the adaptive arm never changed model, so it and the static arm are "
                "the same run; any difference between them would be a bug"
            )
        notes.append(
            "every arm is scored on identical rows, each window predicted by "
            "whichever model that arm had serving at that point in the replay"
        )

        if scheduled_every is not None:
            notes.append(
                f"the scheduled arm retrained {scheduled_cost.retrains} time(s) on a "
                f"{scheduled_every.days}-day calendar and deployed every one "
                "unconditionally; the adaptive arm retrained "
                f"{adaptive_cost.retrains} time(s), only on a promotion"
            )

        return ReplayResult(
            scenario=scenario,
            events=tuple(events),
            candidates=tuple(candidates),
            static=self._arm("static", scored_tables, actual, static, slices, (models_used[0],)),
            adaptive=self._arm(
                "adaptive",
                scored_tables,
                actual,
                adaptive,
                slices,
                tuple(models_used),
                cost=adaptive_cost,
            ),
            scheduled=(
                None
                if scheduled is None
                else self._arm(
                    "scheduled",
                    scored_tables,
                    actual,
                    scheduled,
                    slices,
                    tuple(scheduled_models),
                    cost=scheduled_cost,
                )
            ),
            scored_from=min(min(t.sample_times) for t in scored_tables),
            scored_to=max(max(t.sample_times) for t in scored_tables),
            gates=gates.as_dict(),
            actual=actual,
            locations=tuple(zone for table in scored_tables for zone in table.locations),
            record_ids=tuple(record for table in scored_tables for record in table.record_ids),
            episodes=tuple(monitor.tracker.episodes),
            notes=tuple(notes),
        )

    _fits: dict[str, ChallengerFit] = field(default_factory=dict, repr=False)
    _leakage: LeakageReport | None = field(default=None, repr=False)

    def _investigate(
        self, candidate: CandidateUpdate, *, index: int, slices: SliceSpec
    ) -> CandidateUpdate:
        """Train and judge the challengers for one candidate update."""
        rules = self.config.candidates
        evaluation = evaluation_window(
            self.table,
            decision_time=candidate.decision_time,
            gap=rules.gap,
            span=rules.evaluation_span,
        )

        shared_leakage = self._shared_leakage()

        evaluations = []
        for window in rules.training_windows:
            version = version_name(self.base_version, index=index, window=window.name)
            try:
                fit = replace(
                    train_challenger(
                        self.table,
                        decision_time=candidate.decision_time,
                        window=window,
                        probe_leakage=False,
                    ),
                    leakage=shared_leakage,
                )
            except ChallengerError as exc:
                candidate.notes.append(f"{version}: not trained -- {exc}")
                continue

            if len(evaluation) == 0:
                candidate.notes.append(
                    f"{version}: trained but not judged -- no evaluation rows after "
                    f"the {rules.gap_hours}h gap"
                )
                continue

            self._fits[version] = fit
            evaluations.append(
                evaluate_challenger(
                    self.champion
                    if candidate.champion_version.endswith("static")
                    else self._fits.get(candidate.champion_version, fit).model,
                    fit,
                    evaluation,
                    rules=self.config.promotion,
                    version=version,
                    timezone_offset_hours=self.timezone_offset_hours,
                )
            )

        return decide(
            candidate,
            evaluations,
            rules=self.config.promotion,
            evaluation_rows=len(evaluation),
        )

    def starting_entry(self, version: str) -> ModelVersion:
        """The frozen champion's registry entry, which challengers inherit from.

        Carries the SPEC section 31 fields that do not change between versions
        -- algorithm, hyperparameters, feature descriptions -- so a challenger's
        entry cannot drift out of step with the lineage it belongs to.
        """
        described = self.champion.describe()
        earliest = min(self.table.sample_times)
        boundary = (
            self.training_end if self.training_end is not None else min(self.reference.sample_times)
        )
        training = _window(self.table, earliest, boundary)
        return ModelVersion(
            version=version,
            problem_type="traffic",
            target=self.table.feature_set.target_metric,
            horizon_hours=self.table.feature_set.horizon_hours,
            algorithm=str(described["algorithm"]),
            algorithm_rationale=str(described["algorithm_rationale"]),
            hyperparameters=self.champion.hyperparameters,
            features=self.table.feature_set.describe(),
            training_period={
                "start": min(training.sample_times).isoformat(),
                "end": max(training.sample_times).isoformat(),
                "rows": len(training),
                "window": "frozen pre-drift champion",
            },
            validation_period={
                "start": min(self.reference.sample_times).isoformat(),
                "end": max(self.reference.sample_times).isoformat(),
                "role": "out-of-sample drift reference",
            },
            test_period={"role": "the replayed stream itself"},
            gap_hours=self.table.feature_set.max_reach_hours,
            metrics=[],
            leakage_checks={"probed_per_challenger": True},
            excluded_rows={},
            dataset_version={"source": "data/sample"},
            interval={},
            status=ModelStatus.CHAMPION,
            notes=[
                "The starting champion of a replay scenario, frozen for the run "
                "so the static arm has something to be static about.",
            ],
        )

    def record(
        self,
        result: ReplayResult,
        *,
        root: Path,
        now: datetime | None = None,
    ) -> tuple[Path, str]:
        """Write the run's lineage to a **fresh** registry under ``root``.

        Each call gets its own run directory, so replaying a scenario twice --
        or replaying the other scenario -- cannot inherit a champion from a
        previous run. That was a real failure: a second run re-seeded its
        starting champion beside the first run's, and the next promotion raised
        ``2 champions`` naming a version the run had not reached yet.

        Returns:
            The registry directory written, and the final champion.
        """
        run_root = replay_run_root(root, scenario=result.scenario, now=now)
        store = ModelRegistry(run_root / "registry", run_root / "artifacts")

        base = self.starting_entry(result.adaptive.models_used[0])
        store.save(base)
        # A fresh directory should hold exactly the seed. Checked rather than
        # assumed, so a dirty root fails here instead of several promotions later.
        serving = assert_single_champion(store)

        for candidate in result.candidates:
            for evaluation in candidate.evaluations:
                promoted = candidate.promoted_version == evaluation.version
                register_challenger(
                    store,
                    version=evaluation.version,
                    base=base,
                    candidate_id=candidate.candidate_id,
                    drift_episode_id=candidate.episode_id,
                    supersedes=serving if promoted else None,
                    training_rows=evaluation.training_rows,
                    training_start=min(self.table.sample_times).isoformat(),
                    training_end=evaluation.training_end.isoformat(),
                    training_window=evaluation.training_window,
                    metrics=[{"overall": {"mae": round(evaluation.challenger_mae, 4)}}],
                    leakage={"clean": evaluation.leakage_clean},
                    decision=evaluation.as_dict(),
                )
                if promoted:
                    serving, _ = promote(store, evaluation.version, incumbent=serving)
                else:
                    reject(store, evaluation.version)

        return store.root, assert_single_champion(store)

    def _shared_leakage(self) -> LeakageReport:
        """The leakage verdict for this run's feature set, computed once.

        The probe is the slowest step in a replay and every challenger -- in
        either arm -- shares one feature set, so it runs once per replay and the
        verdict is attached to each fit.

        Running it for only the first training window, which an earlier version
        did, left the others with no report at all and `leakage_clean` returning
        true vacuously. The rule then applied to one challenger and not its
        rivals, which is worse than not having it: a false failure on the probed
        one handed promotions to unprobed ones. Caching it here also keeps the
        scheduled arm held to the same standard as the adaptive arm, which is
        the whole basis of the comparison.
        """
        if self._leakage is None:
            self._leakage = detect_leakage(
                self.reconciled,
                self.table.feature_set,
                timezone_offset_hours=self.timezone_offset_hours,
            )
        return self._leakage

    def _arm(
        self,
        name: str,
        tables: Sequence[FeatureTable],
        actual: NDArray[np.float64],
        predicted: NDArray[np.float64],
        slices: SliceSpec,
        models_used: tuple[str, ...],
        cost: RetrainCost | None = None,
    ) -> ArmResult:
        """Score one arm overall and per slice."""
        computed: list[SliceMetrics] = []
        for slice_name, chooser in slices:
            masks = [chooser(table) for table in tables]
            mask = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
            computed.append(evaluate_slice(slice_name, actual[mask], predicted[mask]))
        return ArmResult(
            name=name,
            overall=evaluate_slice("overall", actual, predicted),
            slices=tuple(computed),
            rows=int(actual.size),
            models_used=models_used,
            cost=cost if cost is not None else RetrainCost(),
            predictions=predicted,
        )


def scenario_bounds(
    table: FeatureTable, *, train_days: int, reference_days: int
) -> tuple[datetime, datetime, datetime]:
    """Split a stream into train end, reference end and replay end.

    A helper so the two scenarios derive their boundaries from the data rather
    than hard-coding dates, which would make a regenerated dataset silently
    change what is being tested.
    """
    earliest = min(table.sample_times)
    latest = max(table.sample_times)
    train_end = earliest + timedelta(days=train_days)
    reference_end = train_end + timedelta(days=reference_days)
    return train_end, reference_end, latest

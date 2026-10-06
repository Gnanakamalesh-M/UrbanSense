"""Running the three-arm comparison across seeds (SPEC sections 38, 39).

One dataset is an anecdote. The planted drift is the same in every generated
stream, but the noise, the rainfall and the missing-data draws are not, and a
difference between arms smaller than that variation is not a finding. So each
seed is generated, run and scored independently, and the report carries mean and
spread rather than a single number.

**Data is generated into a caller-supplied directory and ``data/sample`` is
never written.** The committed demo dataset is a fixture other phases' tests
assert against; regenerating it here with a different seed would break them in a
way that looks unrelated to this module.

**Nothing here reads ``ground_truth.json``.** The experiment measures error
against observed values; it does not need to know where the drift was planted,
and the arms certainly must not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from urbansense.adaptation.config import AdaptationConfig, load_adaptation_config
from urbansense.adaptation.replay import AdaptiveReplay, ReplayResult
from urbansense.config import Settings, load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.data_integrity.unified import ReconciledResult
from urbansense.evaluation.metrics import friday_evening_slices
from urbansense.experiments.config import ADAPTIVE, SCHEDULED, STATIC, ExperimentConfig
from urbansense.experiments.metrics import ArmMetrics, score_arm
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import ZoneRegistry, load_historical
from urbansense.prediction.calibration import ProbabilityCalibrator
from urbansense.prediction.congestion import CongestionThresholds, fit_thresholds
from urbansense.prediction.gradient import GradientForecaster
from urbansense.preprocessing import load_location_registry
from urbansense.schemas import ProblemType
from urbansense.synthetic import load_generator_config
from urbansense.synthetic.generator import (
    HISTORY_FILENAME,
    ZONES_FILENAME,
    generate_to_directory,
)


@dataclass(frozen=True)
class SeedRun:
    """Everything one seed produced.

    Attributes:
        seed: The generator seed.
        metrics: One entry per arm.
        replay: The underlying replay, kept so a caller can inspect the drift
            events and candidate decisions behind the numbers.
        rows: Rows scored, identical across arms.
        thresholds: The shared congestion definition used for this seed.
    """

    seed: int
    metrics: dict[str, ArmMetrics]
    replay: ReplayResult
    rows: int
    thresholds: CongestionThresholds

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "seed": self.seed,
            "rows": self.rows,
            "congestion_thresholds": self.thresholds.as_dict()["by_zone"],
            "arms": {name: item.as_dict() for name, item in self.metrics.items()},
            "drift_events_fired": sum(1 for event in self.replay.events if event.is_drift),
            "candidates": [item.as_dict() for item in self.replay.candidates],
        }


@dataclass
class ExperimentRunner:
    """Generates, replays and scores one seed at a time.

    Attributes:
        config: The pre-registered experiment configuration.
        workspace: Directory the per-seed datasets are generated into. A caller
            passes a temporary path; nothing here touches ``data/sample``.
        settings: Problem configuration.
        adaptation: The Phase 7 rules every arm obeys.
    """

    config: ExperimentConfig
    workspace: Path
    settings: Settings = field(default_factory=load_settings)
    adaptation: AdaptationConfig = field(default_factory=load_adaptation_config)

    def _generate(self, seed: int) -> Path:
        """Generate one seed's dataset and return its directory."""
        base = load_generator_config()
        target = self.workspace / f"seed-{seed}"
        generate_to_directory(base.model_copy(update={"seed": seed}), target, force=True)
        return target

    def _table(self, directory: Path) -> tuple[FeatureTable, ReconciledResult]:
        """Ingest, validate, reconcile and build features for one dataset."""
        zones = directory / ZONES_FILENAME
        registry = ZoneRegistry.from_csv(zones) if zones.is_file() else ZoneRegistry.empty()
        validated = validate_and_normalize(
            load_historical(directory / HISTORY_FILENAME, registry=registry),
            settings=self.settings,
            registry=load_location_registry(),
        )
        reconciled = reconcile(validated, settings=self.settings)
        table = build_feature_table(
            reconciled,
            load_feature_set(),
            timezone_offset_hours=load_generator_config().timezone_offset_hours,
        )
        return table, reconciled

    def run_seed(self, seed: int) -> SeedRun:
        """Generate, replay and score one seed.

        The boundaries are spans from the start of this seed's own stream, so a
        different seed cannot silently change what is being measured.
        """
        directory = self._generate(seed)
        table, reconciled = self._table(directory)
        offset = load_generator_config().timezone_offset_hours

        earliest = min(table.sample_times)
        train_end = earliest + timedelta(days=self.config.train_days)
        reference_end = train_end + timedelta(days=self.config.reference_days)
        latest = max(table.sample_times)
        replay_end = (
            latest
            if self.config.replay_days is None
            else min(latest, reference_end + timedelta(days=self.config.replay_days))
        )

        training = _window(table, earliest, train_end)
        champion = GradientForecaster().fit(training)

        # The congestion definition and the probability calibrator are both
        # fitted here, once, from train/validation only, and shared by every
        # arm. See metrics.py for why sharing them is what keeps the comparison
        # about the models.
        congestion = self.settings.problems[ProblemType.TRAFFIC].congestion
        if congestion is None:
            raise ValueError("the traffic problem config defines no congestion block")
        thresholds = fit_thresholds(training, congestion)

        reference = _window(table, train_end, reference_end)
        calibrator = ProbabilityCalibrator.from_residuals(training.y, champion.predict(training))
        reference_thresholds = np.array(
            [thresholds.by_zone.get(zone, float("nan")) for zone in reference.locations],
            dtype=np.float64,
        )
        calibrator.fit(
            calibrator.raw(champion.predict(reference), reference_thresholds),
            thresholds.congested_mask(reference),
        )

        replay = AdaptiveReplay(
            table=table,
            reconciled=reconciled,
            champion=champion,
            reference=reference,
            config=self.adaptation,
            training_end=train_end,
            timezone_offset_hours=offset,
        )
        result = replay.run(
            scenario=f"seed-{seed}",
            start=reference_end,
            end=replay_end,
            slices=friday_evening_slices(table, timezone_offset_hours=offset),
            scheduled_every=(self.config.retrain_every if SCHEDULED in self.config.arms else None),
            scheduled_window=self.config.scheduled_window,
        )

        metrics: dict[str, ArmMetrics] = {}
        for name in (STATIC, SCHEDULED, ADAPTIVE):
            arm = result.arms.get(name)
            if arm is None or name not in self.config.arms:
                continue
            metrics[name] = score_arm(
                arm,
                seed=seed,
                actual=result.actual,
                predicted=arm.predictions,
                locations=result.locations,
                thresholds=thresholds,
                calibrator=calibrator,
            )

        return SeedRun(
            seed=seed,
            metrics=metrics,
            replay=result,
            rows=result.static.rows,
            thresholds=thresholds,
        )

    def run(self) -> list[SeedRun]:
        """Run every configured seed, in the order they were pre-registered."""
        return [self.run_seed(seed) for seed in self.config.seeds]


def _window(table: FeatureTable, start: datetime, end: datetime) -> FeatureTable:
    """Rows whose sample time falls in ``[start, end)``."""
    mask = np.array([start <= moment < end for moment in table.sample_times], dtype=bool)
    return table.select(mask)

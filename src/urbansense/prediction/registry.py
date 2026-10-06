"""Model registry v1 (SPEC sections 31, 32).

    Every model must have a version. Store: training period, features,
    algorithm, hyperparameters, metrics, dataset version, creation timestamp,
    status.

The point of recording all of it is that a metric without its context is not a
result. "MAE 212" means nothing without knowing which period it was trained on,
which features it saw, which rows were excluded and whether the leakage check
passed -- and six months later nobody will remember.

Two deliberate choices:

**The metadata is tracked in git; the weights are not.** ``models/registry/``
holds JSON, ``models/artifacts/`` is ignored. The metadata *is* the record, and
it is small, reviewable in a diff, and reproducible from. A pickled estimator is
large, binary, and tied to a scikit-learn version.

**Status is recorded, never inferred.** ``champion`` here is a claim about this
model's role, which is what Phase 6's champion/challenger comparison will
read. Nothing in this phase promotes or demotes anything.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urbansense.schemas.enums import ModelStatus

#: Filename pattern for a registry entry.
REGISTRY_SUFFIX = ".json"

#: Filename pattern for a serialized estimator.
ARTIFACT_SUFFIX = ".pkl"


@dataclass(frozen=True)
class ModelVersion:
    """One versioned model and everything needed to interpret its metrics.

    Attributes:
        version: Version name, e.g. ``Traffic-v1.0``.
        problem_type: Problem domain.
        target: Metric predicted.
        horizon_hours: Lead time of the prediction.
        algorithm: Fully qualified estimator.
        algorithm_rationale: Why this algorithm, so the choice is reviewable.
        hyperparameters: Settings used.
        features: Full feature descriptions, including each one's backward reach.
        training_period: Train segment bounds and row count.
        validation_period: Validation segment bounds and row count.
        test_period: Test segment bounds and row count.
        gap_hours: Hours excluded between segments.
        metrics: Evaluation results, per model and slice.
        leakage_checks: Which leakage clauses ran and what they found.
        excluded_rows: What was kept out of training, and why.
        dataset_version: Which generated dataset produced the data.
        interval: How the uncertainty hint was derived, and its limitation.
        congestion: The congestion definition and the fitted per-zone
            thresholds, including which segment they were fitted on.
        calibration: How congestion probabilities were produced and how well
            they held up, with the reliability tables. Recorded because a
            probability without its calibration evidence cannot be judged.
        explainability: Which attribution method ran, whether it is additive,
            and why that method was chosen.
        error_analysis: Where the model is weak, by condition and slice.
        status: Role of this version.
        created_at: When it was trained.
        code_version: Package version that produced it.
        artifact_path: Where the estimator was written, if it was.
        notes: Honest caveats about the result.
    """

    version: str
    problem_type: str
    target: str
    horizon_hours: int
    algorithm: str
    algorithm_rationale: str
    hyperparameters: dict[str, Any]
    features: list[dict[str, Any]]
    training_period: dict[str, Any]
    validation_period: dict[str, Any]
    test_period: dict[str, Any]
    gap_hours: int
    metrics: list[dict[str, Any]]
    leakage_checks: dict[str, Any]
    excluded_rows: dict[str, Any]
    dataset_version: dict[str, Any]
    interval: dict[str, Any]
    congestion: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.CHAMPION
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    code_version: str = ""
    artifact_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "version": self.version,
            "status": self.status.value,
            "problem_type": self.problem_type,
            "target": self.target,
            "horizon_hours": self.horizon_hours,
            "algorithm": self.algorithm,
            "algorithm_rationale": self.algorithm_rationale,
            "hyperparameters": self.hyperparameters,
            "features": self.features,
            "training_period": self.training_period,
            "validation_period": self.validation_period,
            "test_period": self.test_period,
            "gap_hours": self.gap_hours,
            "metrics": self.metrics,
            "leakage_checks": self.leakage_checks,
            "excluded_rows": self.excluded_rows,
            "dataset_version": self.dataset_version,
            "interval": self.interval,
            "congestion": self.congestion,
            "calibration": self.calibration,
            "explainability": self.explainability,
            "error_analysis": self.error_analysis,
            "created_at": self.created_at.isoformat(),
            "code_version": self.code_version,
            "artifact_path": self.artifact_path,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        """Pretty, sorted JSON, so a diff between versions is readable."""
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str) + "\n"


class ModelRegistry:
    """Reads and writes versioned model metadata.

    Attributes:
        root: Registry directory, typically ``models/registry/``.
        artifacts: Artifact directory, typically ``models/artifacts/``.
    """

    def __init__(self, root: Path, artifacts: Path | None = None) -> None:
        """Initialize.

        Args:
            root: Where metadata JSON is written. Tracked in git.
            artifacts: Where estimators are written. Gitignored. Defaults to a
                sibling ``artifacts`` directory.
        """
        self.root = root
        self.artifacts = artifacts if artifacts is not None else root.parent / "artifacts"

    def metadata_path(self, version: str) -> Path:
        """Where a version's metadata lives."""
        return self.root / f"{version}{REGISTRY_SUFFIX}"

    def artifact_path(self, version: str) -> Path:
        """Where a version's estimator lives."""
        return self.artifacts / f"{version}{ARTIFACT_SUFFIX}"

    def save(
        self,
        model_version: ModelVersion,
        *,
        estimator: object | None = None,
    ) -> Path:
        """Write metadata, and the estimator when one is given.

        Args:
            model_version: The metadata to record.
            estimator: Fitted estimator to pickle. Optional, because the metadata
                is the record and is useful on its own.

        Returns:
            The metadata path.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        recorded = model_version
        if estimator is not None:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            target = self.artifact_path(model_version.version)
            with target.open("wb") as handle:
                pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
            recorded = ModelVersion(
                **{
                    **{
                        key: getattr(model_version, key)
                        for key in model_version.__dataclass_fields__
                    },
                    "artifact_path": str(target.as_posix()),
                }
            )

        path = self.metadata_path(recorded.version)
        path.write_text(recorded.to_json(), encoding="utf-8", newline="\n")
        return path

    def set_status(self, version: str, status: ModelStatus) -> Path:
        """Change one version's status in place, leaving its metrics intact.

        Used to archive a superseded champion. The entry is kept rather than
        deleted: knowing which model served, and how well, is the experimental
        record (SPEC sections 31, 32). Only the status field moves.

        Raises:
            FileNotFoundError: if the version is not registered.
        """
        path = self.metadata_path(version)
        if not path.is_file():
            raise FileNotFoundError(f"no registry entry for {version!r} at {path}")
        metadata: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        metadata["status"] = status.value
        path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    def load_metadata(self, version: str) -> dict[str, Any]:
        """Read one version's metadata.

        Raises:
            FileNotFoundError: if the version is not registered.
        """
        path = self.metadata_path(version)
        if not path.is_file():
            raise FileNotFoundError(f"no registry entry for {version!r} at {path}")
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def load_estimator(self, version: str) -> object:
        """Read one version's estimator.

        Raises:
            FileNotFoundError: if the artifact is absent. Artifacts are
                gitignored, so a fresh clone legitimately has metadata without
                weights -- the metadata says how to retrain.
        """
        path = self.artifact_path(version)
        if not path.is_file():
            raise FileNotFoundError(
                f"no artifact for {version!r} at {path}. Artifacts are not tracked "
                "in git; retrain with scripts/train_baseline.py to recreate it."
            )
        with path.open("rb") as handle:
            return pickle.load(handle)

    def versions(self) -> tuple[str, ...]:
        """Registered versions, sorted."""
        if not self.root.is_dir():
            return ()
        return tuple(sorted(path.stem for path in self.root.glob(f"*{REGISTRY_SUFFIX}")))

    def champions(self) -> tuple[str, ...]:
        """Versions currently marked champion."""
        found: list[str] = []
        for version in self.versions():
            metadata = self.load_metadata(version)
            if metadata.get("status") == ModelStatus.CHAMPION.value:
                found.append(version)
        return tuple(found)

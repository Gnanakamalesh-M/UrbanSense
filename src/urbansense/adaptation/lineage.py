"""Champion/challenger registry operations (SPEC sections 31, 32).

    Support: Champion / Challenger / Rejected / Archived
    Current production model: Champion. New candidate: Challenger.

Builds on Phase 6's :class:`~urbansense.prediction.registry.ModelRegistry`
rather than replacing it. Three guarantees live here:

**Exactly one champion.** :func:`promote` archives the incumbent and installs
the new version, and :func:`assert_single_champion` is called after every
mutation. Two champions is not a cosmetic problem -- it means two answers to
"what is serving", and any later code picking between them picks arbitrarily.

**Rejected challengers stay on disk.** A rejected model keeps its entry, its
metrics and the rule outcomes that rejected it. Knowing which updates failed
evaluation, and by how much, is the experimental record; deleting them leaves a
registry that implies every candidate ever raised was promoted.

**The lineage is explicit.** Each version records what it ``supersedes``, the
``drift_episode_id`` that prompted it and the ``candidate_id`` that decided it,
so a model traces back to the evidence rather than to a date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urbansense.prediction.registry import ModelRegistry, ModelVersion
from urbansense.schemas.enums import ModelStatus


class LineageError(RuntimeError):
    """Raised when a registry operation would break the champion invariant."""


def assert_single_champion(registry: ModelRegistry) -> str:
    """Check that exactly one version is champion, and return it.

    Raises:
        LineageError: if there are none or several. Called after every mutation
            rather than only in the tests, so a broken registry fails where it
            was broken instead of somewhere downstream.
    """
    champions = registry.champions()
    if len(champions) == 1:
        return champions[0]
    if not champions:
        raise LineageError(
            "no champion in the registry: nothing would serve predictions. "
            "A promotion must have failed partway through."
        )
    raise LineageError(
        f"{len(champions)} champions in the registry ({', '.join(champions)}); "
        "exactly one version may serve at a time (SPEC section 32)"
    )


def promote(
    registry: ModelRegistry,
    version: str,
    *,
    incumbent: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Make ``version`` the champion and archive **every** other champion.

    "Make this the champion" means demoting whatever else holds the title, so
    that is what this does -- it does not archive only a named incumbent and
    hope the rest of the registry agrees. An earlier version took the incumbent
    as the single source of truth and archived just that one; pointed at a
    registry that already held a stale champion from a previous run, it produced
    two champions and then failed its own post-condition several steps later,
    with an error naming versions the caller had never mentioned.

    Order matters: the incumbents are archived **first**, so a failure between
    the steps leaves no champion rather than two. No champion is a loud,
    immediately detected state; two champions is a quiet one that corrupts every
    later decision.

    Args:
        registry: The registry to mutate.
        version: The version to promote. Must already be registered.
        incumbent: The expected current champion, when the caller knows it. Used
            only to order the returned list sensibly; every champion is archived
            either way.

    Returns:
        The promoted version and every version archived to make room for it.

    Raises:
        LineageError: if the version is not registered, or if the registry does
            not end up with exactly one champion.
    """
    if version not in registry.versions():
        raise LineageError(f"cannot promote {version!r}: it is not in the registry")

    outgoing = [other for other in registry.champions() if other != version]
    if incumbent is not None and incumbent in outgoing:
        # Named first in the return value, since that is the handover a caller
        # is usually reporting.
        outgoing = [incumbent, *(other for other in outgoing if other != incumbent)]

    for other in outgoing:
        registry.set_status(other, ModelStatus.ARCHIVED)

    registry.set_status(version, ModelStatus.CHAMPION)
    assert_single_champion(registry)
    return version, tuple(outgoing)


def replay_run_root(root: Path, *, scenario: str, now: datetime | None = None) -> Path:
    """A fresh directory for one replay's registry.

    Every run gets its own, so a second run of a scenario -- or the other
    scenario -- cannot inherit a champion from the first. That inheritance was
    a real failure, not a hypothetical one: a second run re-seeded its starting
    champion into a registry that still held the previous run's, and the next
    promotion raised ``2 champions`` naming a version the run had not touched
    yet.

    Isolation by fresh directory rather than by deleting the previous run,
    because a replay's registry is evidence about a run and the project does not
    overwrite evidence. The microsecond stamp keeps two runs in the same second
    apart.
    """
    stamp = (now if now is not None else datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%f")
    return root / scenario / f"run-{stamp}"


def reject(registry: ModelRegistry, version: str) -> str:
    """Mark a challenger rejected, keeping its entry and metrics on disk.

    Raises:
        LineageError: if the version is not registered, or if rejecting it
            would leave the registry without a champion -- which would mean the
            champion itself was being rejected.
    """
    if version not in registry.versions():
        raise LineageError(f"cannot reject {version!r}: it is not in the registry")
    registry.set_status(version, ModelStatus.REJECTED)
    assert_single_champion(registry)
    return version


@dataclass(frozen=True)
class LineageEntry:
    """One version's place in the lineage, for a report.

    Attributes:
        version: The version name.
        status: Its current role.
        created_at: When it was trained.
        supersedes: The version it replaced, if any.
        candidate_id: The candidate update that decided it.
        drift_episode_id: The drift episode that prompted it.
        training_rows: Rows it trained on.
        mae: Its headline error, when recorded.
    """

    version: str
    status: str
    created_at: str
    supersedes: str | None = None
    candidate_id: str | None = None
    drift_episode_id: str | None = None
    training_rows: int | None = None
    mae: float | None = None

    def describe(self) -> str:
        """One line."""
        parts = [f"{self.version:<28} {self.status:<11} {self.created_at[:19]}"]
        if self.supersedes:
            parts.append(f"supersedes {self.supersedes}")
        if self.drift_episode_id:
            parts.append(f"episode {self.drift_episode_id}")
        if self.mae is not None:
            parts.append(f"MAE {self.mae:,.1f}")
        return "  ".join(parts)


def lineage(registry: ModelRegistry) -> tuple[LineageEntry, ...]:
    """Every registered version with its lineage fields, oldest first.

    Reads the metadata rather than any in-memory state, so the lineage a report
    prints is the lineage on disk.
    """
    entries: list[LineageEntry] = []
    for version in registry.versions():
        metadata: dict[str, Any] = registry.load_metadata(version)
        adaptation = metadata.get("adaptation")
        adaptation = adaptation if isinstance(adaptation, dict) else {}
        training = metadata.get("training_period")
        training = training if isinstance(training, dict) else {}
        metrics = metadata.get("metrics")
        headline: float | None = None
        if isinstance(metrics, list):
            for entry in metrics:
                if isinstance(entry, dict) and isinstance(entry.get("overall"), dict):
                    value = entry["overall"].get("mae")
                    if isinstance(value, int | float):
                        headline = float(value)
                        break
        entries.append(
            LineageEntry(
                version=version,
                status=str(metadata.get("status", "unknown")),
                created_at=str(metadata.get("created_at", "")),
                supersedes=adaptation.get("supersedes"),
                candidate_id=adaptation.get("candidate_id"),
                drift_episode_id=adaptation.get("drift_episode_id"),
                training_rows=training.get("rows"),
                mae=headline,
            )
        )
    entries.sort(key=lambda item: item.created_at)
    return tuple(entries)


def render_lineage(registry: ModelRegistry) -> str:
    """A table of the registry, for a replay report."""
    entries = lineage(registry)
    if not entries:
        return "registry is empty"
    champions = [entry.version for entry in entries if entry.status == ModelStatus.CHAMPION.value]
    lines = [f"{len(entries)} version(s) in the registry:"]
    lines.extend(f"  {entry.describe()}" for entry in entries)
    lines.append(
        f"  champion: {', '.join(champions) if champions else 'NONE'}"
        + ("" if len(champions) == 1 else "  <-- the invariant is broken")
    )
    return "\n".join(lines)


def register_challenger(
    registry: ModelRegistry,
    *,
    version: str,
    base: ModelVersion,
    candidate_id: str,
    drift_episode_id: str | None,
    supersedes: str | None,
    training_rows: int,
    training_start: str,
    training_end: str,
    training_window: str,
    metrics: list[dict[str, Any]],
    leakage: dict[str, Any],
    decision: dict[str, Any],
    status: ModelStatus = ModelStatus.CHALLENGER,
    artifact_dir: Path | None = None,
    estimator: object | None = None,
) -> Path:
    """Write a challenger's registry entry, with its full lineage.

    Takes the champion's entry as ``base`` so the §31 fields that do not change
    between versions -- algorithm, hyperparameters, feature descriptions,
    dataset version -- are carried over rather than re-derived and allowed to
    drift out of step with the model that is actually serving.
    """
    entry = ModelVersion(
        version=version,
        problem_type=base.problem_type,
        target=base.target,
        horizon_hours=base.horizon_hours,
        algorithm=base.algorithm,
        algorithm_rationale=base.algorithm_rationale,
        hyperparameters=dict(base.hyperparameters),
        features=list(base.features),
        training_period={
            "start": training_start,
            "end": training_end,
            "rows": training_rows,
            "window": training_window,
        },
        validation_period=dict(base.validation_period),
        test_period=dict(base.test_period),
        gap_hours=base.gap_hours,
        metrics=metrics,
        leakage_checks=leakage,
        excluded_rows=dict(base.excluded_rows),
        dataset_version=dict(base.dataset_version),
        interval=dict(base.interval),
        congestion=dict(base.congestion),
        calibration={},
        explainability=dict(base.explainability),
        error_analysis={},
        status=status,
        created_at=datetime.now(UTC),
        code_version=base.code_version,
        notes=[
            "Trained by the adaptive loop in response to detected drift, not on a "
            "schedule and not because new data arrived.",
            f"Training window: {training_window}. The window was one of several "
            "tried; the promotion rules chose between them.",
            "Calibration and error analysis are not recorded for a challenger: "
            "both are Phase 6 artifacts of a serving model, and recording them "
            "for a model that may be rejected would imply it served.",
        ],
    )
    path = registry.save(entry, estimator=estimator)

    # The adaptation block is what makes a version traceable to its evidence.
    # It is merged into the written file rather than added to ModelVersion,
    # because that schema is deliberately fixed -- widening it every phase would
    # make a v1.0 entry unreadable by the code that wrote it.
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["adaptation"] = {
        "candidate_id": candidate_id,
        "drift_episode_id": drift_episode_id,
        "supersedes": supersedes,
        "training_window": training_window,
        "decision": decision,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path

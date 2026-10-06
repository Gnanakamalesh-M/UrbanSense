"""Loading the pre-registered experiment configuration (SPEC sections 38, 39).

    Create reproducible experiments. Report results honestly.
    Do not cherry-pick favorable results.

The configuration is a committed file, read-only at run time, and copied verbatim
into every report. "Pre-registered" is doing real work here: a seed list or a
retrain cadence chosen after seeing which arm won would make the comparison
worthless, and the only way to show that did not happen is for the parameters to
be in version control with a timestamp from before the first run.

The loader is strict like every other in the project. A typo in a key is an
error rather than a silent default, because an experiment that quietly ran with
three seeds instead of five would still produce a plausible-looking table.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The pre-registered configuration shipped with the repository.
DEFAULT_EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "experiments" / "static_vs_adaptive.yaml"
)

#: Arm names, in the order they are reported.
STATIC = "static"
SCHEDULED = "scheduled"
ADAPTIVE = "adaptive"
ARM_ORDER: tuple[str, ...] = (STATIC, SCHEDULED, ADAPTIVE)


class ExperimentConfigError(RuntimeError):
    """Raised when the experiment configuration is missing or malformed."""


class ExperimentConfig(BaseModel):
    """Everything the three-arm comparison needs, fixed in advance.

    Attributes:
        pre_registered: The date the parameters were committed. Carried into the
            report so a reader can check it against the git history rather than
            taking the claim on trust.
        description: Why the experiment exists.
        arms: Arms to run.
        seeds: Generator seeds. More than one because a single dataset cannot
            distinguish a real difference between arms from one lucky draw.
        retrain_every_days: The scheduled arm's calendar.
        scheduled_window: Which training window the scheduled arm deploys.
        train_days: Span the frozen champion trains on.
        reference_days: Out-of-sample reference span for the drift monitor.
        replay_days: Days to replay after the reference window, or ``None`` for
            the rest of the stream.
        metrics: Metric families to compute.
        adaptation_config: Path to the Phase 7 rules this run obeys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pre_registered: str = Field(min_length=4)
    description: str = ""
    arms: tuple[str, ...] = ARM_ORDER
    seeds: tuple[int, ...] = ()
    retrain_every_days: int = Field(default=28, gt=0)
    scheduled_window: str = Field(default="expanding", min_length=1)
    train_days: int = Field(default=75, gt=0)
    reference_days: int = Field(default=60, gt=0)
    replay_days: int | None = Field(default=None, gt=0)
    metrics: tuple[str, ...] = ()
    adaptation_config: Path = Path("configs/adaptation/traffic.yaml")

    @property
    def retrain_every(self) -> timedelta:
        """The scheduled cadence as a duration."""
        return timedelta(days=self.retrain_every_days)

    @model_validator(mode="after")
    def _is_runnable(self) -> Self:
        """Reject a configuration that cannot produce a comparison.

        Each check is a way the experiment could run to completion and report a
        table that means nothing.
        """
        if not self.seeds:
            raise ValueError(
                "experiment: 'seeds' must list at least one seed; a run with none "
                "would report a table with no data behind it"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                f"experiment: duplicate seeds {sorted(self.seeds)}. A repeated seed is "
                "the same dataset counted twice, which narrows the reported spread "
                "without adding evidence."
            )
        unknown = [arm for arm in self.arms if arm not in ARM_ORDER]
        if unknown:
            raise ValueError(f"experiment: unknown arm(s) {unknown}; expected {list(ARM_ORDER)}")
        if len(self.arms) < 2:
            raise ValueError("experiment: at least two arms are needed to compare anything")
        return self

    def as_dict(self) -> dict[str, object]:
        """Serializable form, copied verbatim into the report."""
        return {
            "pre_registered": self.pre_registered,
            "description": self.description,
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "retrain_every_days": self.retrain_every_days,
            "scheduled_window": self.scheduled_window,
            "train_days": self.train_days,
            "reference_days": self.reference_days,
            "replay_days": self.replay_days,
            "metrics": list(self.metrics),
            "adaptation_config": str(self.adaptation_config.as_posix()),
        }

    def describe(self) -> str:
        """Human-readable summary of what was fixed in advance."""
        return (
            f"pre-registered {self.pre_registered}: {len(self.seeds)} seed(s), "
            f"arms {', '.join(self.arms)}, scheduled retrain every "
            f"{self.retrain_every_days}d on the {self.scheduled_window} window, "
            f"train {self.train_days}d + reference {self.reference_days}d, "
            f"rules from {self.adaptation_config.as_posix()}"
        )


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    """Load and validate the pre-registered configuration.

    Raises:
        ExperimentConfigError: if the file is absent, unparseable, not a mapping,
            or declares something invalid.
    """
    target = path if path is not None else DEFAULT_EXPERIMENT_PATH

    if not target.is_file():
        raise ExperimentConfigError(f"experiment configuration not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"{target} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExperimentConfigError(
            f"{target} must contain a mapping at the top level, got "
            f"{type(raw).__name__ if raw is not None else 'nothing'}"
        )

    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ExperimentConfigError(f"invalid experiment configuration in {target}: {exc}") from exc

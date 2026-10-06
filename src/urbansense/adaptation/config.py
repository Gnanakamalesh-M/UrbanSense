"""Loading the adaptation rules (SPEC sections 30, 32).

    Deploy the new model only if it satisfies predefined validation criteria.

The rules live in YAML so "predefined" is literally true: they are a reviewable
artifact with a timestamp, not a constant someone can nudge after seeing a
challenger's score. The loader is strict like every other in the project -- an
unknown key or a malformed block is an error, never a silent default, because a
promotion rule that quietly failed to load would let every challenger through.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbansense.drift.detectors import DriftKind

#: Default rules shipped with the repository.
DEFAULT_ADAPTATION_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "adaptation" / "traffic.yaml"
)


class AdaptationConfigError(RuntimeError):
    """Raised when the adaptation configuration is missing or malformed."""


class TrainingWindow(BaseModel):
    """One window a challenger may be trained on.

    Attributes:
        name: Identifier, used in the version name and the decision record.
        days: How far back from the decision time to train. ``None`` means
            everything available, which measured best on this dataset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    days: int | None = Field(default=None, gt=0)

    @property
    def span(self) -> timedelta | None:
        """The window as a duration, or ``None`` for expanding."""
        return None if self.days is None else timedelta(days=self.days)

    def describe(self) -> str:
        """Human-readable summary."""
        return f"{self.name} ({'all available data' if self.days is None else f'{self.days}d'})"


class MonitoringRules(BaseModel):
    """How drift is watched.

    Attributes:
        step_days: Replay step between evaluations.
        window_days: How far back each evaluation looks.
        watched_features: Features compared distributionally.
        sigma_multiplier: Standard deviations above the reference mean.
        min_reference_windows: Reference windows needed to establish a gate.
        cooldown_days: Minimum gap between candidates from one episode.
        candidate_kinds: Drift families allowed to raise a candidate. The others
            are still detected and logged as leading indicators.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_days: int = Field(default=7, gt=0)
    window_days: int = Field(default=7, gt=0)
    watched_features: tuple[str, ...] = ("roll_mean_24h",)
    sigma_multiplier: float = Field(default=3.0, gt=0.0)
    min_reference_windows: int = Field(default=4, gt=0)
    cooldown_days: int = Field(default=28, ge=0)
    candidate_kinds: tuple[DriftKind, ...] = (DriftKind.PREDICTION,)

    @property
    def step(self) -> timedelta:
        """Replay step."""
        return timedelta(days=self.step_days)

    @property
    def window(self) -> timedelta:
        """Monitoring window."""
        return timedelta(days=self.window_days)

    @property
    def cooldown(self) -> timedelta:
        """Candidate cooldown."""
        return timedelta(days=self.cooldown_days)

    @property
    def required_kinds(self) -> frozenset[DriftKind]:
        """Families allowed to raise a candidate, as a set."""
        return frozenset(self.candidate_kinds)


class CandidateRules(BaseModel):
    """How challengers are trained and judged.

    Attributes:
        training_windows: Windows to try. Each produces one challenger.
        gap_hours: Hours excluded between training data and the evaluation
            window, so a training row and an evaluation row cannot share an
            input.
        evaluation_days: Length of the evaluation window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    training_windows: tuple[TrainingWindow, ...] = ()
    gap_hours: int = Field(default=168, ge=0)
    evaluation_days: int = Field(default=28, gt=0)

    @property
    def gap(self) -> timedelta:
        """The gap as a duration."""
        return timedelta(hours=self.gap_hours)

    @property
    def evaluation_span(self) -> timedelta:
        """The evaluation window as a duration."""
        return timedelta(days=self.evaluation_days)

    @model_validator(mode="after")
    def _windows_are_named_once(self) -> Self:
        """Duplicate window names would collide in the version naming."""
        if not self.training_windows:
            raise ValueError("candidates: 'training_windows' must list at least one window")
        names = [window.name for window in self.training_windows]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"candidates: duplicate training window names {sorted(duplicates)}")
        return self


class PromotionRules(BaseModel):
    """What a challenger must satisfy to be promoted.

    Attributes:
        min_relative_mae_improvement: Relative MAE improvement required.
        max_slice_regression: How much worse a protected slice may get.
        protected_slices: Slices that may not regress materially.
        min_slice_rows: Rows a protected slice needs before it may block a
            promotion. Below this it is reported as too thin to judge.
        min_evaluation_rows: Rows the evaluation window needs for a decision.
        require_leakage_clean: Whether the leakage probe must pass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_relative_mae_improvement: float = Field(default=0.03, ge=0.0, lt=1.0)
    max_slice_regression: float = Field(default=0.05, ge=0.0)
    protected_slices: tuple[str, ...] = ()
    min_slice_rows: int = Field(default=8, ge=1)
    min_evaluation_rows: int = Field(default=200, gt=0)
    require_leakage_clean: bool = True

    def describe(self) -> str:
        """Human-readable summary of the criteria."""
        return (
            f"promote only if: MAE improves by at least "
            f"{self.min_relative_mae_improvement:.1%}; no protected slice worse by "
            f"more than {self.max_slice_regression:.1%} "
            f"({', '.join(self.protected_slices) or 'none declared'}, needing "
            f"{self.min_slice_rows}+ rows to count); at least "
            f"{self.min_evaluation_rows} evaluation rows"
            + ("; leakage probe clean" if self.require_leakage_clean else "")
        )


class AdaptationConfig(BaseModel):
    """The whole adaptation rule set.

    Attributes:
        monitoring: Drift monitoring parameters.
        candidates: Challenger training parameters.
        promotion: Promotion criteria.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    monitoring: MonitoringRules = MonitoringRules()
    candidates: CandidateRules
    promotion: PromotionRules = PromotionRules()


def load_adaptation_config(path: Path | None = None) -> AdaptationConfig:
    """Load and validate the adaptation rules.

    Raises:
        AdaptationConfigError: if the file is absent, unparseable, not a
            mapping, or declares something invalid.
    """
    target = path if path is not None else DEFAULT_ADAPTATION_PATH

    if not target.is_file():
        raise AdaptationConfigError(f"adaptation configuration not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AdaptationConfigError(f"{target} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise AdaptationConfigError(
            f"{target} must contain a mapping at the top level, got "
            f"{type(raw).__name__ if raw is not None else 'nothing'}"
        )

    try:
        return AdaptationConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError and friends
        raise AdaptationConfigError(f"invalid adaptation configuration in {target}: {exc}") from exc

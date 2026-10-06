"""Which sources fed a prediction, and how fresh they were (SPEC section 26).

    The system should know which sources contributed to a prediction.
    Also display data freshness.

**What the percentages mean, precisely.** They are shares of the *observations
the model's declared features actually read* for this prediction -- a data
provenance breakdown. They are **not** shares of influence: a single rainfall
reading could move a prediction more than fifty traffic lags and still be 2% of
the input count. Section 27's explanation layer answers the influence question,
this one answers "where did the inputs come from", and conflating the two would
let a reader credit a source with an effect it never had. Every rendering says
which question it is answering.

**The demo feed is historical only, so the honest answer is 100% historical.**
SPEC section 26's illustrative example shows a real-time and a document share;
this dataset has neither, and printing a fabricated 31% real-time to match the
example would be inventing evidence. The multi-source path is real code and is
covered by a hand-built test, because the shipped data would otherwise leave it
unexercised.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from urbansense.features.specs import FeatureSet
from urbansense.schemas.enums import Recency, SourceType
from urbansense.schemas.observation import Observation

#: Freshness bands, in hours, used to describe the newest reading behind a
#: prediction. Mirrors the Recency bands Phase 2 assigns to arrivals.
FRESH_HOURS = 2
RECENT_HOURS = 24


@dataclass(frozen=True)
class SourceShare:
    """One source's share of the observations behind a prediction.

    Attributes:
        source_id: The reporting source.
        source_type: Which stream it arrived on.
        observations: Readings this source contributed to the inputs.
        share: Those readings as a fraction of all readings used.
        newest: Most recent event time among them.
        oldest: Earliest event time among them.
    """

    source_id: str
    source_type: SourceType
    observations: int
    share: float
    newest: datetime | None
    oldest: datetime | None

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "observations": self.observations,
            "share": round(self.share, 4),
            "newest_event_time": self.newest.isoformat() if self.newest else None,
            "oldest_event_time": self.oldest.isoformat() if self.oldest else None,
        }


@dataclass(frozen=True)
class SourceBreakdown:
    """Where one prediction's inputs came from, and how old they are.

    Attributes:
        shares: Per-source shares, largest first.
        by_type: Share per source type, which is the headline figure.
        observations_used: Readings the declared features reached.
        newest_event_time: Most recent reading behind the prediction.
        prediction_time: The sample time ``t`` freshness is measured against.
        missing_inputs: Feature windows that found no data at all. Reported
            because a prediction resting on fewer inputs is a weaker one, and
            an absent window is invisible in a share table otherwise.
    """

    shares: tuple[SourceShare, ...]
    by_type: dict[SourceType, float]
    observations_used: int
    newest_event_time: datetime | None
    prediction_time: datetime
    missing_inputs: tuple[str, ...] = ()

    @property
    def freshness(self) -> timedelta | None:
        """Age of the newest reading at prediction time."""
        if self.newest_event_time is None:
            return None
        return self.prediction_time - self.newest_event_time

    @property
    def recency(self) -> Recency:
        """Freshness as a band, matching Phase 2's vocabulary."""
        age = self.freshness
        if age is None:
            return Recency.HISTORICAL
        hours = age.total_seconds() / 3600.0
        if hours <= FRESH_HOURS:
            return Recency.CURRENT
        if hours <= RECENT_HOURS:
            return Recency.RECENT
        return Recency.HISTORICAL

    @property
    def is_single_source_type(self) -> bool:
        """Whether every input came from one stream."""
        return len([share for share in self.by_type.values() if share > 0]) == 1

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        age = self.freshness
        return {
            "question_answered": (
                "which sources the model's inputs came from, by observation count "
                "-- not how much each source influenced the prediction"
            ),
            "observations_used": self.observations_used,
            "by_type": {
                source_type.value: round(share, 4)
                for source_type, share in sorted(
                    self.by_type.items(), key=lambda pair: pair[0].value
                )
            },
            "by_source": [share.as_dict() for share in self.shares],
            "freshness_hours": (round(age.total_seconds() / 3600.0, 2) if age else None),
            "recency": self.recency.value,
            "newest_event_time": (
                self.newest_event_time.isoformat() if self.newest_event_time else None
            ),
            "missing_inputs": list(self.missing_inputs),
        }

    def describe(self) -> str:
        """Human-readable summary, in the spec's shape."""
        lines = ["contributing sources (share of input observations, not of influence):"]
        for source_type, share in sorted(
            self.by_type.items(), key=lambda pair: (-pair[1], pair[0].value)
        ):
            lines.append(f"  {source_type.value:<14} {share:>6.1%}")
        if self.is_single_source_type and self.shares:
            only = next(iter(self.by_type))
            lines.append(
                f"  this feed carries {only.value} data only; there is no real-time "
                "or document source in this dataset to report a share for"
            )
        age = self.freshness
        lines.append(
            "  freshness: newest reading "
            + (
                f"{age.total_seconds() / 3600.0:.1f}h before prediction time ({self.recency.value})"
                if age is not None
                else "unavailable"
            )
        )
        if self.missing_inputs:
            lines.append(
                f"  {len(self.missing_inputs)} feature window(s) found no data: "
                f"{', '.join(self.missing_inputs[:4])}"
                + (" ..." if len(self.missing_inputs) > 4 else "")
            )
        return "\n".join(lines)


class SourceLedger:
    """Indexes observations so a prediction's inputs can be traced to sources.

    Built once per run and reused across predictions: the index is over the
    whole unified layer, and rebuilding it per prediction would make a day's
    forecast quadratic in the dataset.
    """

    def __init__(self, observations: Iterable[Observation]) -> None:
        """Index observations by location, metric and event time.

        Args:
            observations: The unified layer. Readings with no value are skipped
                -- a hole contributed nothing and crediting a source for it
                would overstate what that source supplied.
        """
        self._by_key: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for observation in observations:
            if observation.value is None:
                continue
            self._by_key[(observation.location_id, observation.metric)].append(observation)
        for group in self._by_key.values():
            group.sort(key=lambda item: item.event_time)

    def __len__(self) -> int:
        """Number of indexed series."""
        return len(self._by_key)

    def _in_window(
        self, location_id: str, metric: str, earliest: datetime, latest: datetime
    ) -> list[Observation]:
        """Observations of one series within an inclusive time window."""
        group = self._by_key.get((location_id, metric), [])
        return [item for item in group if earliest <= item.event_time <= latest]

    def attribute(
        self,
        feature_set: FeatureSet,
        *,
        location_id: str,
        sample_time: datetime,
    ) -> SourceBreakdown:
        """Tally the sources behind one prediction's inputs.

        Walks each feature's **declared** reading window -- the same offsets the
        leakage probe verifies behaviourally -- so a source can only be credited
        for data the model was permitted to see. Nothing after ``sample_time``
        is reachable, because :meth:`FeatureSpec.latest` never exceeds it.
        """
        counted: dict[str, list[Observation]] = defaultdict(list)
        seen: set[str] = set()
        missing: list[str] = []

        for spec in feature_set.specs:
            if spec.metric is None:
                # A calendar or spatial feature reads no observations at all,
                # so it has no source to attribute.
                continue
            target_zone = spec.location if spec.location is not None else location_id
            found = self._in_window(
                target_zone, spec.metric, spec.earliest(sample_time), spec.latest(sample_time)
            )
            if not found:
                missing.append(spec.name)
                continue
            for observation in found:
                # Counted once per prediction: several features reading the same
                # hour is one piece of evidence, not three.
                if observation.observation_id in seen:
                    continue
                seen.add(observation.observation_id)
                counted[observation.source_id].append(observation)

        total = sum(len(items) for items in counted.values())
        shares: list[SourceShare] = []
        by_type: dict[SourceType, float] = {}
        for source_id, items in counted.items():
            share = len(items) / total if total else 0.0
            shares.append(
                SourceShare(
                    source_id=source_id,
                    source_type=items[0].source_type,
                    observations=len(items),
                    share=share,
                    newest=max(item.event_time for item in items),
                    oldest=min(item.event_time for item in items),
                )
            )
            by_type[items[0].source_type] = by_type.get(items[0].source_type, 0.0) + share

        shares.sort(key=lambda item: (-item.observations, item.source_id))
        newest = max((share.newest for share in shares if share.newest), default=None)
        return SourceBreakdown(
            shares=tuple(shares),
            by_type=by_type,
            observations_used=total,
            newest_event_time=newest,
            prediction_time=sample_time,
            missing_inputs=tuple(missing),
        )

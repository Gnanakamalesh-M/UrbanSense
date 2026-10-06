"""Drift events and episodes, stored append-only (SPEC section 29).

    When significant drift is detected:
    Drift detected -> Investigate -> Candidate model update

An event is a dated observation about the world, so the log is append-only for
the same reason the quarantine queue and the outcome log are: rewriting what was
believed on 2026-03-12 destroys the evidence for why a model was retrained.

**Episodes exist because drift does not end when it is noticed.** On the real
data PSI stays above its gate for every one of the fifteen weeks after the
planted shift. Fifteen events is accurate; fifteen retrains is not. Consecutive
firings therefore collapse into one :class:`DriftEpisode`, which is what a
candidate update attaches to, and a cooldown governs whether a continuing
episode may raise a second candidate.

**Not every firing qualifies to raise a candidate.** Measured on the demo data
against a drift-free copy of the same dataset -- same seed, same everything, the
level shift simply switched off -- the three families behave very differently:

====================  ===============  ==================
family                drifted stream   drift-free stream
====================  ===============  ==================
feature drift         16 firings       15 firings
performance drift     16 firings       10 firings
**prediction drift**  15 firings       **0 firings**
====================  ===============  ==================

Feature and performance drift fire on both streams, because this generator has
annual seasonality: a champion trained on autumn genuinely sees different inputs
and genuinely gets worse by spring, with no change in the city at all. Only
**prediction drift** separates the two, and the reason is instructive -- the
model's *output* distribution moves when the input-output relationship changes
in a way the model actually responds to, while seasonal variation it was trained
to handle moves its inputs without moving its answers.

So :attr:`EpisodeTracker.required_kinds` gates candidate creation on the
discriminating family, and the others are logged as leading indicators. They are
genuinely useful as such -- performance drift fires first, four days after the
change against prediction drift's eleven -- but acting on them alone would
retrain the model every spring.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from urbansense.drift.detectors import DriftKind, DriftSignal

#: Default directory for drift logs.
DEFAULT_DRIFT_DIR = Path("reports") / "drift"


@dataclass(frozen=True)
class DriftEvent:
    """One detector's verdict at one point in the replay.

    Attributes:
        detected_at: The replay instant this was evaluated at. Not the wall
            clock: an event belongs to the stream's timeline, so a re-run places
            it identically.
        window_start: First sample time in the window judged.
        window_end: Last sample time in the window judged.
        signals: Every detector's verdict for this window, firing or not. All of
            them are kept: knowing that feature drift was quiet while
            performance drift fired is what distinguishes a changed city from a
            broken model.
        episode_id: The episode this event belongs to, when it fired.
    """

    detected_at: datetime
    window_start: datetime
    window_end: datetime
    signals: tuple[DriftSignal, ...]
    episode_id: str | None = None

    @property
    def fired(self) -> tuple[DriftSignal, ...]:
        """Signals that crossed their gate."""
        return tuple(signal for signal in self.signals if signal.fired)

    @property
    def is_drift(self) -> bool:
        """Whether anything fired."""
        return bool(self.fired)

    @property
    def kinds(self) -> tuple[DriftKind, ...]:
        """Families that fired, in SPEC order."""
        order = (DriftKind.FEATURE, DriftKind.PREDICTION, DriftKind.PERFORMANCE)
        firing = {signal.kind for signal in self.fired}
        return tuple(kind for kind in order if kind in firing)

    def qualifies(self, required: frozenset[DriftKind]) -> bool:
        """Whether this firing includes a family allowed to raise a candidate.

        An empty requirement accepts any firing. Separate from
        :attr:`is_drift` on purpose: a feature-drift-only firing is real drift
        and is logged as such, it is simply not on its own grounds to retrain.
        """
        if not self.is_drift:
            return False
        return True if not required else bool(set(self.kinds) & required)

    @property
    def inconclusive(self) -> tuple[DriftSignal, ...]:
        """Signals that could not reach a verdict."""
        return tuple(signal for signal in self.signals if not signal.conclusive)

    @property
    def strongest(self) -> DriftSignal | None:
        """The firing signal furthest past its gate, or ``None``."""
        firing = [s for s in self.fired if s.magnitude == s.magnitude]
        return max(firing, key=lambda s: s.magnitude) if firing else None

    def as_dict(self) -> dict[str, object]:
        """Serializable form, carrying every field SPEC section 29 names."""
        strongest = self.strongest
        return {
            "detected_at": self.detected_at.isoformat(),
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
            },
            "is_drift": self.is_drift,
            "kinds": [kind.value for kind in self.kinds],
            "episode_id": self.episode_id,
            "magnitude": (None if strongest is None else round(strongest.magnitude, 4)),
            "strongest_subject": None if strongest is None else strongest.subject,
            "signals": [signal.as_dict() for signal in self.signals],
            "inconclusive": [signal.subject for signal in self.inconclusive],
        }

    def describe(self) -> str:
        """Human-readable summary."""
        if not self.is_drift:
            quiet = "stable"
            if self.inconclusive:
                quiet = f"stable ({len(self.inconclusive)} detector(s) inconclusive)"
            return f"{self.detected_at:%Y-%m-%d}  {quiet}"
        parts = ", ".join(
            f"{signal.kind.value}/{signal.subject} {signal.statistic:.3f}"
            f" (gate {signal.threshold:.3f})"
            for signal in self.fired[:4]
        )
        extra = "" if len(self.fired) <= 4 else f" +{len(self.fired) - 4} more"
        return f"{self.detected_at:%Y-%m-%d}  DRIFT  {parts}{extra}"


@dataclass
class DriftEpisode:
    """A run of consecutive firings treated as one event in the world.

    Attributes:
        episode_id: Stable identifier, derived from the onset instant.
        onset: When the episode first fired.
        last_seen: The most recent firing in it.
        events: How many evaluations fired within it.
        kinds: Every family that fired at any point in the episode.
        peak_magnitude: Furthest past the gate any signal reached.
        candidates_raised: Candidate updates this episode has produced.
        closed_at: When the episode stopped firing, if it has.
    """

    episode_id: str
    onset: datetime
    last_seen: datetime
    events: int = 1
    kinds: tuple[DriftKind, ...] = ()
    peak_magnitude: float = 0.0
    candidates_raised: int = 0
    closed_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        """Whether the episode is still firing."""
        return self.closed_at is None

    @property
    def duration(self) -> timedelta:
        """How long the episode has been running."""
        return self.last_seen - self.onset

    def as_dict(self) -> dict[str, object]:
        """Serializable form."""
        return {
            "episode_id": self.episode_id,
            "onset": self.onset.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "events": self.events,
            "kinds": [kind.value for kind in self.kinds],
            "peak_magnitude": round(self.peak_magnitude, 4),
            "candidates_raised": self.candidates_raised,
            "duration_days": round(self.duration.total_seconds() / 86400.0, 2),
        }

    def describe(self) -> str:
        """One line."""
        state = "open" if self.is_open else f"closed {self.closed_at:%Y-%m-%d}"
        return (
            f"{self.episode_id}: onset {self.onset:%Y-%m-%d}, {self.events} event(s) "
            f"over {self.duration.days}d, peak {self.peak_magnitude:.2f}x gate, "
            f"{self.candidates_raised} candidate(s), {state}"
        )


class DriftStore:
    """Append-only JSONL log of drift events and episodes.

    Attributes:
        root: Directory the logs live in.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialize.

        Args:
            root: Directory to write into. Defaults to ``reports/drift/``.
        """
        self.root = root if root is not None else DEFAULT_DRIFT_DIR

    @property
    def events_path(self) -> Path:
        """The event log."""
        return self.root / "events.jsonl"

    @property
    def episodes_path(self) -> Path:
        """The episode log."""
        return self.root / "episodes.jsonl"

    def append_events(self, events: Iterable[DriftEvent]) -> int:
        """Append events to the log. Returns the number written."""
        rows = list(events)
        if not rows:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            for event in rows:
                handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
        return len(rows)

    def append_episodes(self, episodes: Iterable[DriftEpisode]) -> int:
        """Append episode summaries. Returns the number written."""
        rows = list(episodes)
        if not rows:
            return 0
        self.root.mkdir(parents=True, exist_ok=True)
        with self.episodes_path.open("a", encoding="utf-8", newline="\n") as handle:
            for episode in rows:
                handle.write(json.dumps(episode.as_dict(), sort_keys=True) + "\n")
        return len(rows)

    def read_events(self) -> tuple[dict[str, object], ...]:
        """Read the event log back, in the order written."""
        return self._read(self.events_path)

    def read_episodes(self) -> tuple[dict[str, object], ...]:
        """Read the episode log back, in the order written."""
        return self._read(self.episodes_path)

    @staticmethod
    def _read(path: Path) -> tuple[dict[str, object], ...]:
        """Read one JSONL file."""
        if not path.is_file():
            return ()
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return tuple(rows)


@dataclass
class EpisodeTracker:
    """Groups consecutive firings into episodes and applies the cooldown.

    Two rules, both aimed at "do not retrain blindly":

    - A firing while an episode is already open **continues** it rather than
      starting a new one.
    - A continuing episode may raise a second candidate only after
      ``cooldown`` has elapsed since the last one. Without that, the fifteen
      consecutive firings this dataset produces become fifteen retrains.

    Attributes:
        cooldown: Minimum gap between candidates from one episode.
        required_kinds: Families that must fire for an episode to open and a
            candidate to be raised. Defaults to prediction drift, the only one
            that distinguished a real shift from seasonal degradation on this
            dataset. An empty set accepts any firing.
        episodes: Every episode seen, in onset order.
    """

    cooldown: timedelta = timedelta(days=28)
    required_kinds: frozenset[DriftKind] = frozenset({DriftKind.PREDICTION})
    episodes: list[DriftEpisode] = field(default_factory=list)
    _open: DriftEpisode | None = field(default=None, repr=False)
    _last_candidate_at: datetime | None = field(default=None, repr=False)

    @property
    def open_episode(self) -> DriftEpisode | None:
        """The episode currently firing, if any."""
        return self._open

    def observe(self, event: DriftEvent) -> tuple[DriftEvent, bool]:
        """Fold one evaluation into the episode history.

        Returns:
            The event with its ``episode_id`` attached, and whether it should
            raise a candidate update. The second value is ``False`` for a
            continuing episode inside its cooldown -- the drift is still real
            and still logged, but it has already been acted on -- and for a
            firing that no required family participated in.
        """
        if not event.qualifies(self.required_kinds):
            if self._open is not None:
                self._open.closed_at = event.detected_at
                self._open = None
            return event, False

        magnitude = max(
            (signal.magnitude for signal in event.fired if signal.magnitude == signal.magnitude),
            default=0.0,
        )

        if self._open is None:
            episode = DriftEpisode(
                episode_id=f"EP-{event.detected_at:%Y%m%dT%H%M%S}",
                onset=event.detected_at,
                last_seen=event.detected_at,
                kinds=event.kinds,
                peak_magnitude=magnitude,
            )
            self.episodes.append(episode)
            self._open = episode
            raise_candidate = True
        else:
            episode = self._open
            episode.last_seen = event.detected_at
            episode.events += 1
            episode.kinds = tuple(dict.fromkeys((*episode.kinds, *event.kinds)))
            episode.peak_magnitude = max(episode.peak_magnitude, magnitude)
            raise_candidate = (
                self._last_candidate_at is None
                or event.detected_at - self._last_candidate_at >= self.cooldown
            )

        if raise_candidate:
            episode.candidates_raised += 1
            self._last_candidate_at = event.detected_at

        stamped = DriftEvent(
            detected_at=event.detected_at,
            window_start=event.window_start,
            window_end=event.window_end,
            signals=event.signals,
            episode_id=episode.episode_id,
        )
        return stamped, raise_candidate

    def close(self, at: datetime | None = None) -> None:
        """Close any open episode, at the end of a replay."""
        if self._open is not None:
            self._open.closed_at = at if at is not None else datetime.now(UTC)
            self._open = None


def false_alarm_count(
    events: Sequence[DriftEvent],
    *,
    before: datetime,
    required: frozenset[DriftKind] | None = None,
) -> int:
    """Firings before a known-stable boundary.

    The measure that matters for a detector's credibility: one that fires on a
    quiet stretch will be ignored when it fires on a real one. Tests print this
    alongside the detection delay.

    Args:
        events: The replay's events.
        before: The boundary before which any firing is a false alarm.
        required: Count only firings that include one of these families -- the
            ones that would actually have raised a candidate. ``None`` counts
            every firing, which is the stricter reading.
    """
    return sum(
        1
        for event in events
        if event.detected_at < before
        and (event.is_drift if required is None else event.qualifies(required))
    )


def detection_delay(
    events: Sequence[DriftEvent],
    *,
    change_point: datetime,
    required: frozenset[DriftKind] | None = None,
) -> timedelta | None:
    """Time from a known change point to the first firing after it.

    Returns ``None`` when nothing fired. Used by the tests, which are the only
    place that knows where the planted change is.

    Args:
        events: The replay's events.
        change_point: The known change instant.
        required: Measure the delay to the first firing involving one of these
            families. ``None`` measures to the first firing of any kind, which
            is the faster -- and less specific -- figure.
    """
    for event in sorted(events, key=lambda item: item.detected_at):
        if event.detected_at < change_point:
            continue
        fired = event.is_drift if required is None else event.qualifies(required)
        if fired:
            return event.detected_at - change_point
    return None

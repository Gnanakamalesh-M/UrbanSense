"""Canonical location registry and resolution (SPEC section 13).

Sources name the same place differently: ``Anna Nagar``, ``Zone-17``,
``ZONE-B``, bare coordinates. A canonical registry maps the curated spellings
onto one id so observations from different feeds can be compared.

**Matching is exact-only, by design.** An id match or a *curated alias* match
resolves; anything else does not resolve at all. There is no similarity
threshold above which a name is accepted, because the spec is explicit:

    Do not automatically assume that similar names are identical.

``Anna Nagar East`` is a different road from ``Anna Nagar``. Auto-matching them
on 80% string overlap would merge two zones' traffic permanently, and the merge
would be invisible afterwards -- there is no later stage that can detect it,
because the evidence is gone. A name that does not resolve goes to review
instead (``LOCATION_REVIEW_REQUIRED``), which costs a human a minute and cannot
corrupt anything.

A similarity score *is* computed, but only to name candidate zones in the
review record so the human decision is quick. It never causes acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Similarity at or above which a non-matching name is worth suggesting to a
#: reviewer. Purely presentational -- it gates what the review record mentions,
#: never whether a row is accepted.
SUGGESTION_THRESHOLD = 0.6

#: How many candidate suggestions to record for a reviewer.
MAX_SUGGESTIONS = 3


def normalize_name(name: str) -> str:
    """Fold a location string for exact comparison.

    Case, surrounding whitespace, and the choice between spaces, hyphens and
    underscores are spelling noise rather than meaning: ``zone_b``, ``Zone-B``
    and ``ZONE B`` are the same token. Everything else is preserved, so
    ``Anna Nagar East`` stays distinct from ``Anna Nagar``.
    """
    folded = name.strip().lower()
    for separator in ("-", "_"):
        folded = folded.replace(separator, " ")
    return " ".join(folded.split())


class MatchKind(StrEnum):
    """How a location string resolved, if it did."""

    #: Matched the canonical ``location_id``.
    CANONICAL_ID = "canonical_id"
    #: Matched the registry's display name.
    NAME = "name"
    #: Matched a curated alias.
    ALIAS = "alias"
    #: Did not resolve. Requires review; never silently guessed.
    UNRESOLVED = "unresolved"


class ZoneEntry(BaseModel):
    """One canonical location.

    Attributes:
        location_id: The canonical id every observation resolves to.
        name: Human-readable display name.
        latitude: WGS84 latitude of the reference point.
        longitude: WGS84 longitude.
        aliases: Curated alternative spellings. Each is a deliberate human
            assertion that the alias means this zone -- which is what makes
            trusting it at confidence 1.0 safe, unlike a string-similarity
            guess.
        description: What this zone covers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    aliases: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class LocationMatch:
    """The outcome of resolving one location string.

    Attributes:
        resolved: Whether the string resolved to a canonical zone.
        location_id: The canonical id, or ``None`` when unresolved.
        zone: The registry entry, or ``None`` when unresolved.
        kind: How it resolved.
        confidence: ``1.0`` for a curated match, ``0.0`` otherwise. There are
            deliberately no values in between: a partial match is not a partial
            acceptance, it is a review item.
        suggestions: Candidate zone ids a reviewer might mean, with their
            similarity. Advisory only.
        detail: Human-readable explanation for the review record.
    """

    resolved: bool
    location_id: str | None
    zone: ZoneEntry | None
    kind: MatchKind
    confidence: float
    suggestions: tuple[tuple[str, float], ...] = ()
    detail: str = ""


class LocationRegistry(BaseModel):
    """The canonical location registry, loaded from configuration.

    Attributes:
        zones: Every known location.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    zones: tuple[ZoneEntry, ...] = ()

    @model_validator(mode="after")
    def _ids_and_aliases_are_unambiguous(self) -> Self:
        """No id, name or alias may point at two different zones.

        An alias claiming two zones is not a tie to break at runtime -- it is a
        configuration error, and resolving it by picking one would silently
        assign traffic to the wrong road.
        """
        ids = [zone.location_id for zone in self.zones]
        duplicate_ids = {i for i in ids if ids.count(i) > 1}
        if duplicate_ids:
            raise ValueError(f"duplicate location_ids in registry: {sorted(duplicate_ids)}")

        seen: dict[str, str] = {}
        for zone in self.zones:
            tokens = [zone.location_id, zone.name, *zone.aliases]
            for token in tokens:
                key = normalize_name(token)
                owner = seen.get(key)
                if owner is not None and owner != zone.location_id:
                    raise ValueError(
                        f"{token!r} is claimed by both {owner!r} and "
                        f"{zone.location_id!r}; an ambiguous alias cannot be resolved "
                        "automatically"
                    )
                seen[key] = zone.location_id
        return self

    def _lookup(self) -> dict[str, tuple[ZoneEntry, MatchKind]]:
        """Build the exact-match index, keyed by folded spelling."""
        index: dict[str, tuple[ZoneEntry, MatchKind]] = {}
        for zone in self.zones:
            index[normalize_name(zone.location_id)] = (zone, MatchKind.CANONICAL_ID)
        for zone in self.zones:
            index.setdefault(normalize_name(zone.name), (zone, MatchKind.NAME))
        for zone in self.zones:
            for alias in zone.aliases:
                index.setdefault(normalize_name(alias), (zone, MatchKind.ALIAS))
        return index

    def get(self, location_id: str) -> ZoneEntry | None:
        """Return the zone with this exact canonical id, if any."""
        wanted = normalize_name(location_id)
        return next((z for z in self.zones if normalize_name(z.location_id) == wanted), None)

    def resolve(self, name: str) -> LocationMatch:
        """Resolve a source's location string to a canonical zone.

        Args:
            name: The location string as the source wrote it.

        Returns:
            A :class:`LocationMatch`. Resolution happens only on an exact id,
            name or curated-alias match. Everything else comes back
            ``resolved=False`` with suggestions for a reviewer -- never a
            best-guess assignment.
        """
        folded = normalize_name(name)
        if not folded:
            return LocationMatch(
                resolved=False,
                location_id=None,
                zone=None,
                kind=MatchKind.UNRESOLVED,
                confidence=0.0,
                detail="location is empty",
            )

        found = self._lookup().get(folded)
        if found is not None:
            zone, kind = found
            return LocationMatch(
                resolved=True,
                location_id=zone.location_id,
                zone=zone,
                kind=kind,
                confidence=1.0,
                detail=f"exact {kind.value} match",
            )

        # No exact match. Gather suggestions for the reviewer -- and stop.
        # Nothing below this line may cause acceptance.
        scored: list[tuple[str, float]] = []
        for zone in self.zones:
            candidates = [zone.location_id, zone.name, *zone.aliases]
            best = max(SequenceMatcher(None, folded, normalize_name(c)).ratio() for c in candidates)
            if best >= SUGGESTION_THRESHOLD:
                scored.append((zone.location_id, round(best, 3)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        suggestions = tuple(scored[:MAX_SUGGESTIONS])

        if suggestions:
            hint = ", ".join(f"{zone_id} ({score:.0%})" for zone_id, score in suggestions)
            detail = (
                f"location {name!r} is not in the registry. Similar zones exist "
                f"({hint}) but similar is not identical, so this needs a human "
                "decision rather than an automatic match."
            )
        else:
            detail = f"location {name!r} is not in the registry and resembles no known zone."

        return LocationMatch(
            resolved=False,
            location_id=None,
            zone=None,
            kind=MatchKind.UNRESOLVED,
            confidence=0.0,
            suggestions=suggestions,
            detail=detail,
        )

    def __len__(self) -> int:
        """Number of registered zones."""
        return len(self.zones)

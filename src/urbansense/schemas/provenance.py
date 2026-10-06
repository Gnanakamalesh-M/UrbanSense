"""Provenance records (SPEC sections 7, 34).

Every observation must answer "where did this value come from?" without
ambiguity and without a lookup that might fail. Provenance is therefore a
required, immutable part of the observation itself rather than a side table.

``DocumentProvenance`` carries the extra evidence a document-derived value owes
the reader: the document, where in it, the original text, how confident the
extraction was, and crucially whether the value was *read* from the source or
*inferred* from it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbansense.schemas.enums import ExtractionKind
from urbansense.schemas.temporal import UTCDateTime

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Provenance(BaseModel):
    """Origin of an observation from a sensor, feed or historical dataset.

    Attributes:
        kind: Discriminator; ``"direct"`` for non-document origins.
        origin: Human-readable origin, e.g. ``"sensor-A12"`` or
            ``"tn_open_data_traffic_2024.csv"``.
        ingested_at: When this process first recorded the value.
        pipeline_version: Code version that produced the record, so a value can
            be traced back to the logic that normalized it.
        upstream_observation_ids: Lineage. A real-time observation that is later
            finalized as history points back here rather than overwriting it.
        notes: Free-text trail for reviewers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["direct"] = "direct"
    origin: str = Field(min_length=1)
    ingested_at: UTCDateTime
    pipeline_version: str = Field(min_length=1)
    upstream_observation_ids: tuple[str, ...] = ()
    notes: str | None = None


class DocumentProvenance(BaseModel):
    """Origin of a value extracted from an uploaded document.

    A document value is not trusted on arrival. It must say where it came from
    precisely enough that a human can open the source and check it.

    Attributes:
        kind: Discriminator; always ``"document"``.
        document_name: File name as uploaded.
        document_hash: Content hash, used to detect duplicate documents without
            relying on the file name (SPEC section 9).
        page: 1-indexed page, when the format has pages.
        section: Section or table identifier, when available.
        original_text: The source text the value was read from. Required -- it is
            what makes the hallucination guard checkable after the fact.
        extraction_kind: Whether the value is present in ``original_text``
            (``EXTRACTED_FACT``) or was derived by a model (``INFERRED``).
        extraction_confidence: Extractor confidence in [0, 1].
        extracted_at: When extraction ran.
        extractor_version: Which extractor produced it.
        ingested_at: When this process first recorded the value.
        pipeline_version: Code version that produced the record.
        upstream_observation_ids: Lineage, as for :class:`Provenance`.
        notes: Free-text trail for reviewers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["document"] = "document"
    document_name: str = Field(min_length=1)
    document_hash: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    original_text: str = Field(min_length=1)
    extraction_kind: ExtractionKind
    extraction_confidence: Confidence
    extracted_at: UTCDateTime
    extractor_version: str = Field(min_length=1)
    ingested_at: UTCDateTime
    pipeline_version: str = Field(min_length=1)
    upstream_observation_ids: tuple[str, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def _extraction_not_before_document_read(self) -> DocumentProvenance:
        """Extraction cannot postdate ingestion of its own result."""
        if self.extracted_at > self.ingested_at:
            raise ValueError(
                f"extracted_at ({self.extracted_at.isoformat()}) is after "
                f"ingested_at ({self.ingested_at.isoformat()}): a value cannot be "
                "ingested before it was extracted."
            )
        return self


#: Either provenance shape, discriminated on ``kind`` so round-tripping through
#: JSON reconstructs the right class instead of collapsing to the looser one.
AnyProvenance = Annotated[Provenance | DocumentProvenance, Field(discriminator="kind")]

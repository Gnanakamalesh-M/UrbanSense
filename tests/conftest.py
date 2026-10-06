"""Shared fixtures and factories for the test suite.

Later phases build their test data on these factories rather than inventing
observation literals, so that when the schema grows a required field the whole
suite learns about it in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from urbansense.schemas import (
    AggregationLevel,
    DocumentProvenance,
    ExtractionKind,
    Observation,
    ProblemType,
    Provenance,
    SourceType,
    ValidationStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A fixed instant in the past, so tests never race the future-time validator.
EVENT_TIME = datetime(2026, 9, 25, 18, 0, tzinfo=UTC)
RECEIVED_TIME = EVENT_TIME + timedelta(minutes=2)


@pytest.fixture
def config_dir() -> Path:
    """The repository's real configs/ directory."""
    return REPO_ROOT / "configs"


@pytest.fixture
def provenance() -> Provenance:
    """Provenance for a direct (sensor/dataset) observation."""
    return Provenance(
        origin="sensor-A12",
        ingested_at=RECEIVED_TIME,
        pipeline_version="0.1.0",
    )


@pytest.fixture
def document_provenance() -> DocumentProvenance:
    """Provenance for a value read verbatim out of a document."""
    return DocumentProvenance(
        document_name="traffic_report_q3.pdf",
        document_hash="sha256:9f2b1c" + "0" * 58,
        page=7,
        section="Table 3",
        original_text=(
            "Traffic in Zone B increased to approximately 8,200 vehicles per hour "
            "between 6 PM and 8 PM."
        ),
        extraction_kind=ExtractionKind.EXTRACTED_FACT,
        extraction_confidence=0.94,
        extracted_at=RECEIVED_TIME,
        extractor_version="docint-0.1.0",
        ingested_at=RECEIVED_TIME + timedelta(seconds=5),
        pipeline_version="0.1.0",
    )


@pytest.fixture
def make_observation(provenance: Provenance) -> Callable[..., Observation]:
    """Return a factory producing a valid traffic observation.

    Keyword overrides are passed straight through, so a test can change exactly
    the one field it is about and let everything else stay valid.
    """

    def factory(**overrides: Any) -> Observation:
        fields: dict[str, Any] = {
            "source_id": "sensor-A12",
            "source_type": SourceType.REAL_TIME,
            "event_time": EVENT_TIME,
            "received_time": RECEIVED_TIME,
            "location_id": "ZONE-B",
            "latitude": 13.0827,
            "longitude": 80.2707,
            "problem_type": ProblemType.TRAFFIC,
            "metric": "traffic_volume",
            "value": 8200.0,
            "unit": "vehicles/hour",
            "aggregation_level": AggregationLevel.HOURLY,
            "validation_status": ValidationStatus.VALID,
            "provenance": provenance,
        }
        fields.update(overrides)
        return Observation(**fields)

    return factory


@pytest.fixture
def observation(make_observation: Callable[..., Observation]) -> Observation:
    """A single valid traffic observation."""
    return make_observation()

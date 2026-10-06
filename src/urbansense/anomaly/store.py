"""Writing anomaly scans to disk (SPEC sections 24, 44).

The scan was already computed and printed; it was never persisted. That was
fine while the only consumer was a terminal, but an API that serves stored
artifacts has nothing to read, and a flagged reading that exists only in
scrollback cannot be revisited when the explanation arrives later.

One timestamped file per run, appended rather than overwritten, for the reason
the pattern store gives: the detector's threshold and the patterns it consults
both change between phases, and overwriting would destroy what was flagged
before.

The population header is written with the records. A count of 41 anomalies means
something different against 26,000 scanned readings than against 400, and
crucially this scan runs on the **pre-quarantine** population -- valid plus
held-back observations -- so the defects Phase 2 already removed are still
visible. A reader who assumed the unified layer would reach the wrong conclusion
about the rate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from urbansense.anomaly.detector import AnomalyResult

#: Default directory for anomaly reports.
DEFAULT_ANOMALY_DIR = Path("reports") / "anomalies"


def _stamp(now: datetime | None = None) -> str:
    """Filename-safe UTC timestamp."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def anomaly_payload(result: AnomalyResult) -> dict[str, object]:
    """Build the serializable scan report.

    The header comes before the records so that anything reading the file sees
    the population and the threshold before the findings.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run": {
            "observations_scanned": result.observations_scanned,
            "flagged": len(result.anomalies),
            "rate": round(result.rate, 6),
            "threshold_mad_sigma": result.threshold,
            "unexplained": len(result.unexplained),
            "by_leading_explanation": dict(result.by_leading_explanation),
            "population": (
                "valid plus quarantined observations, so the defects validation "
                "already removed are still visible here; the rate is not "
                "comparable to one computed over the unified layer alone"
            ),
        },
        "anomalies": [anomaly.as_dict() for anomaly in result.anomalies],
        "notes": [
            "Nothing here is classified as an error: is_error is None on every "
            "record. A flagged reading may be a sensor fault, an event or a "
            "genuine extreme, and this evidence cannot separate them.",
            "Explanations are ranked candidates, not verdicts. The leading one "
            "is the best supported, which is not the same as the cause.",
            "Expected values are the reading's own hour/weekday/zone cell "
            "median; deviation is in MAD-sigma, robust to the outliers it is "
            "looking for.",
        ],
    }


def write_anomaly_report(
    result: AnomalyResult,
    *,
    root: Path = DEFAULT_ANOMALY_DIR,
    now: datetime | None = None,
) -> Path | None:
    """Write one scan as a timestamped JSON file.

    Args:
        result: The scan to record.
        root: Directory to write into. Created on demand.
        now: Timestamp for the filename. Defaults to now.

    Returns:
        The file written, or ``None`` when the scan flagged nothing. An empty
        run leaves no file, so the presence of one means there is something to
        read -- the same convention as the pattern store.
    """
    if not result.anomalies:
        return None
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"anomalies-{_stamp(now)}.json"
    target.write_text(
        json.dumps(anomaly_payload(result), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target

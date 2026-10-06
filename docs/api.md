# The API

A read-only HTTP layer over the artifacts the pipeline has already written (SPEC §44). It exists so
the system is inspectable without running Python: the unified data layer with its provenance, the
discovered patterns with their evidence, the recorded predictions against their outcomes, the drift
log, the model registry and the experiment reports.

```bash
python scripts/serve_api.py              # 127.0.0.1:8000
python scripts/serve_api.py --port 8080 --rebuild
python scripts/serve_api.py --check      # what is available, then exit
```

Then open <http://127.0.0.1:8000/docs> for the interactive schema.

## Three postures, and why

**Read-only.** Only `GET` routes are registered. There is no write, update or delete endpoint
anywhere in the application, so a `POST` is rejected by the router rather than by a guard that a
later route could forget to apply. A test walks `/openapi.json` and asserts that no path declares
anything but `get`, and another fingerprints the whole artifact tree before and after hitting every
endpoint to prove no request writes a file.

**Unauthenticated, and loud about it.** There is no auth layer. Rather than leave that implicit,
`/health` returns `"authentication": "none"` with the bind hint, and `scripts/serve_api.py` binds
`127.0.0.1`. `--host` can override that and prints a warning when it does: exposing an
unauthenticated service should be a decision, not a side effect of a flag.

**Nothing is recomputed per request.** Every endpoint opens a file. An endpoint that rebuilt
features or refitted a model would turn "inspect what the pipeline produced" into "run the pipeline
again, slightly differently", and the numbers served would stop matching the numbers reported.

### The one build step

Twelve of the thirteen artifact families already existed on disk, written by earlier phases as they
worked. The unified data layer did not: it is rebuilt from CSV in about half a minute.

So it is materialized once, to the already-gitignored `data/processed/`:

| file | contents |
|---|---|
| `unified.jsonl` | one row per unified observation, **with its provenance** |
| `conflicts.json` | recorded disagreements, every candidate value kept |
| `data_quality.json` | counts and per-source health from the validation run |

The alternative was to rebuild it inside the API process and cache it, and that would have made the
service not read-only: it would run ingestion, validation and reconciliation on a request, hold the
result in memory, and lose it on restart. Exporting keeps the serving path to "open a file", and the
export is itself an artifact a person can `grep`.

`serve_api.py` runs the export automatically when none is present, and `--rebuild` forces it.

## Endpoints

| route | returns |
|---|---|
| `GET /health` | status, version, posture, and which artifact families are present |
| `GET /data/unified` | unified observations with provenance; filter `zone`, `metric`, `from`, `to` |
| `GET /conflicts` | recorded disagreements, every candidate value kept |
| `GET /quarantine` | held-back records with their raw payloads; filter `reason`, `source_id` |
| `GET /data-quality` | counts and per-source health |
| `GET /patterns` | discovered patterns with evidence; filter `kind`, `status` |
| `GET /patterns/{pattern_id}` | one pattern **plus the discovery run that found it** |
| `GET /predictions` | recorded predictions and outcomes; filter `zone`, `date`, `model_version` |
| `GET /anomalies` | flagged readings with ranked candidate explanations; filter `zone` |
| `GET /drift` | drift evaluations, episodes and candidate decisions; `firing_only` |
| `GET /models` | model versions, oldest first; filter `status` |
| `GET /models/{version}` | one registry entry in full |
| `GET /experiments` | experiment reports, newest first |

Paged endpoints return `{items, total, limit, offset, note}`. `total` is the count **after filtering
and before paging**, which is what lets a client tell "that is all of them" from "there are twelve
more". `limit` is bounded at 500; asking for more is a `422` rather than a silent full scan of
59,000 rows.

### Provenance

Every `/data/unified` item carries `source_id`, `source_type`, all three timestamps and the
`provenance` block:

```json
{
  "observation_id": "OBS-REC-000003",
  "record_id": "REC-000003",
  "source_id": "sensor-A12",
  "source_type": "historical",
  "location_id": "ZONE-A",
  "metric": "average_speed",
  "value": 47.0,
  "unit": "km/hour",
  "event_time": "2025-09-30T18:30:00+00:00",
  "source_time": "2025-09-30T19:30:37+00:00",
  "received_time": "2026-10-04T10:49:58.015804+00:00",
  "validation_status": "valid",
  "is_imputed": false,
  "duplicate_of": null,
  "conflicts_with": [],
  "provenance": {
    "kind": "direct",
    "origin": "sensor-A12",
    "ingested_at": "2026-10-04T10:49:58.015804Z",
    "pipeline_version": "ingestion-1.0.0",
    "upstream_observation_ids": []
  }
}
```

"Where did this value come from?" is the question this project exists to answer, so it is answerable
from a single response rather than requiring a second lookup. The three timestamps are all kept
because they answer different questions — when the measurement happened, when the source stamped it,
when we received it — and collapsing them would make a late arrival indistinguishable from a prompt
one.

A `value` of `null` means missing. It is never a zero, and `is_imputed` is `false` on every row in
this project.

## A missing artifact is not an error

A fresh clone has no discovery report and no experiment report, because nobody has run them yet.
Those endpoints answer `200` with an empty page and a `note` naming the command that would produce
the file:

```json
{
  "items": [],
  "total": 0,
  "limit": 50,
  "offset": 0,
  "note": "no experiment report; run: python scripts/run_experiment.py"
}
```

"Nothing has been computed yet" is a true and actionable answer to "list the experiments", and a
`500` is not. `/health` lists what is and is not available, so an empty endpoint can be told from a
broken one without reading the server's logs.

A `404` is reserved for a genuinely unknown id, and names both the id and what was searched:

```
unknown model version 'Traffic-v9.9'; 2 version(s) in the registry: Traffic-v1.0, Traffic-v1.1
unknown pattern 'PAT-999'; no discovery report; run: python scripts/discover_patterns.py
```

Those are different problems with different fixes, so they do not share a message.

## Limitations

- **No authentication, no rate limiting, no TLS.** Localhost-only by default, and not written for
  anything else.
- **No write endpoints at all**, by design. Corrections enter through the pipeline, where they are
  validated and keep their provenance — not through HTTP.
- **Artifacts are cached for the life of the process.** They are written by batch jobs and do not
  change under a running server, so re-parsing a 35MB JSONL on every page would be waste rather
  than freshness. A re-export needs a restart, or `--rebuild` on the way up.
- **Filters are applied in Python over the whole artifact.** Fine at 59,000 rows; this is not a
  database, and a dataset an order of magnitude larger would want one.
- **`/data/unified` is as current as the last export**, not as the last ingestion. `/health` does not
  report the export's age.
- **Single process, no concurrency story beyond what uvicorn gives.** It serves one person
  inspecting a portfolio project, which is what it was built for.

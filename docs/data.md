# Data model

The canonical record is `urbansense.schemas.Observation`: one metric, at one place, over one time
interval, from one source. Every module reads and writes this type.

It is `frozen=True` and `extra="forbid"`. Immutable because a validated observation must not change
under a consumer; closed because a misspelled field name should fail loudly rather than be silently
absorbed and ignored.

## Fields

| Group | Fields |
|---|---|
| Identity | `observation_id`, `source_id`, `source_type` |
| Time | `event_time`, `received_time`, `source_time`, `measurement_duration`, `reporting_period_start/end` |
| Place | `location_id`, `latitude`, `longitude` |
| Measurement | `problem_type`, `metric`, `value`, `unit`, `original_value`, `original_unit`, `aggregation_level` |
| Trust | `confidence`, `validation_status`, `provenance`, `is_imputed`, `imputation_method` |
| Integrity links | `duplicate_of`, `conflicts_with`, `quarantine_reason` |
| Extension | `attributes` |

`attributes` is the extension point: problem-specific detail lives there, so the core schema stays
closed while remaining extensible (SPEC section 6).

## The invariants, and why each exists

### Time is never ambiguous

A naive datetime is rejected outright, and aware values are normalized to UTC. An instant without an
offset cannot be compared across sources, and guessing the zone is how a 6 PM congestion peak
silently becomes a 12:30 PM one. Local wall-clock belongs in provenance or in display code.

### `event_time` is not `received_time`

"When it happened" and "when we were told" are separate fields because late-arriving data, backfills
and newly uploaded old reports all depend on the distinction. A report uploaded in 2026 about 2022
is 2022 data; `reporting_period_start/end` records the span it covers.

`received_time` may not precede `event_time` — that combination is a clock bug. It is admissible
only on a quarantined record, because the bug itself is worth keeping.

### No future event times

An `event_time` beyond now (past a five-minute skew tolerance) is either a clock error or future
data leaking toward a training set. Either way it must not pass silently. This is the schema-level
half of the leakage prevention in SPEC section 33; the evaluation-level half is temporal splitting.

### Missing is never zero

`value` is `float | None`. A sensor that reported nothing did not report zero vehicles, and
substituting zero would teach a model that outages look like empty roads.

If a value *was* filled, the record says so: `is_imputed=True` requires both the value and an
`imputation_method`. The reverse is enforced too — a method without the flag is rejected — so an
estimate can never be mistaken for a measurement.

### Provenance is mandatory

`provenance` is a required field, not an optional one with a default. "Where did this value come
from?" must be answerable from the record itself, without a join that might fail.

`Provenance.upstream_observation_ids` carries lineage: a real-time observation later finalized as
history produces a *new* record pointing back, rather than overwriting the original.

### Document values must be auditable

A `source_type=DOCUMENT` observation must carry `DocumentProvenance`: document name, content hash,
page or section, the `original_text` the value was read from, extraction confidence, and timestamp.
The content hash is what lets duplicate documents be detected without trusting filenames.

The hallucination guard is `extraction_kind`. `EXTRACTED_FACT` means the value is present in the
source text; `INFERRED` means a model derived it. An `INFERRED` record **cannot** be created with
`validation_status=VALID` — it must go to `NEEDS_REVIEW`. So "traffic increased" can never quietly
become "traffic increased by 27%".

### Units are never dropped

Normalization retains `original_value` and `original_unit` alongside the canonical `value` and
`unit`, and neither half of the original may be dropped on its own. A conversion that cannot be
re-checked is a conversion that will eventually be wrong without anyone noticing.

### Nothing is destroyed

- **Duplicates** are linked via `duplicate_of` and marked `DUPLICATE`. Both records survive.
- **Conflicts** are recorded in `conflicts_with` and marked `CONFLICT`. Disagreement between
  historical 7900, real-time 8200 and a document's 8100 is information, not a merge problem.
- **Invalid records** become `QuarantineRecord` entries holding `raw_payload` exactly as received —
  deliberately typed as a plain mapping, because a record that fails schema validation is precisely
  the one a reviewer most needs to see, and normalizing it on the way in would destroy the evidence.

A quarantined record must state a `quarantine_reason`; data is never held without an explanation.

## Validation states

`PENDING` to `VALID` is the happy path. The others each hold a record *in* the system rather than
dropping it: `SUSPECT`, `CONFLICT`, `DUPLICATE`, `QUARANTINED`, `NEEDS_REVIEW`, `REJECTED`.

`SUSPECT`, `QUARANTINED` and `REJECTED` relax the cross-field consistency checks, since quarantine
exists to hold malformed evidence that would otherwise be unrepresentable.

## Configuration

Metrics, canonical units, accepted input units, aliases and plausible ranges are declared per
problem in `configs/problems/<name>.yaml` — not in code. This is what makes the terminology mapping
from SPEC section 5 data rather than logic: `vehicle count`, `vehicles observed` and `flow` all
resolve to `traffic_volume` through `Settings.resolve_metric`, matching case-insensitively and
treating spaces, hyphens and underscores alike.

Plausible ranges flag outliers for *investigation*. Exceeding `plausible_max` does not mark a value
wrong — 20,000 vehicles where 5,000 is normal could be a festival, a diversion or a failed sensor,
and which one it is cannot be decided by a range check (SPEC section 16).

---

## The synthetic dataset (Phase 1)

`data/sample/` holds a generated traffic dataset with **known ground truth**, committed so the
repository works offline on clone. It exists so later phases can be *scored* rather than merely
demonstrated: a phase claiming to find the Friday pattern or the drift event is checked against
what was actually planted, recorded in `data/sample/ground_truth.json`.

### Why labels never appear in the CSV

The feed columns are only what a real source sends:

```
record_id, source_id, interval_end_local, location_id, metric, value, unit
```

No `is_friday_evening`, no duplicate flag, nothing naming an injected effect. A label column in the
training data is a leakage channel — a phase that could read the answer off its input would be
credited with a discovery it never made. Labels live in `ground_truth.json`, keyed by `record_id`
or by time window, and the test suite asserts the header contains nothing label-shaped.

Coordinates live in `zones.csv`, a location registry (SPEC section 13), rather than being repeated
on every reading — which is both how a real feed works and what keeps the file small.

### The three timestamps, concretely

The feed stamps the **end** of each measurement interval, in its own local time (IST, +05:30), with
the reporting sensor's clock skew baked in. From one row, ingestion derives:

| Field | Value for a row stamped `2026-02-06 19:00:37+05:30` |
|---|---|
| `source_time` | `2026-02-06T13:30:37Z` — what the source asserted, skew intact |
| `event_time` | `2026-02-06T12:30:00Z` — when the measured hour began |
| `received_time` | when ingestion read it (now, or the stream's emission instant) |

`event_time` is derived by snapping `source_time` to the nearest interval boundary and subtracting
one interval. The snap matters: without it, a sensor 52 seconds slow would place its 18:00 reading
at 17:59:08, scattering one hour's readings across two buckets and breaking every cross-source
join. Snapping happens in the source's own offset, because a half-hour offset such as IST has its
hour boundaries at `:30` past the UTC hour.

The skew is normalized out of `event_time` but **kept** in `source_time`. That is deliberate: a
pipeline cannot subtract an offset nobody told it about, and preserving the raw instant is what
leaves a drifting or frozen sensor clock detectable later (SPEC section 17).

`provenance.notes` carries the original local string verbatim, so the whole normalization can be
re-checked from the record alone.

### What ingestion does and does not do

Ingestion transcribes. It produces `PENDING` observations, resolves metric aliases from config, and
accounts for every input row — each one becomes an observation or a `QuarantineRecord` carrying its
original payload, which `IngestionResult.accounts_for_every_row` asserts.

It does **not** judge values. A negative vehicle count parses fine and passes straight through,
because catching it is the integrity engine's job (Phase 2) and doing it at the boundary would
pre-empt the thing Phase 2 exists to prove. Only rows that cannot be *represented* are rejected: an
unparseable timestamp, a naive timestamp with no offset, a unit the metric does not accept.

Rejections are **returned, not written**. Nothing in `urbansense.ingestion` opens a file for
writing, so the module cannot touch `data/raw/`; persisting quarantine is the caller's decision
(`python scripts/ingest_demo.py --write-quarantine`).

---

## What normalization does to an observation (Phase 2)

An ingested record arrives `PENDING` with the source's own spellings. Normalization produces a
*new* record (the schema is frozen) with:

| field | before | after |
|---|---|---|
| `location_id` | `Anna Nagar` | `ZONE-B` (canonical) |
| `latitude` / `longitude` | from the dataset's `zones.csv` | from the canonical registry |
| `metric` | `vehicle count` | `traffic_volume` |
| `value` / `unit` | `2050` / `vehicles/15min` | `8200.0` / `vehicles/hour` |
| `original_value` / `original_unit` | — | `2050.0` / `vehicles/15min` |
| `validation_status` | `PENDING` | `VALID`, or `SUSPECT` if flagged |
| `attributes["recency"]` | — | `historical` |
| `attributes["arrival_lag_seconds"]` | — | the event-to-receipt gap |

Both representations of the value are retained, which is what makes the conversion auditable later
(SPEC §11). Unit *spelling* is also folded — `km/h`, `kmph` and `KPH` all resolve to `km/hour` —
but only through an explicit synonym table; an unrecognized unit is never guessed at.

### Where the unit knowledge lives

A deliberate split. Which units a metric may *arrive* in is policy, and lives in
`configs/problems/traffic.yaml` as `accepted_units`. How to convert between two known units is
physics, does not vary by deployment, and lives in `urbansense.preprocessing.units`.

A test asserts every `accepted_units` entry in every problem config is convertible to that metric's
`canonical_unit`, so the two cannot drift apart silently — if they did, every row using that unit
would be quarantined and the cause would look like bad data rather than a missing conversion.

### Recency is not source type

Two independent axes, kept separate on purpose:

- `source_type` — **which stream** the value arrived on (`HISTORICAL`, `REAL_TIME`, `DOCUMENT`).
  SPEC §4 requires this to stay traceable, so normalization never overwrites it.
- `attributes["recency"]` — **how old** it was on arrival (`CURRENT` ≤ 2 h, `RECENT` ≤ 48 h,
  `HISTORICAL` beyond; thresholds in `configs/base.yaml`).

A reading can arrive on the live feed and still be months old. Collapsing the two would lose the
ability to ask "what did the real-time feed actually report?", and overwriting the stream label to
mean "old" would discard provenance the project is built around. The whole demo history classifies
`HISTORICAL`, and so does the replayed July holdout — which is correct: a replay of past data is
historical evidence, not current conditions.

### Missing survives all of it

A `None` value is converted in name only: the unit is normalized so the record still says what the
reading *would* have been measured in, but the value stays `None`, `is_imputed` stays `False`, and
no imputation happens anywhere in Phase 2. A missing reading is also not a range violation —
treating absence as invalid would quarantine every gap in the feed.

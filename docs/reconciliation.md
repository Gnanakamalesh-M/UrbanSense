# Integrity and reconciliation

> Status: **not implemented** (Phase 2). The record types and the quarantine contract exist today;
> the detectors do not.

## The rule the whole module serves

> Never silently overwrite, hide or destroy conflicting information.

Everything below is a consequence of it.

## Duplicate detection (SPEC section 9)

Candidate duplicates are identified on location, timestamp, metric, value, measurement scope and
source. Detection does **not** delete:

```
Duplicate detected -> link related observations -> preserve source records
                   -> use one logical observation where appropriate
```

The link is `Observation.duplicate_of`, with status `DUPLICATE`. Both records stay queryable, so a
wrong duplicate judgement is reversible.

Duplicate *documents* are detected by `DocumentProvenance.document_hash`, not by filename.

## Conflict detection (SPEC section 10)

Three sources reporting 7900, 8200 and 8100 may or may not be in conflict. First establish whether
they describe the same observation at all — same timestamp, location, metric, unit, measurement
duration, spatial coverage, aggregation level and event.

- Different observations: keep them separately. There is no conflict.
- Same observation, disagreeing values: mark `CONFLICT`, record `conflicts_with`, and send to
  reconciliation or review.

No source automatically wins. "Real-time is fresher" and "historical is authoritative" are both
policies, not facts, and a policy that silently discards a value destroys the evidence that it was
the wrong policy.

## Normalization before comparison

Comparison is meaningless until units, times, locations and aggregation levels are commensurable
(SPEC sections 11–13):

- **Units**: `vehicles/15min` to `vehicles/hour`, `cm` to `mm`. The original value and unit are
  always retained. An unrecognized unit is quarantined, never coerced.
- **Time**: everything in UTC; event time, receipt time and reporting period kept distinct.
- **Location**: a canonical registry maps "Anna Nagar", "Anna Nagar East" and "Zone-17". Similar
  names are **not** assumed identical — low mapping confidence raises `LOCATION REVIEW REQUIRED`
  rather than guessing.

## Invalid, missing, outlying

- **Invalid** (negative traffic, impossible timestamps, unknown units or locations): quarantined
  with the raw payload, never deleted (SPEC section 14).
- **Missing**: represented explicitly, never as zero. Strategies — leave missing, interpolate,
  forward fill, model-based impute, exclude — are recorded on the record when applied
  (SPEC section 15).
- **Outliers**: detected, then investigated. 20,000 vehicles where 5,000 is normal could be sensor
  failure, an accident, a festival or a diversion. A range check cannot tell which, so it flags
  rather than judges (SPEC section 16).
- **Sensor failure**: frozen values, repeated identical readings, abnormal frequency, long gaps and
  sudden stream loss feed a per-source health record. Missing sensor data is never read as zero
  (SPEC section 17).

## Quarantine (SPEC section 18)

`urbansense.schemas.QuarantineRecord` holds the reason, a human-readable detail, the detector that
raised it, the timestamp, and `raw_payload` **exactly as received**. The parsed observation is
optional, because the records most worth keeping are often the ones that failed to parse at all.

Nothing in `quarantine/` is deleted by the pipeline. Entries are dispositioned by review and marked
resolved; the original is retained.

## Unified data layer (SPEC section 19)

Only validated, reconciled sources are combined, and the unified layer retains source references
throughout — a combined figure a user cannot decompose is a figure they cannot audit.

---

## What Phase 2 implemented

The detectors below are live. Duplicate detection, conflict detection and
reconciliation remain Phase 3.

### Entry point

```python
validate_and_normalize(ingestion_result, *, settings, registry) -> ValidatedResult
```

Stage order is fixed and load-bearing:

1. **location** — resolve to a canonical zone; an unresolvable zone makes everything
   downstream meaningless, so it is settled first;
2. **metric and unit** — convert to the canonical unit. Range bounds are *declared* in canonical
   units, so checking before converting would compare `2050 vehicles/15min` against a per-hour
   ceiling and pass a value four times the limit;
3. **recency** — classify how old the reading was on arrival;
4. **ranges** — against the metric's declared bounds;
5. **frozen sensors** — needs the whole series, so it runs after the per-row passes;
6. **outliers** — needs the surviving population, so it runs last.

Steps 1–4 can quarantine a row. Steps 5–6 are the two that need *context*: a frozen run is
invisible one reading at a time, and an outlier is only unusual relative to its peers.

### The two-bucket partition

`len(valid) + len(quarantined) == input rows`, asserted by
`ValidatedResult.accounts_for_every_row`. Ingestion's own rejections are carried forward so the
total reconciles against the file rather than against our own bookkeeping.

Suspicious observations are **not** a third bucket. An outlier stays in `valid` with
`validation_status=SUSPECT`, and `.suspicious` is a view over `valid`. That keeps the partition
exact while honouring SPEC §16 — a festival spike is real data.

### Impossible vs unusual

The distinction the whole module turns on:

| | example | verdict |
|---|---|---|
| **impossible** | −484 vehicles/hour, 480 km/h | not a measurement of anything → **quarantined** |
| **unusual** | 20,000 where 5,000 is normal | could be a festival, accident, diversion or a genuine record → **flagged `SUSPECT` and kept** |

Nothing is clipped. A negative count is never silenced with `max(0, value)`; that would turn a
sensor fault into a plausible-looking reading and destroy the evidence.

### Two detector subtleties worth knowing

**Outliers are conditioned on `(zone, metric, weekday, hour)`.** Traffic at 03:00 and at 18:00 are
different populations, and so are Friday and Sunday. Unconditioned, the detector would flag every
Friday-evening peak and every post-drift reading — the two strongest *real* patterns in the data —
as anomalies. Within its own weekday-and-hour group a recurring peak is the norm, which is correct:
a pattern is not an anomaly. Median and MAD rather than mean and standard deviation, because the
mean is dragged by the very outliers being looked for.

**A frozen run of a no-activity value is not a frozen sensor.** A rain gauge reading 0 mm through a
dry April is working perfectly, and a detector on a closed road legitimately counts nothing for
hours. `validation.frozen_ignored_values` (default `[0.0]`) excludes those. Long zero runs are not
ignored by the system as a whole — they still face the outlier detector, which flags a daytime zero
against its own hour's median. "No activity" is suspicious but possible; a frozen *non-zero*
reading is not a measurement at all.

### Measured on the demo dataset

58,997 rows in → 58,928 valid + 69 quarantined, partition exact:

| reason | count | what it was |
|---|---|---|
| `sensor_failure` | 60 | the planted ZONE-D frozen window, matched exactly |
| `invalid_value` | 6 | 4 negative volumes + 2 impossible speeds |
| `invalid_timestamp` | 2 | carried forward from ingestion |
| `unknown_unit` | 1 | `vehicles/fortnight`, not coerced |

**Precision 1.000, recall 1.000** against the planted defects, with zero false positives — no
Friday-peak row and no post-drift row was quarantined. 47 rows (0.08%) flagged `SUSPECT` and
retained. `sensor-D21` reports `FAILED`; the other sensors and the rain gauge report `OK`.

`city-warehouse` also reports `FAILED`, correctly: it is the planted conflict source, contributes
eight readings in total and then goes silent, which is exactly what a stale batch feed looks like.

### No detector reads the answers

Nothing under `src/` references `ground_truth.json` outside the generator, enforced by a test that
parses every source file's AST (prose in a docstring is fine; a code-level reference is not). A
detector that consulted the planted answers would make every later measurement a tautology.

---

## What Phase 3 implemented

Duplicate detection, conflict detection, reconciliation and the unified data layer. Phase 2's
detectors are unchanged; `reconcile()` is a separate entry point taking a `ValidatedResult`.

### One grouping key, three outcomes

Duplicates and conflicts are the same question — "do these records describe the same observation?" —
with different answers once the values are compared. So there is **one** key and one pass:

`ObservationKey` = `(location_id, event_time, metric, unit, measurement_duration,
aggregation_level)` — SPEC §10's checklist, as an explicit tuple. A field accidentally left out
would silently merge records that measure different things, and nothing downstream could detect it
afterwards because the evidence would be gone.

`value` and `source_id` are deliberately **not** in the key. They are what separate the outcomes
*within* a group; putting them in would make a conflict unrecognizable, since the two disagreeing
records would land in different groups and never be compared.

| members | outcome |
|---|---|
| values agree, same `source_id` | `SAME_SOURCE_REPEAT` — a pipeline or retry artefact |
| values agree, different `source_id` | `CROSS_SOURCE_AGREEMENT` — corroboration, not redundancy |
| any pair materially disagrees | `ConflictGroup` — the **whole** group |

One dissenter makes the entire group a conflict, even when two of three agree. Splitting it into
"the agreeing majority plus a dissenter" would be picking a winner by vote — and a majority is not
evidence, since two feeds can share one upstream fault.

### Materiality

Two values agree when `|a − b| ≤ max(absolute_tolerance, relative_tolerance · max(|a|, |b|))`.
Defaults in `configs/base.yaml`: 2% relative, 0.5 absolute. The absolute floor matters — without
it, two near-zero readings differing by one rounding step would look like a 100% disagreement and
manufacture conflicts out of quiet hours.

Missing values: two `None`s agree (both say nothing was measured). A `None` and a reading do **not**
— that is a real disagreement about whether anything was observed, and calling it agreement would
let a gap silently absorb a reading.

### Nothing is deleted or overwritten

**Duplicates are linked.** The representative (lowest `received_time`, `event_time`,
`observation_id`) goes to `unified` carrying `source_references` for every member; the others keep
their own records with `duplicate_of` set and `validation_status=DUPLICATE`. Deleting the copy
would be the obvious implementation and is exactly wrong: a mistaken duplicate judgement has to
stay recoverable.

**Conflicts are recorded.** Every member keeps its own value, gets `conflicts_with` linked, and
goes to the review queue in `quarantine/conflicts/` with all candidates, the policy name and
`resolved: false`. **No member reaches the unified layer** — a disputed slot is a visible hole
rather than an invented value.

The only policy in v1 is `manual_review`, and that is a deliberate limitation. The field exists so
that when automatic policies arrive, every reconciled record says which rule produced it.

### The three-way partition

`len(unified) + len(duplicate_linked) + len(conflicted_observations) == len(validated.valid)`,
asserted by `accounts_for_every_observation`. Set disjointness is asserted separately by
`observation_ids_are_disjoint`, because counts matching is necessary but not sufficient — a record
in two buckets while another vanished would still add up.

Only *input* observations populate the buckets; nothing synthetic is added, which is what keeps the
invariant a simple equality.

### Unified layer and source views

Source-specific views come **before** the combined one (SPEC §20): `.historical_view`,
`.real_time_view`, `.document_view`, then `.unified`. Every unified row — collapsed or singleton —
carries `source_references`, `contributing_source_ids` and `contributing_source_types`, so a
combined figure can always be decomposed back into the streams that produced it.

### Provenance

`explain_provenance(identifier, result)` accepts an `observation_id` **or** the source's own
`record_id`, because a user holding a feed export knows the latter. It resolves records in all
three buckets — a tool that only handled unified rows would fail for exactly the records people ask
about — and returns the record, its provenance, the original feed row, the duplicate group with
every member's source, the conflict peers with their values, and the reconciliation state.

`origin_of()` handles both provenance shapes: a sensor reading's origin is its feed, a
document-derived value's is the document, page and extraction confidence.

### Measured on the demo dataset

58,928 valid observations → **58,900 unified + 12 duplicate-linked + 16 conflicted**, exact and
disjoint.

| | count | scoring |
|---|---|---|
| `SAME_SOURCE_REPEAT` groups | 12 | precision 1.000, recall 1.000 |
| `CROSS_SOURCE_AGREEMENT` groups | 0 | none in the demo data; covered by a hand-built test |
| conflict groups | 8 (all cross-source) | precision 1.000, recall 1.000 |

The cross-source agreement path has no instance in the generated file, so it is exercised by a
hand-built fixture asserting the collapse to one row, both source references retained, and the
group reported as `CROSS_SOURCE_AGREEMENT` rather than `SAME_SOURCE_REPEAT`.

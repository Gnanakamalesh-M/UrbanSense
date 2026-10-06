# Pattern discovery and evolution

> Status: **implemented (Phase 5)** for traffic. Discovery, evolution tracking, interaction testing
> and anomaly detection run over the unified layer.

UrbanSense is meant to *discover* patterns, not only to predict predefined targets (SPEC
section 21). Phase 4 **measured** the Friday ZONE-B effect by being told where to look — the slice
was named in the test. Phase 5 has to **find** it.

```bash
python scripts/discover_patterns.py
python scripts/discover_patterns.py --no-write
python scripts/discover_patterns.py --kind spatiotemporal --min-strength 1.5
```

## What one run reports

```
4 pattern(s) from 672 hypotheses at FDR q=0.05; 0 cell(s) too sparse to test;
58,900 unified rows; seed 20260101
```

That header prints **before** any finding and cannot be suppressed. One pattern out of 672 tests is
a different claim from one out of ten, and a reader who sees the findings first will believe the
stronger one.

| discovered | effect | planted | status |
|---|---|---|---|
| `ZONE-B` + Friday + 18:00–20:00 | **×2.287** (1.939–2.404) | ×2.2 | stable |
| rainfall → `traffic_volume` | **+0.0309/mm** (+0.0287–+0.0332) | +0.03/mm | stable |
| `ZONE-A` baseline from 2026-03 | **×1.176** (1.118–1.233) | ×1.18 | strengthening |
| `ZONE-B` baseline from 2026-03 | **×1.167** (1.110–1.220) | ×1.18 | stable |

Zero spurious discoveries in `ZONE-C` and `ZONE-D`, where nothing was planted. That matters more
than recovering the three real effects: a scan reporting ten patterns would be worthless even with
the true one among them, so `tests/test_patterns.py` prints the realized false-discovery rate
alongside the hypothesis count.

The hourly, weekday and seasonal profiles and the zone hotspots are reported too, in a separate
section. They are the context the tested findings were measured against, and they were never part of
the multiple-testing family — folding them into the discovery count would overstate what survived
correction.

## Controlling the obvious confounds

A raw "ZONE-B is busy on Friday evening" is not a discovery; it is three confounds in a trench coat.

**Hour of day and zone level.** Each `(zone, weekday, hour)` cell is compared against *the same zone
and the same hour on other weekdays*. 18:00 is a peak everywhere and `ZONE-A` carries four times
`ZONE-C`, so comparing a cell against the city average would report both as findings.

**The shared weekday effect.** Friday is busier than Tuesday in *every* zone, so a within-zone lift
still is not zone-specific. Each cell's lift is divided by the median lift across zones at that
`(weekday, hour)`. On this dataset the raw ZONE-B figure is **2.379** and the localized one
**2.287** against a planted 2.2 — the adjustment is doing real work, not decoration.

**Seasonality, for level shifts.** A difference-in-differences against the other zones, so a lift
every zone shares cancels. The reference is the **lower quartile** of zones rather than the median,
and that is a correctness fix rather than a preference: see below.

**Rain in its own baseline.** The rain slope is fitted on volumes normalized by their
`(zone, weekday, hour)` median computed from **dry hours only**. Using all hours would build the
effect being measured into the baseline it is measured against and bias the slope toward zero.

Every pattern records what was controlled *and* what was not. The Friday pattern states that
rainfall is not controlled for; the level shifts state the identifying assumption they depend on.

## Multiple testing is not optional

4 zones × 7 weekdays × 24 hours = **672 hypotheses**. At an uncorrected p<0.05, roughly **34 cells**
would clear the bar by chance alone, so an uncorrected scan would report a page of discoveries from
pure noise. **Benjamini-Hochberg at q=0.05** gates the scan, and the test count travels with the
result into both the console output and the JSON report.

The correction is tested against pure noise on its own: 672 uniform p-values must yield zero
discoveries. Without that test, the four findings above would rest on an unverified gate.

Significance is **Mann-Whitney U** — traffic counts are not normal, and a t-test on them would be
confident for the wrong reason. The effect size is a **ratio of medians** with a percentile
bootstrap interval, because the ratio of two medians has no convenient closed form and a
delta-method approximation would understate the uncertainty on exactly the small cells that matter.

## The level-shift reference, and why the median failed

The planted drift lifts `ZONE-A` and `ZONE-B` by ×1.18 from 2026-03-01 while `ZONE-C` and `ZONE-D`
stay flat. With a **median-across-zones** reference, the reference sat *between* the shifted and
unshifted groups, and the scan:

- credited **+8%** to `ZONE-A`, which had genuinely shifted, as only +8% rather than +18%
- credited **−8%** to `ZONE-D`, which had not shifted at all
- reported a real 18% change as 8%

Two of four units being treated is enough to contaminate the median. A **lower-quartile** reference
leans on the assumption that at least some zones are unaffected, which is weaker and is stated on
every level-shift pattern: *a change affecting every zone equally cannot be told from seasonality
without an external reference.* That limitation is real and is recorded rather than papered over.

## Patterns evolve, so they carry history (SPEC section 22)

Each pattern stores `pattern_id`, `first_detected`, `last_observed`, `frequency_per_week`,
`confidence`, `historical_strength`, `recent_strength`, `strength_by_window` and `status`.

Strength is recomputed **per calendar month**, which is what makes evolution measurable rather than
asserted:

```
LVL-ZONE-A-2026-03
  2025-10 2.46   2025-11 2.34   2025-12 2.50   2026-01 2.44   2026-02 2.45
  2026-03 2.82   2026-04 2.90   2026-05 2.82   2026-06 2.81
  Status: STRENGTHENING — recent 2.815 against historical 2.461 (+14.4%),
          significant upward trend (Spearman +0.67, p=0.0499)
```

| condition | status |
|---|---|
| fewer than three measurable windows | `INSUFFICIENT_HISTORY` |
| first appears in the recent windows | `NEW` |
| present historically, absent recently | `DISAPPEARED` |
| direction reverses between windows | `CONFLICTING` |
| recent > historical **and** a significant positive trend | `STRENGTHENING` |
| recent < historical **and** a significant negative trend | `WEAKENING` |
| otherwise | `STABLE` |

`STRENGTHENING` and `WEAKENING` require a **significant Spearman trend**, not just a higher last
window. A status that flipped on one noisy month would carry no information.

Windows too sparse to measure are recorded as unmeasured, with their sample counts. Interpolating
them would manufacture the very trend being looked for.

Three caveats worth stating:

- **Evolution is not tracked for the descriptive profiles.** A peak-to-trough ratio for March is
  computable, but it moves with which hours happened to be observed. They report that reason
  explicitly rather than `insufficient_history`, which would wrongly suggest more data would help.
- **`ZONE-B`'s level shift classifies `STABLE`, not `STRENGTHENING`.** The per-window measure
  compares a zone against the others, and the other drifted zone rose too, which dilutes the trend.
  The `LEVEL_SHIFT` *kind* carries the finding; the status is the weaker signal.
- A strengthening pattern can reflect changing sensor coverage rather than a changing city — see
  [limitations](limitations.md#analytical-limitations).

## Interaction is tested, never assumed (SPEC section 23)

`Friday → high traffic` and `rain → high traffic` do **not** imply `Friday + rain → even higher`.
The joint effect may be sub-additive (people already avoid Friday evenings), additive, or
super-additive.

```
Friday x rain at ZONE-B + 18:00-20:00 [traffic_volume]: NO_EVIDENCE
  scale: log(value) — effects composing multiplicatively show no interaction
  interaction: -0.0635 (-0.1966 to +0.0696)
  cells: both=16, first_only=62, second_only=81, neither=383
  note: absence of evidence, not evidence of absence
```

Three things this output does deliberately:

**It names the scale.** "No interaction" is meaningless until you say on what scale. Effects that
compose multiplicatively show exactly zero interaction in logs and a large one on the raw scale, and
the generator composes multiplicatively — so the log scale is where the planted truth is zero.

**It reports the verdict from the interval, not the point estimate.** The point estimate is −0.064
when the truth is 0. A test on the point estimate would report a spurious *negative* interaction,
which is precisely the overclaiming SPEC section 38 warns about.

**It states the power.** With **16 wet Friday evenings**, a real interaction of moderate size would
also fail to reach significance. `NO_EVIDENCE` therefore means absence of evidence; `UNTESTABLE` is
a fourth verdict for when even that cannot be said.

`CONFLICTING` status comes from the same instinct: patterns that make incompatible predictions under
the same conditions are surfaced rather than averaged.

## Anomaly detection (SPEC section 24)

> Do not automatically classify an anomaly as an error.

Detection **reuses Phase 2's robust core** rather than reimplementing median/MAD. One definition of
"unusual" in the codebase means two stages cannot disagree about what counts — and the existing
conditioning on `(zone, metric, weekday, hour)` is already what keeps a recurring peak from looking
anomalous. Within its own weekday-and-hour group, the Friday peak **is** the norm: 0 of its 78 rows
are flagged, despite being ×2.2 above the zone's usual volume.

Each anomaly carries ranked **candidate** explanations with their evidence:

| candidate | evidence looked for |
|---|---|
| `RAINFALL` | rain was falling in this hour |
| `RECURRING_PATTERN` | a discovered pattern covers this zone, weekday and hour |
| `LEVEL_SHIFT` | the reading sits after a detected change-point |
| `SENSOR_SUSPECT` | adjacent readings repeat this exact value, or source health is not OK |
| `UNEXPLAINED` | nothing above fired — *stated*, not hidden |

```
REC-014111 ZONE-D traffic_volume 2025-12-05T01:30:00+00:00
  observed 316 against an expected 3,031 (8.5 MAD-sigma, n=39)
  candidates (not a verdict):
    - sensor_suspect (0.75): 6 adjacent readings carry this exact value
    - sensor_suspect (0.70): the reporting source is failed
```

`is_error` is `None` on every record, on a frozen dataclass with no setter, and `as_dict()` writes it
out explicitly with a note so a reader can see the system *declined* to judge rather than forgot to.
SPEC section 24's own candidate list holds a sensor failure, a festival and an accident, and the
number alone cannot separate them. Guessing would discard real events as glitches, which is the
expensive direction to be wrong in.

Support scores rank candidates; they are not probabilities and do not sum to one, because the
candidates are not mutually exclusive.

### Scoring the detector honestly

**No planted point anomaly survives into the unified layer.** Phase 2 quarantines all 9 invalid
values and all 60 frozen readings before discovery ever runs, which is correct pipeline behaviour but
makes the unified layer the wrong place to measure recall. `pre_quarantine_population()` rebuilds the
population Phase 2 saw. It is a measurement tool, not a pipeline stage — **discovery itself still
reads the unified layer only**.

On that population: **54 flagged of 26,233** (0.21%), of which

- **4 of 4** planted negative volumes
- **39 of 60** frozen readings
- **0 of 78** recurring Friday peak rows

The frozen recall is asserted as a **floor, not perfection**, and the reason is recorded: the sensor
froze at 316 vehicles/hour, which is implausible at 09:00 and unremarkable at 03:00. The 21 misses
are the overnight hours, where a value-distribution detector genuinely cannot tell the reading from
real data. Phase 2's frozen-run detector is what catches those, by **repetition** rather than
magnitude — two detectors with different blind spots, which is the point of having both.

## What discovery never does

- **It never reads `ground_truth.json`.** No module under `src/` imports or names that file, and an
  AST-based source-tree test enforces it. Only the test suite opens it.
- **It never imputes.** Unknown readings are dropped from a comparison; cells with too little data
  are reported as untested rather than filled.
- **It never touches the quarantined, duplicate-linked or conflicted buckets.** `ObservationFrame`
  reads `result.unified` only, the same structural guarantee as the Phase 4 feature builder.
- **It never retrains or adapts a model.** A level shift is a change in the city, not a data error;
  deciding what a model should do about it is Phase 6–7.

## Determinism

The bootstrap is seeded, so two runs produce the same pattern ids, strengths, statuses and interaction
verdicts. That is a requirement rather than a nicety: evolution cannot be tracked across runs that
disagree with each other.

Reports are written to `reports/patterns/patterns-<timestamp>.json`, one file per run, appended never
overwritten — tracking evolution means comparing what was believed then against what is believed
now, and overwriting destroys the earlier belief.

## Relationship to prediction

Patterns inform features and explanations, and they are inspectable in their own right. A pattern is
descriptive evidence — a recurring co-occurrence — with its evidence attached: the test used, the
statistic, the raw and adjusted p-values, an effect size with a confidence interval, the sample sizes
and the controls. A strength without its interval cannot be judged, so the two are never stored
apart.

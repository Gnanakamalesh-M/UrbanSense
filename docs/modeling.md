# Modelling

> Status: **implemented** (Phases 4-6). The leakage rules below are enforced mechanically by
> `features/leakage.py`, which probes behaviour rather than trusting declarations. The prediction
> output, calibration, explainability and error analysis are documented in
> [prediction.md](prediction.md).

## What is predicted (SPEC section 25)

For each problem — traffic only in v1 — the engine predicts probability, severity, expected time
window, location and confidence:

```
Traffic Risk: 87%   Zone: B   Expected: 18:00-20:00   Severity: High   Confidence: 89%
```

## Source-aware predictions (SPEC section 26)

Every prediction records which sources contributed and how fresh they were. A forecast resting
mostly on a six-hour-old document is a different object from one resting on live sensors, and the
user needs to be able to tell them apart.

## Leakage prevention (SPEC section 33)

**Mandatory, and the reason temporal discipline runs through the whole project.**

When predicting Monday 18:00, the model must not use Monday 20:00 actual traffic — nor any value
whose `event_time` is at or after the target, nor any aggregate computed over a window that includes
it.

Three enforcement layers:

1. **Schema** (exists today): an `event_time` in the future is rejected outside quarantine.
2. **Features** (Phase 3): each feature builder declares its lookback window, so an accidental
   forward-looking window is a mechanical failure rather than a review catch.
3. **Evaluation** (Phase 5): **temporal splits only.** Train on the past, test on the future. No
   random shuffling, no k-fold over time, and `temporal.min_prediction_gap_minutes` enforces a gap
   between the newest feature observation and the target.

A model that scores well because it saw the answer is worse than no model, because it is
convincing.

## Baselines first

Before any adaptive machinery, a static baseline is trained and reported — seasonal-naive and a
simple regressor. The project's central experiment is "does adaptation actually improve on static?"
(SPEC section 38), and that question needs an honest static number to be meaningful.

## Model versioning (SPEC sections 31, 32)

Every version records training period, features, algorithm, hyperparameters, metrics, dataset
version, creation timestamp and status: `champion`, `challenger`, `rejected` or `archived`.

Rejected candidates are kept. Which updates failed evaluation, and why, is part of the experimental
record — see [`drift.md`](drift.md) for the promotion gate.

## Explainability (SPEC section 27)

Major predictions report contributing factors via feature importance and SHAP-style attributions:

```
Traffic Risk: 84%
  Historical traffic   38%
  Rainfall             24%
  Peak hour            19%
  Friday               11%
  Local event           8%
```

These are **contributions to the model's output**, not causes in the world. The distinction is kept
in the wording of every user-facing explanation.

---

## What Phase 4 implemented

Features over the unified layer, temporal splits with gaps, two naive baselines, a gradient model,
and the model registry. No drift detection, no adaptation, no probability/severity output yet.

### Model choice: scikit-learn `HistGradientBoostingRegressor`

Chosen over LightGBM for one decisive reason — **it handles `NaN` natively**, learning a split
direction for missing values. That is what lets "missing stays missing" hold all the way to the
model. The alternative was an imputer, and filling 3.5% of lag values with a column mean would teach
the model that a sensor outage looks like an average hour. It is also deterministic under
`random_state`, so the committed registry metrics are reproducible, and it needs no platform binary.

### No pandas, and that is a correctness decision

Lags are looked up **by timestamp**, not by row position. Quarantining the frozen ZONE-D window left
a **2 day 13 hour hole** in that series, and ZONE-A has four two-hour holes. A positional
`.shift(1)` across such a hole returns the value from two and a half days earlier and labels it
`lag_1h` — silently wrong, entirely plausible-looking, and undetectable downstream.

Rolling means use cached prefix sums for speed (9.3s → 1.25s per feature table), but
`window_mean_exact` remains the definition and a test asserts the two agree across 2,250 checks.

### Leakage prevention: two independent mechanisms

**Declared.** Every `FeatureSpec` states its backward offset, and a negative one raises at
construction. A forward-looking feature cannot be declared through the normal path.

**Probed.** `detect_leakage()` perturbs every observation after `t` (multiply *and* shift, so the
change cannot cancel at zero), rebuilds the features, and flags any value that moved. This catches
leakage however the feature was written — the test suite plants a `lead_1h` reading `t + 1h` by
bypassing `__init__`, exactly as a careless hand-rolled builder would, and requires it to be caught.
A checker that never fires proves nothing.

Two clauses, honestly reported:

| clause | status on this data |
|---|---|
| `event_time <= t` | **enforced** — probed on 19 features × 24 sample times |
| `received_time <= t` | **not applicable** — one receipt instant (a batch load) |

The receipt clause is recorded as not applicable rather than silently skipped: after a batch load
every record shares the instant the file was opened, so enforcing it would reject the entire dataset.
When receipts *are* distinct (a live or replayed feed) the same perturbation technique probes it for
real, and a test exercises that path with staggered receipt times.

### Temporal splits

`temporal_split()` has **no seed parameter anywhere**. A shuffled split of a time series is not a
weaker evaluation but a meaningless one: with `lag_1h` as a feature it puts hour 14 in train and hour
15 in test, so the model predicts what it has already seen and scores beautifully.

```
train:      2025-09-30 .. 2026-03-13   15,492 rows
            <-- 168h gap, 1,324 rows excluded -->
validation: 2026-03-20 .. 2026-05-07    4,545 rows
            <-- 168h gap -->
test:       2026-05-14 .. 2026-06-30    4,526 rows
```

The gap is 168h — the furthest any feature reaches — applied at **both** boundaries.

### Results, test split, traffic_volume at +1h

| model | MAE | RMSE | relative MAE |
|---|---|---|---|
| `naive_last` | 972.4 | 1,454.5 | 34.1% |
| `same_hour_last_week` | 589.9 | 923.9 | 20.7% |
| **`gradient_boosting`** | **284.9** | **508.9** | **9.5%** |

70.7% better than naive overall, and better in every zone. On the July holdout — a month nothing
trained on — overall MAE is **295.2**, only 3.6% worse than the test split, so the split was barely
flattering.

### Where it is weak

Stated plainly, because an average hides it:

- **The Friday ZONE-B peak is the hardest thing in the dataset.** MAE there is 4,194 against an
  overall 284.9. In relative terms it is 3.07× the overall figure, and on validation it was 1.7× —
  so the model degrades on that slice between validation and test. The planted pattern appears in
  only **47 training rows** (0.3%), which is the root cause and is not fixable by tuning.
- The model *is* recovering most of the spike: ignoring the pattern entirely would give MAE 10,282
  there, so it is 59% better than that counterfactual. That is the test's actual assertion.
- **On the holdout, `ZONE-C Fri 18-20` is worse than naive** (256.8 vs 224.3). One slice out of
  eight, on 9 rows, but it is a real loss and the report prints it.
- **Training straddles the drift.** The planted +18% shift at 2026-03-01 sits at 55% of the series,
  so train is mostly pre-drift and validation/test are entirely post-drift. The lag features carry
  the new level so the model adapts through its inputs; Phase 7 detects the shift explicitly
  and decides whether to retrain -- see [drift.md](drift.md).

### Two tuning decisions, and where they were made

Both on the **validation** segment. The test segment was evaluated once.

- `loss="squared_error"` rather than `absolute_error`: 4% worse overall MAE (258 vs 248) for **48%
  better** on the congestion peak (2,095 vs 4,004). `absolute_error` optimizes the median, and a
  spike in 0.3% of rows is not the median. For a congestion forecast the peak is the point.
- `min_samples_leaf=20` rather than the usual 40: with only 47 training examples of the pattern, a
  leaf requiring 40 samples can never isolate "ZONE-B, Friday, hour 18" and the spike is averaged
  away entirely.

A third change was a *feature* fix, not a hyperparameter: `lag_168h` is anchored to the sample time,
so at a 1h horizon it reads last week's 17:00 when the target is 18:00 — one hour off a sharp peak,
which is why `same_hour_last_week` initially beat the model on exactly that slice. Adding
target-anchored `seasonal_lags` cut the slice error from 6,840 to 2,095. It is a general seasonal
lag, not a hand-coded "is this the Friday pattern" flag; writing that would be planting the answer.

### Uncertainty

The interval is the 10th/90th percentile of **validation** residuals, per zone. Cheap, and honest
about the typical spread — but constant within a zone, which is wrong in a known direction: traffic
error scales with volume, so the band is too wide overnight and too narrow at peak. A calibrated
interval needs quantile regression. The registry records this limitation in the entry itself.

# Prediction output, explainability and error intelligence

> Status: **implemented (Phase 6)** for traffic. Calibrated congestion probabilities, severity
> bands, congestion windows, source attribution, per-prediction factors, an append-only outcome
> store and the error breakdown.

Phase 4 produced a **number**: predicted `traffic_volume` at +1h with an MAE. SPEC section 25 asks
for something a person can act on, and sections 26–28 ask for the three things that make such a
claim auditable — which sources fed it, which factors contributed to it, and how wrong it turned
out to be.

```bash
python scripts/predict_demo.py --zone ZONE-B --date 2026-06-26
python scripts/error_report.py --dataset both
```

## What one prediction looks like

```
ZONE-B  2026-06-26 18:00 local
  congestion probability   99.1%  (calibrated)
  severity                severe  (x2.03 the zone threshold of 4,652)
  predicted volume         9,455 [8,712 to 9,717]
  interval tightness       89.4%  (how precise the volume is, not a chance)

contributing factors (shap_tree, additive):
  historical traffic    84.4%  contributed +6,027 upward to the prediction
  peak hour              6.6%  contributed   +469 upward to the prediction
  location               4.4%  contributed   +313 upward to the prediction
  weekday                2.3%  contributed   +167 upward to the prediction
  baseline 2,312 plus the contributions above reconstructs the predicted 9,455
  these are associations the model learned from past data, not causes

contributing sources (share of input observations, not of influence):
  historical     100.0%
  freshness: newest reading 0.0h before prediction time (current)
```

## Congestion is a config decision, not a fact

"Congested" is policy: a six-lane arterial and a residential collector do not become problems at the
same vehicle count. So it lives in `configs/problems/traffic.yaml` — the metric, the per-zone
percentile, and the severity bands as ratios to the threshold.

**The threshold is fitted on the training segment only**, and that is enforced by the signature
rather than by discipline: `fit_thresholds()` takes one `FeatureTable` and has no parameter through
which validation or test data could arrive. The test inflates validation and test tenfold and
requires the fitted thresholds to be **bit-identical**; a companion test concatenates the splits and
requires that they *do* move, so the first test cannot pass by tautology.

| zone | threshold (90th pct of train) |
|---|---|
| ZONE-A | 5,900 |
| ZONE-B | 4,652 |
| ZONE-C | 1,605 |
| ZONE-D | 3,691 |

### The definition ages, on purpose

| split | congested rate |
|---|---|
| train | 0.100 |
| validation | **0.234** |
| test | **0.281** |

The city gets busier — the planted ×1.18 drift plus seasonality — so a definition anchored to the
past marks more of the present as congested. This looks like a bug and is not. The alternative,
re-fitting per period, would let the test split set its own pass mark.

Severity bands are validated at load as strictly ascending, starting at zero, with names ordered to
match. Monotonic severity is therefore a property of the config loader, not something the classifier
has to remember to preserve.

## Probability, and whether it can be believed

`P(volume > threshold) = sf((threshold − prediction) / σ)` with σ the spread of the **training**
residuals (158 on this data), then mapped through an isotonic regression fitted on **validation**.

A separate classifier was built, measured and rejected:

| approach | Brier validation | Brier test |
|---|---|---|
| **residual tail, isotonic-recalibrated** | 0.0667 | **0.0745** |
| separate `HistGradientBoostingClassifier` | 0.0564 | 0.0920 |
| base-rate-only reference | 0.1791 | 0.2042 |

The classifier is better on validation and much worse on test: it learned the training period's 10%
congestion rate when the real rate is 28% by then. Choosing it would also have meant selecting on
the very split the calibrator is fitted on. The residual approach has a second advantage — the
probability and the predicted volume come from one model, so they cannot disagree, which is what
makes "severity monotonic in predicted volume" true by construction.

### The reliability table is not optional

A Brier score alone says nothing about whether a stated 70% means 70%, so `describe()` always prints
the scores *and* the table, with the base-rate reference beside them.

Two labels do real work here:

- **The validation table is flagged in-sample.** Its gaps are zero because the isotonic map was
  fitted on exactly those rows. That is arithmetic, not evidence, and the output says so.
- **The test table is flagged under-confident, with the cause quantified.** Fitted where congestion
  occurs 23.4% of the time and applied where it occurs 28.1%, every bin's gap is positive:

```
calibration on test (4,526 rows, base rate 0.281)
  brier  raw 0.0833  calibrated 0.0745  base-rate-only 0.2042
  expected calibration error 0.0294  (under-confident)
  bin        n   predicted   observed       gap
  0.0-0.1   2797      0.010      0.024    +0.014
  0.4-0.5    137      0.418      0.577    +0.158
  0.9-1.0    375      0.982      0.979    -0.003
```

So: the probabilities are calibrated on validation, **beat the base rate substantially on test**,
and are **systematically under-confident there**. All three are true and all three are printed.

### Probabilities never say 0% or 100%

Isotonic regression maps a bin where nothing happened to exactly zero, which reads as *impossible* —
and the test table shows those hours congest 2.4% of the time. Observing no events in `n` rows
bounds the rate at about `3/n`, so that is the floor and `1 −` it the ceiling. Quiet hours read
0.1%, not 0.0%.

### Confidence is a width, not a second probability

SPEC section 25's example prints "Confidence: 89%" beside a risk of 87%, inviting both to be read as
likelihoods when only one is calibrated. Here `confidence` is `1 − width/prediction` and is labelled
as interval tightness everywhere it renders.

## Congestion windows, and what they do not mean

Consecutive congested hours merge into windows, breaking at a gap rather than bridging it.

**A window is not a Friday detector.** On the test split, 14 of 14 ZONE-B Friday evening hours clear
the threshold — and so do **65 of 82 other-weekday evening hours**, because the threshold is
anchored to a quieter training period. On the planted Friday the window actually runs 13:00–20:00.

What distinguishes the planted peak is **severity**: ×2.46 the threshold against ×1.98 for ordinary
evenings. The test asserts the window is found *and* that Friday's severity exceeds a non-Friday
evening's, rather than pretending Friday is unique.

## Source-aware output (SPEC section 26)

For each prediction, the sources behind the observations its declared feature windows actually read,
plus the age of the newest one.

**The percentages are shares of input observations, not shares of influence.** A single rainfall
reading could move a prediction more than fifty traffic lags and still be 2% of the input count. The
explanation layer answers the influence question; this one answers where the inputs came from, and
every rendering says which.

The demo feed is historical only, so the honest output is **`historical 100%`** with the reason
stated. Section 26's example shows a 31% real-time share; printing one to match would be inventing
evidence. A hand-built test covers the multi-source path (historical + real-time + document, shares
summing to 100%), because the shipped data would otherwise leave it unexercised. Feature windows
that found no data are reported too — a prediction resting on fewer inputs is a weaker one.

## Explainability (SPEC section 27)

**SHAP works here.** `shap.TreeExplainer` does support `HistGradientBoostingRegressor` as of shap
0.52, handles the `NaN` values this pipeline deliberately preserves, and is exactly additive —
measured on the real model, the largest gap between `base value + contributions` and the prediction
was **0.000000** across 200 rows. That additivity is why it is preferred: the contributions
decompose the prediction rather than merely ranking the inputs.

It remains an **optional** dependency (`pip install -e ".[explain]"`). shap 0.52 ships a cp312-abi3
wheel, so this machine (3.14) and CI (3.11) cannot be guaranteed the same version, and pinning one
would break the other. The explainer is pluggable:

| available | method | additive |
|---|---|---|
| shap, able to parse the estimator | exact Shapley values | yes |
| otherwise | NaN-occlusion: blank one input, re-predict | **no** |

Occlusion is a genuine "without this input" counterfactual rather than an imputed stand-in, because
the estimator handles `NaN` natively. Which method ran is recorded on every explanation and in the
registry, and `is_additive` says whether the numbers decompose the prediction or only rank it —
presenting occlusion deltas as a decomposition would be a quiet lie, since they sum to nothing in
particular. CI exercises the fallback; this machine exercises SHAP.

### Factors, not columns

Nineteen columns group into six readable factors — historical traffic, rainfall, peak hour, weekday,
season, location — and the mapping lives in config so that adding a feature forces a decision about
how to explain it. The loader **rejects a feature that matches no group**, and one that matches two:
a silently ungrouped feature would move every prediction while appearing in no explanation, which is
the most dangerous way this layer can be wrong.

Global importance (permutation, on validation) reads:

```
historical traffic    85.6%     weekday          3.7%
peak hour              8.1%     location         2.3%
rainfall               0.6%     season           0.0%
```

Historical traffic dominates because the seasonal lags encode the Friday pattern directly. That is
honest rather than flattering, and it is printed as it comes.

### Nothing claims causation

The vocabulary is fixed — *contributed to the prediction*, *associated with*, *the model weighted* —
and a test greps every rendered string for causal verbs. A style note in a docstring does not
survive a hurried edit; a failing test does. The grep strips disclaimers first, because "not causes"
is the opposite of a causal claim.

The leakage probe runs on the feature set *and* the explanation inputs. An explanation of a
prediction that saw the future is more dangerous than no explanation, because it is convincing —
`predict_demo.py` exits non-zero rather than explaining one.

## Outcome tracking and error intelligence (SPEC section 28)

Every prediction/actual pair is appended to `reports/outcomes/outcomes.jsonl` as one line carrying
prediction, actual, error, location, time, conditions and model_version. **Append-only**: the file
is opened in `"a"` mode, so existing rows cannot be truncated even by a caller who meant to replace
them. A prediction's record is what the model said at the time, and rewriting it to match what
happened would destroy the only evidence it was ever wrong.

**Conditions are recorded at target time — the rainfall during the hour predicted, which the model
never saw.** That is not leakage: it is an attribute of the outcome, written after the fact and
never read back as an input. Recording the conditions the model *did* see would leave the error
analysis unable to answer the only question it exists for.

An hour with no gauge reading records `None` and bands as `unknown`, never `dry`. Treating a missing
reading as zero rain would move those rows into the dry bucket and flatter the dry-weather numbers.

### Where the model is weak

The breakdowns print **before** the overall figure, because an aggregate MAE averages over
conditions that behave nothing like each other and a reader who sees it first will anchor on it.

Test split, 4,526 outcomes:

| slice | rows | MAE | relative |
|---|---|---|---|
| **overall** | 4,526 | 284.9 | 10.0% |
| dry | 3,302 | 272.3 | 9.7% |
| light rain | 1,192 | 306.6 | 10.4% |
| **heavy rain** | 32 | **770.4** | **17.2%** |
| night | 1,109 | 64.5 | 10.4% |
| evening | 1,327 | 339.6 | 10.6% |
| **Friday 18:00–20:00** | 55 | **1,318.0** | **19.0%** |
| all other hours | 4,471 | 272.2 | 9.7% |
| predicted *severe* | 17 | **3,912.9** | **31.5%** (thin) |

Four things this says plainly:

- **Heavy rain is 1.8× harder** than dry weather, which is SPEC section 28's own example.
- **The Friday peak is the weakest real slice** at 19.0% against 10.0% overall.
- **Predictions the model labels "severe" are the worst of all** at 31.5%. A model that is accurate
  when it says "low" and wild when it says "severe" is more dangerous than its average suggests,
  which is why severity is one of the cuts — but only 17 rows, so it is flagged thin.
- The **July holdout** (a month nothing trained on) gives 295.2 overall against 284.9 on test, so
  the test split was barely flattering. The Friday window is worse there: 23.0%.

Slices under 30 rows are excluded from the weakest-slices ranking — a nine-row slice topping the
list would be noise presented as a finding — but they still appear in their own tables. Empty groups
are omitted rather than shown as zeros: "we never predicted in heavy rain" and "we predicted
perfectly in heavy rain" are opposite findings.

Error maths is not reimplemented. Every cell goes through Phase 4's `evaluate_slice`, and the tests
check the numbers against means computed by hand on a small fixture rather than against another run
of the same code.

## Model registry

`Traffic-v1.1` is Champion; `Traffic-v1.0` is `ARCHIVED`, so `champions()` returns exactly one. v1.0
keeps its metrics on disk — knowing which model served, and how well, is the experimental record.

v1.1 adds four metadata blocks, each recording what a reader needs in order to disbelieve the model
if it deserves it: `congestion` (definition, thresholds, fitting segment), `calibration` (method,
both reliability tables, all three Brier scores, the in-sample and under-confident flags),
`explainability` (method, additivity, why, global importance) and `error_analysis` (the full
breakdown and the weakest slices).

## What Phase 6 does not do

- **No drift detection or adaptation.** The base-rate shift from 0.100 to 0.281 is *reported* here,
  not adapted to. Phase 7.
- No champion/challenger promotion logic, no intervention recommendations, no API or dashboard.
- No imputation anywhere.
- The Phase 4 model's architecture and hyperparameters are untouched; it is refitted, not retuned.

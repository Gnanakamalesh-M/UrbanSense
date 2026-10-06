# Drift detection and adaptive learning

> Status: **implemented (Phase 7)** for traffic. Rolling drift monitoring, candidate updates,
> challenger training, config-declared promotion rules, champion/challenger versioning, and a
> replay harness comparing a static model against the adaptive system on one stream.

The dataset carries a planted level shift at **2026-03-01** — ×1.18 in `ZONE-A` and `ZONE-B`, with
`ZONE-C` and `ZONE-D` as controls. Every earlier phase either trained across it or merely reported
it. This phase builds the loop that notices it and decides what to do:

```
Drift detected -> Investigate -> Candidate update -> Train challenger
  -> Evaluate -> Compare with champion -> Promote, or REJECT and keep the champion
```

```bash
python scripts/replay_adaptive.py
python scripts/replay_adaptive.py --scenario july_holdout
```

## Which detector can be trusted — the phase's main finding

SPEC section 29 names three families, and the obvious expectation is that feature drift is the
primary signal. On this data that is wrong, and the way to find out was to run the **same monitor
against a drift-free copy of the dataset** — same seed, the level shift simply switched off:

| family | drifted stream | drift-free stream |
|---|---|---|
| feature drift | 16 firings | 15 firings |
| performance drift | 16 firings | 10 firings |
| **prediction drift** | **15 firings** | **0 firings** |

Feature and performance drift fire on *both* streams. The generator has annual seasonality, so a
champion trained on autumn genuinely sees different inputs and is genuinely worse by spring **with
nothing having changed in the city**. A detector that cannot tell those apart would retrain every
March.

Only **prediction drift** separates them, and the reason is instructive: the model's *output*
distribution moves when the input-output relationship changes in a way the model actually responds
to, while seasonal variation it was trained to handle moves its inputs without moving its answers.

So `candidate_kinds` in `configs/adaptation/traffic.yaml` gates candidate creation on prediction
drift alone. The other two are still computed, logged and reported — **performance drift fires
first, four days after the change against prediction drift's eleven** — so they are genuinely useful
leading indicators. They are simply not grounds to retrain on their own.

The cost of that specificity is a week of delay, and both delays are printed rather than only the
flattering one.

## Two bugs the drift-free comparison caught

Neither would have announced itself. Both are recorded here because the fix in each case was
structural rather than a tuned threshold.

**1. The reference window must not overlap the training period.** An overlapping reference is partly
in-sample, so the reference error is understated and every later window looks drifted. My first
version did exactly this, and the drift-free dataset produced **17 false alarms** — reference MAE
140 against an honest 180. With a strictly out-of-sample reference the gate *tightens* from 1.94 to
1.15 **and** the false alarms go to zero, so the honest reference is also the more sensitive one.
`DriftMonitor` now takes `training_end` and refuses an overlapping reference at construction,
because by the time false alarms appear the cause is a long way away.

**2. A feature averaged over at least the monitoring window cannot be monitored with it.** I chose
the watched features by reasoning ("rolling means concentrate a level shift") and measured only two
of them. Measuring all ten:

| feature | pre-drift max PSI | post-drift | verdict |
|---|---|---|---|
| `roll_mean_24h` | 0.125 | 0.375 – 1.26 | **usable** |
| `roll_mean_168h` | **4.995** | 3.65 – 8.98 | unusable |
| `lag_1h`, `lag_24h` | 0.013 | 0.018 – 0.034 | too weak |
| `roll_mean_3h` | 0.013 | 0.026 – 0.044 | too weak |
| `rainfall_0h` | **1.326** | 0.003 | unusable |

`roll_mean_168h` is a one-week mean compared across one-week windows, so within any window it is
near-constant — roughly one independent value per zone. The reference quantile edges pack together,
mass moves wholesale between bins, and PSI reads 5 or 6 on perfectly stable data.
`MonitorConfig.unmonitorable()` therefore rejects any feature whose reach is at least the window, so
a config change cannot reintroduce the false alarms. Rainfall is excluded as zero-inflated: its
weekly PSI swings to 1.3 with nothing wrong while the planted shift moves it 0.003. The point lags
carry a real but tiny signal, far below the published gate, so watching them adds noise without ever
firing.

## How the gates are set

Every gate is calibrated on the **reference period only** and then frozen. Nothing from the
monitored stretch feeds back into a gate, so a detection can never be a consequence of the gate
moving.

- **Feature drift** uses the published PSI convention — 0.1 minor, **0.25 major** — rather than a
  fitted number, so the result can be stated without reference to this dataset. 0.1 would
  false-alarm (the stable stretch reaches 0.117), so 0.25 is what may act.
- **Prediction and performance drift** use `reference mean + 3 sd` of the statistic's own
  week-to-week values within the reference period.
- A detector with fewer than four reference windows reports **`INSUFFICIENT_REFERENCE`**, not
  silence. A three-sigma gate estimated from two numbers is not a gate, and reporting that as "no
  drift" turns absence of evidence into evidence of absence.
- The performance gate has a **floor of 1.0**: a window that is *better* than the reference is not
  drift worth retraining for.

The KS test is reported alongside PSI but is deliberately **not** the gate. On a 650-row weekly
window almost any difference is "significant" — a 0.4% mean shift on 20,000 rows gives p = 9e-3 —
so the D statistic is what decides and the p-value travels as context.

## Episodes and the cooldown

The planted shift keeps the detectors above their gates for every one of the fifteen weeks that
follow. Fifteen events is accurate; fifteen retrains is not. Consecutive qualifying firings collapse
into one **episode**, a candidate is raised at onset, and further candidates from the same episode
wait for a **28-day cooldown**. A quiet evaluation closes the episode, so a later shift is a new one
rather than a continuation.

On the demo history that turns 15 qualifying firings into **4 candidate updates**.

## A drift event never retrains

`candidate_from_event()` is the only path from a drift event to a fitted model, so "do not blindly
retrain whenever new data arrives" is a property of the call graph rather than a convention. The
`CandidateUpdate` it produces is SPEC section 29's *investigate* arrow made durable: what fired, how
far past its gate, which episode, which decision time, and which champion was incumbent.

Candidates end in one of four states, and the fourth matters:

| status | meaning |
|---|---|
| `PROMOTED` | a challenger passed every rule and became champion |
| `REJECTED` | challengers were judged and none passed; champion kept |
| `INCONCLUSIVE` | **nothing was judged** — too few evaluation rows, usually at the end of a replay |
| `RAISED` | created, not yet decided |

`INCONCLUSIVE` is distinct from `REJECTED` on purpose. Nothing judged and found wanting is not the
same as nothing to judge on.

## A challenger can never see its own evaluation horizon

This is the one mistake that would make every number in the phase meaningless, and it would do so
*flatteringly* — a leaky challenger wins its comparison and gets promoted. Three independent
defences:

1. **`available_at()` filters on the sample time *and* the target time.** Filtering on the sample
   time alone is the subtle wrong answer: a row sampled an hour before the decision carries a target
   from after it, so a challenger "trained up to Thursday" would hold a Friday measurement.
2. **The evaluation window starts a 168-hour gap after the decision time**, so the
   furthest-reaching feature of an evaluation row cannot touch a training row. The tests assert the
   two sets share no `record_id`.
3. **The Phase 4 leakage probe runs against the challengers' feature set**, so a challenger cannot
   inherit a clean bill of health from the champion's check. A challenger that fails it is rejected
   with that as the reason — never promoted, never quietly dropped.

One probe runs per candidate and its verdict is attached to **every** challenger for that candidate.
They share a feature set, so a second probe would compute the same answer at the cost of the
slowest step in the replay — but probing only the *first* window, which an earlier version did, left
the rest with no report at all and `leakage_clean` returning true vacuously. The rule then applied
to one challenger and not its rivals, which is worse than not having it: on the holdout scenario a
false failure on the probed challenger handed four promotions to unprobed ones.

### The receipt clause, and a false accusation

The leakage probe has two clauses. `event_time <= t` is always probed. `received_time <= t` — "had
this record arrived yet?" — can only be probed when the receipts say something about availability,
and deciding when that holds turned out to be subtle.

The first version asked whether more than one distinct receipt instant existed. On a single CSV
there is exactly one (the moment the file was opened), so the clause was correctly reported as
**not applicable**. Merging the history and the holdout gives *two* instants, four seconds apart and
both months after the last event — more than one, so the clause was probed. And then every record
counted as a late arrival, every feature moved, and the probe reported **237 violations** against
feature code that was entirely correct. The same challenger came out clean on one file and leaky on
two.

Counting instants was the wrong question. The right one is whether the receipts **straddle** the
sample time: some records already arrived, some had not. Only then does "was this available yet?"
have two possible answers, and only then can perturbing the late arrivals distinguish a leaky
feature from a clean one. When nothing had arrived, perturbing the late arrivals perturbs the whole
dataset and the clause is unprobeable — which is now what it says, for one file and for two alike.

Narrowing a check invites the question of whether it still catches anything, so two tests answer it.
With receipts staggered to five minutes after each event, the clause reports **violated** and names
`rainfall_0h`, `rainfall_roll_24h`, `roll_mean_24h` and `roll_mean_168h` — the offset-0 features,
which read the value *at* `t` and would not have it under a five-minute reporting lag. With every
reading two days late, it reports violated with 24 findings. The clause bites; it simply no longer
bites the innocent.

## Promotion rules

In `configs/adaptation/traffic.yaml`, because SPEC section 30 says "predefined validation criteria"
and a margin chosen after seeing a challenger's score is a rationalisation rather than a criterion.

```yaml
promotion:
  min_relative_mae_improvement: 0.03
  max_slice_regression: 0.05
  protected_slices: ["ZONE-B Fri 18-20", "ZONE-A Fri 18-20"]
  min_slice_rows: 8
  min_evaluation_rows: 200
  require_leakage_clean: true
```

The **3% margin is derived from reference-period variability**, never from the window a challenger is
judged on: week-to-week MAE on the stable stretch has a coefficient of variation of 3.2%, and a
four-week evaluation window halves that, so 3% sits above the noise floor and well below the 7–15% a
real adaptation produced.

Two details that do real work:

- **Every rule outcome is recorded, passes included.** A decision reporting only its blocking reason
  invites nudging that threshold; "improved 1.80% against a required 3.00%, everything else passed"
  settles it instead.
- **A thin protected slice abstains rather than blocking.** The Friday `ZONE-B` window contributes
  6–10 rows to a four-week evaluation, and a safeguard two observations can swing is a coin toss
  wearing a safeguard's clothes. The slice's numbers are still printed.

The winner is chosen by error **from the passing set only** — a challenger that scores better but
fails a protected-slice rule does not win.

### Two challengers per candidate, and the rules pick

Both an expanding window (everything available at the decision time) and a 60-day recent window are
trained, and the rules choose. Measured at the 2026-03-12 decision point:

| window | MAE | vs champion |
|---|---|---|
| expanding | 236.1 | **−15.4%** |
| recent 60 days | 257.3 | −7.8% |
| recent 30 days | 259.4 | −7.0% |

The textbook "forget the old regime" answer **loses** here, because this drift is a level shift the
lag features already track: nothing needs forgetting, and more data helps the rare Friday pattern
that has few examples either way.

But it does not always lose — and that is the argument for training both. The expanding window
wins **three of four** promotions on `drift_history` and three of five on `july_holdout`; the
60-day window takes the 2026-05-07 decision in both. One promotion in four is a weaker case for
training both than a scenario-level reversal would be, and it is the honest one.

A note on how that figure was arrived at, because an earlier version of this document claimed the
recent window won *every* promotion on the holdout. It did, and for the wrong reason: the leakage
probe was returning a false failure on the merged two-file stream, and the replay probed only the
first training window, so the expanding challenger was rejected on a spurious leakage finding while
its unprobed rival was waved through. Both bugs are fixed — the probe no longer mistakes a second
load instant for late arrivals, and one probe per candidate is now shared across every window so the
rule applies to all of them or none.

## Static versus adaptive

> **Read this with [`experiments.md`](experiments.md).** The comparison below is against a model that
> is never retrained, and Phase 8 added the baseline it was missing: a 28-day calendar retrain.
> That arm matched the adaptive system across five seeds at 21% less training, so the improvement
> reported here is real but it is **not evidence that the drift machinery earns its place**. The
> method and the numbers on this page stand; the conclusion drawn from them does not.

Both arms are scored on **identical rows**, each window predicted by whichever model was champion at
that point in the replay. Scoring the stream with the end-state model would credit it with months it
never served. The static arm is the frozen pre-drift champion — comparing against Phase 6's
`Traffic-v1.1` would be meaningless, since that model trained across the shift.

**`drift_history`** (13,134 rows, 2026-02-12 → 2026-06-30):

| slice | static MAE | adaptive MAE | change |
|---|---|---|---|
| **overall** | 286.0 | **237.1** | **−17.1%** |
| `ZONE-A` (drifted) | 522.0 | 336.6 | −35.5% |
| `ZONE-B` (drifted) | 310.0 | 302.7 | −2.4% |
| `ZONE-C` (control) | 106.9 | 105.4 | −1.3% |
| `ZONE-D` (control) | 200.0 | 199.7 | −0.2% |
| `ZONE-A Fri 18-20` | 1,279.4 | 637.3 | −50.2% |
| `ZONE-B Fri 18-20` | 3,487.2 | 2,026.9 | −41.9% |
| `ZONE-C Fri 18-20` | 148.8 | 152.3 | **+2.4%** |
| `ZONE-D Fri 18-20` | 314.3 | 327.2 | **+4.1%** |

RMSE 515.3 → 396.9. Four candidates, all promoted; nine registry versions.

**The shape of that result is the evidence that adaptation is tracking something real.** The planted
zones improve most and the controls barely move — if adaptation had helped `ZONE-C` and `ZONE-D` as
much as `ZONE-A`, it would be fitting noise rather than following a change. A test asserts exactly
that ordering.

**`july_holdout`** (16,089 rows, carried into the month nothing trained on): overall **−19.6%**,
`ZONE-A` −38.7%, the Friday slices −52.3% and −47.8%, with six candidates — four promoted, one
rejected and one inconclusive at the very end of the data. Two slices got worse, both marginally:
`ZONE-D Fri 18-20` at +1.0% and `ZONE-C Fri 18-20` unchanged at 0.0%. `regressed_slices` is part of
the result rather than a footnote, however small the regressions turn out to be.

## Model versioning (SPEC sections 31, 32)

Versions run `Traffic-v1.2-expanding`, `Traffic-v1.3-recent_60d`, … — the window is in the name
because two challengers exist per candidate and an entry should say which it is without being
opened. Each carries the full section 31 metadata plus an `adaptation` block naming the
`candidate_id`, the `drift_episode_id`, what it `supersedes`, and the decision with every rule
outcome.

```
9 version(s) in the registry:
  Traffic-v1.1-static       archived
  Traffic-v1.2-expanding    archived   supersedes Traffic-v1.1-static   MAE 238.6
  Traffic-v1.2-recent_60d   rejected                                    MAE 252.4
  Traffic-v1.3-expanding    archived   supersedes Traffic-v1.2-expanding MAE 235.4
  Traffic-v1.3-recent_60d   rejected                                    MAE 261.2
  Traffic-v1.4-expanding    rejected                                    MAE 237.6
  Traffic-v1.4-recent_60d   archived   supersedes Traffic-v1.3-expanding MAE 248.1
  Traffic-v1.5-expanding    champion   supersedes Traffic-v1.4-recent_60d MAE 223.9
  Traffic-v1.5-recent_60d   rejected                                    MAE 238.2
  champion: Traffic-v1.5-expanding
```

**Exactly one champion**, checked by `assert_single_champion()` after every mutation rather than only
in the tests. `promote()` archives the incumbent *before* installing the new champion, so a failure
between the two steps leaves **no** champion rather than two: no champion is loud and immediately
detected, two is quiet and corrupts every later decision.

**Rejected challengers stay on disk** with their metrics and the rules that rejected them. Knowing
which updates failed evaluation, and by how much, is the experimental record; deleting them would
leave a registry implying every candidate ever raised was promoted.

A replay writes to a **fresh per-run** registry under
`reports/adaptation/<scenario>/run-<timestamp>/`, not `models/registry/`. A replay is an offline
experiment, and nine challengers per run would bury the model that is actually serving.

The per-run directory is isolation by construction, and it replaced a scenario-scoped one that
inherited state. Running a scenario twice re-seeded the starting champion into a directory that
still held the previous run's champion, and the next promotion raised `2 champions` naming a version
the run had not reached yet. Two things were wrong: the directory was shared, and `promote()`
archived only a named incumbent rather than every other champion. Both are fixed — "make this the
champion" now demotes whatever else holds the title, and each run gets its own directory, so nothing
is deleted to achieve isolation.

## What Phase 7 does not do

- **No online or incremental learning.** Every challenger is a fresh fit.
- **No automatic rollback** after a promotion, and no shadow traffic or bandit allocation.
- **No imputation** anywhere.
- Nothing adapts without a recorded decision. There is no code path from a drift event to a serving
  model that does not pass through a `CandidateUpdate` and the promotion rules.
- Detectors, challengers and promotion logic **never read `ground_truth.json`**. Only the tests do,
  and an AST-based source-tree test enforces it across both new packages.

# Experiments

> Status: **the central experiment is implemented and run** (Phase 8). Three arms — a frozen model,
> calendar retraining, and the adaptive system — replay the same stream and are scored on the same
> rows, across five generator seeds.
>
> **The headline: scheduled retraining matched the adaptive system, at 21% less training.** Phase 7
> reported adaptation beating the frozen model by 17%, which was true and measured against the wrong
> baseline. See [the result](#the-result) and [what this means for Phase 7's claim](#what-this-does-to-phase-7s-claim).

```bash
python scripts/run_experiment.py                                      # 5 seeds, ~12 min
python scripts/run_experiment.py --seeds 20260101 --replay-days 60 --no-write
```

## The question, and why the baseline changed

**Does selective adaptation actually improve predictions?** (SPEC §38.)

Phase 7 answered a narrower question: does the adaptive system beat a model that is never retrained?
It does, by 17%. But that comparison has an obvious gap — the cheapest thing anyone would try before
building drift detection is **retraining on a calendar**. If a 28-day cron job gets the same
accuracy, then the drift detectors, the candidate gating, the challenger evaluation and the
promotion rules are machinery that buys nothing, and a portfolio project that did not check would be
hiding its weakest point.

So Phase 8 puts that baseline in the room.

## The three arms

All three replay the **same** stream and are scored on the **same** rows — not the same number of
rows, the same records in the same order. That holds by construction: the replay loop predicts three
times on the one window slice it already accumulated, rather than each arm walking the stream
itself. A test compares `record_ids`, not counts.

| arm | behaviour | cost |
|---|---|---|
| `static` | the frozen pre-drift champion, never retrained | nothing |
| `scheduled` | retrained every 28 days on the expanding window, **deployed unconditionally** | one fit per retrain |
| `adaptive` | Phase 7's drift → candidate → promotion-rules system | several fits per candidate, most rejected |

The scheduled arm reuses the adaptive arm's machinery exactly: the same `train_challenger`, the same
`available_at` filter on both sample and target time, the same 168-hour gap, the same leakage probe.
The only difference is what happens next — it deploys without asking. Any leakage advantage would
therefore apply to both arms equally, which is what makes the comparison fair rather than merely
close.

## Pre-registration

The parameters were fixed in [`configs/experiments/static_vs_adaptive.yaml`](../configs/experiments/static_vs_adaptive.yaml)
**before the first run** and are copied verbatim into every report:

```yaml
pre_registered: "2026-10-04"
seeds: [20260101, 20260202, 20260303, 20260404, 20260505]
retrain_every_days: 28          # the scheduled arm's calendar
scheduled_window: expanding
train_days: 75                  # the frozen champion's training span
reference_days: 60              # out-of-sample drift reference
```

Promotion margin, drift gates, cooldown and challenger windows are **not** redeclared here. They
come from `configs/adaptation/traffic.yaml`, fixed in Phase 7, and the report records which file it
read. Restating them would invite the two to diverge.

**No parameter was changed after seeing a result on the evaluation stream.** That is the whole point
of writing them down first, and it is also why the verdict below is reported as the pre-registered
rule computes it rather than as the data arguably supports — see the [caveat](#the-tie-rule-cuts-both-ways).

## The result

Five seeds, 13,134 rows scored per arm per seed. Every arm saw identical rows.

| arm | MAE | RMSE | congestion F1 | Brier | retrains | fits | training rows |
|---|---|---|---|---|---|---|---|
| `static` | 292.6 ± 6.3 | 532.5 ± 23.8 | 0.690 ± 0.026 | 0.0960 ± 0.0038 | 0 | 0 | 0 |
| **`scheduled`** | **229.5 ± 4.9** | **388.0 ± 9.5** | **0.806 ± 0.004** | **0.0651 ± 0.0009** | 4.0 | 4.0 | **77,629 ± 54** |
| `adaptive` | 237.5 ± 9.4 | 400.2 ± 14.8 | 0.794 ± 0.007 | 0.0682 ± 0.0020 | 3.2 ± 0.8 | 7.6 ± 0.9 | 97,788 ± 10,496 |

Both retraining arms beat the frozen model decisively — 21.6% lower MAE for scheduled and 18.8% for
adaptive, recall up from 0.589 to roughly 0.78, Brier down by a third. **Retraining matters.** That
part of Phase 7's story survives intact.

What does not survive is the attribution. **Scheduled retraining had the lower MAE, and it got there
on 21% fewer training rows and roughly half the model fits.** The adaptive arm spends its extra
budget training challengers that its own promotion rules then reject: 7.6 fits to deploy 3.2 models.

### Per-seed, because the mean hides this

| seed | static | scheduled | adaptive |
|---|---|---|---|
| 20260101 | 286.0 | **234.9** | 237.1 |
| 20260202 | 290.4 | **224.3** | 228.2 |
| 20260303 | 291.3 | **226.1** | 240.4 |
| 20260404 | 303.0 | **227.8** | 230.0 |
| 20260505 | 292.4 | **234.4** | 251.7 |

**Scheduled won on all five.**

### The tie rule cuts both ways

The pre-registered tie criterion is "a gap smaller than the cross-seed spread is not a finding", and
by that rule this is a **tie**: the 8.0 MAE gap sits inside adaptive's own ±9.4 spread, so the
report says the two arms are *not distinguished* and that choosing between them on these numbers
would be reading noise.

That is the registered verdict and it stands. But reporting honestly means saying what the rule does
not capture: **the direction was consistent on 5 of 5 seeds** — a one-sided sign test gives p = 1/32 ≈ 0.03,
and the cost difference is not noisy at all — scheduled used fewer training rows on every single
seed, with a spread of ±54 against adaptive's ±10,496.

So the fair summary is: *scheduled retraining is at least as accurate as the adaptive system here,
consistently, and unambiguously cheaper.* It is not "adaptive is worse by 8 MAE" — the per-seed
noise does not support that precision. It is also not "they are equal".

Changing the tie rule now, having seen this, would be exactly the tuning the pre-registration exists
to prevent. The rule stays; the caveat is stated.

## Where adaptation did **not** help

This is the section the experiment was built to produce.

- **Overall accuracy.** Adaptive's mean MAE was higher than scheduled's, on every seed.
- **Cost.** 97,788 training rows against 77,629, and 7.6 fits against 4.0, to end up no better.
- **Six of eight slices.** Adaptive was beaten on all four per-zone slices and on `ZONE-A Fri 18-20`
  and `ZONE-C Fri 18-20`. Scheduled was beaten on two.
- **The drifted zone it was built for.** `ZONE-A`, where the level shift was planted, is the clearest
  case *against* the adaptive arm: 367.6 ± 28.1 versus scheduled's 343.3 ± 5.3. Detecting the drift
  and gating a candidate did worse than not looking and retraining anyway.
- **Congestion detection.** F1 0.794 ± 0.007 against 0.806 ± 0.004, driven by recall (0.766 vs
  0.787). Adaptive missed more congested hours.
- **Calibration.** Brier 0.0682 against 0.0651.

Where adaptive **did** come out ahead: `ZONE-B Fri 18-20` (2,097.6 vs 2,235.1) and `ZONE-D Fri 18-20`
(319.5 vs 322.9). Both are thin slices — 8–10 rows per evaluation — and both spreads are wide enough
that neither is worth leaning on.

### Why scheduled wins here, most likely

Not a defect in the detectors. They work: Phase 7 measured zero false alarms on the pre-drift period
and the planted shift is caught. The issue is that **detection is a delay**, and this drift does not
need one. A 28-day calendar retrains four times whether or not anything happened; the adaptive arm
waits for drift to accumulate past a gate, raises a candidate, trains challengers and checks a
promotion margin — and retrains 3.2 times, later. Against a single clean step change that the
expanding window absorbs anyway, caution costs more than it saves.

That reasoning predicts where adaptation *should* pay: drift that arrives between scheduled
retrains, or a regime change large enough that the promotion margin is the thing stopping a bad model
from shipping. **Neither is tested here**, which is a limitation, not a defence.

## What this does to Phase 7's claim

Phase 7's number was 17.1% MAE improvement, adaptive over static. It was computed correctly and it
is not withdrawn. But it answered "is adaptation better than doing nothing?" when the question worth
asking is "is adaptation better than the cheap thing?" — and the answer to the second, on this data,
is no.

[`docs/drift.md`](drift.md) retains the Phase 7 method and its static-versus-adaptive table. It
should be read with this page: the adaptive machinery is correct, tested, and not justified by these
results.

## Reporting rules

**Results are reported as measured.** The report generator ranks arms by measured MAE with none
privileged, and a test feeds it hand-built results in which adaptive loses outright and in which it
ties, asserting the output says so. There is a converse test too — a generator that could only ever
report a tie would be as broken as one that could only report a win. That test is the reason this
page says what it says.

No cherry-picking: five seeds were pre-registered and all five are reported, including the one where
adaptive did worst (20260505, 251.7 against 234.4).

## Reproducibility (SPEC §39)

Every report pins the seeds, the pre-registered config, the adaptation config it read, the dataset
version and the per-seed drift events and candidate decisions behind the numbers. Reports are
written to `reports/experiments/` as a timestamped JSON and markdown pair, appended rather than
overwritten, so a later run can be compared against an earlier conclusion.

```bash
python scripts/run_experiment.py --seeds 20260101 --replay-days 60   # ~3 min, one seed
```

Each seed is generated into a temporary directory and deleted afterwards — regenerable from the
seed, which is recorded. **`data/sample` is never written**: other phases' tests assert against it.

**Nothing in the experiment reads `ground_truth.json`.** Error is measured against observed values;
where the drift was planted is not an input to any arm or any metric. Only tests consult it.

## Limitations

Carried in every report, not only here:

- **Synthetic data.** The drift is a clean multiplicative level shift with a known start date. Real
  drift is messier, and an arm suited to this shape may not transfer.
- **One drift type.** No gradual drift, no seasonal regime change, no sensor replacement. As argued
  above, this is the limitation most likely to be hiding the adaptive arm's actual advantage.
- **Thin protected slices.** The Friday evening windows contribute 8–10 rows per four-week
  evaluation. Their per-slice figures move a lot between seeds and should not be read as precisely
  as the overall numbers.
- **One fixed calibrator.** Fitted once from the shared starting champion and never refitted, so the
  Brier column compares regressors rather than complete serving systems. Recalibrating per retrain
  would be more realistic and would confound the model's contribution with the calibrator's.
- **Five seeds.** A spread from five datasets is itself imprecise. A tie here means "not
  distinguished", not "equal".
- **One schedule.** 28 days, pre-registered. A 7-day or 90-day calendar might land very differently,
  and sweeping it after seeing these results would be the tuning this design forbids.

## Synthetic streaming scenarios (SPEC §37)

Because this is a GitHub-only project, adaptive behaviour is demonstrated against a simulator driven
from historical data:

```
Historical Dataset -> Streaming Simulator -> new observation every N seconds -> UrbanSense
```

Scenarios: normal city, heavy rainfall, festival, road closure, sudden traffic change, sensor
failure, missing data, distribution drift.

These test the **mechanism** — does the system notice the drift, does it gate the candidate
correctly, does it refuse to read a failed sensor as zero. They are evidence about behaviour under
conditions we constructed, not a forecast of accuracy on a real city's data. See
[`limitations.md`](limitations.md).

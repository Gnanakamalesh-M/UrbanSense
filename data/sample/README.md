# `data/sample/` — tracked in git

The generated demo dataset, committed so the repository is demonstrable
immediately on clone (SPEC section 48) with no downloads and no network.

| File | What it is |
|---|---|
| `traffic_history.csv` | ~59k hourly observations, Oct 2025 – Jun 2026, 4 zones + a city rain gauge |
| `traffic_holdout.csv` | ~6.7k observations for Jul 2026, replayed by the real-time simulator |
| `zones.csv` | Location registry: zone coordinates and the sensor reporting each |
| `ground_truth.json` | **Every planted effect**, with parameters and exact record ids |

## Regenerating

Deterministic from the seed — the same seed reproduces these files byte for
byte, which is what makes the committed copy verifiable:

```bash
python scripts/generate_demo_data.py --out data/sample --force
git diff --exit-code -- data/sample    # should be clean
```

CI runs exactly that check, so the data can never silently drift from the code.

## The CSVs carry no labels

The feed columns are only what a real source would send:

```
record_id, source_id, interval_end_local, location_id, metric, value, unit
```

No `is_friday_evening`, no duplicate flag, nothing naming an injected effect.
A label column in the data would be a leakage channel — a later phase that
could read the answer off its input would be credited with a discovery it never
made. All labels live in `ground_truth.json`.

Two details worth knowing before reading the data:

- **`interval_end_local` stamps the END of the measurement hour**, in the
  source's local time (IST, +05:30), with each sensor's clock skew baked in.
  The event began one interval earlier. Ingestion snaps to the nearest interval
  to absorb the skew, keeping the raw instant as `source_time`.
- **An empty `value` cell means missing**, and is never written as `0`. A dry
  hour of rainfall *is* a measured `0.0`; the two are different facts.

## Rules for anything added here

- **Synthetic or public only.** No personal data of any kind; zone-level
  aggregates (SPEC section 45).
- Reproducible from a script and a seed, not hand-edited.

Unlike the other `data/` subdirectories, contents here **are** committed.

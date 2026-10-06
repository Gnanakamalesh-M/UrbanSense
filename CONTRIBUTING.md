# Contributing to UrbanSense

## Non-negotiable rules

These are the project's reasons for existing. A change that breaks one of them will be rejected
even if the tests pass, so please raise an issue first if you think one needs to change.

1. **Raw data is immutable.** Never overwrite or delete anything under `data/raw/`. Corrections
   arrive as new files.
2. **Nothing questionable is destroyed.** Invalid, conflicting and suspicious records go to
   `quarantine/` with their payload exactly as received. Duplicates are *linked*, not deleted.
   Conflicts are *recorded*, not resolved by overwriting.
3. **Every observation keeps its provenance.** `provenance` is required and never defaulted. If you
   cannot say where a value came from, it does not enter the dataset.
4. **Missing is never zero.** Use `value=None`. If you fill a value, set `is_imputed=True` and say
   how with `imputation_method`.
5. **Temporal splits only.** No random shuffling of time series, and no feature may be computed from
   data at or after the prediction target time.
6. **Time is unambiguous.** Timezone-aware datetimes only; stored in UTC. `event_time` is when it
   happened, `received_time` is when we were told — never conflate them.
7. **Config-driven.** Traffic is the only problem type in v1 and nothing special-cases it. A new
   domain is a YAML file in `configs/problems/`, not a code branch.
8. **Honest reporting.** If the adaptive model loses to the baseline, the report says so.

## Setup

Requires Python 3.11+. `make` is optional — every target wraps a plain `python` command.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"             # add ,dashboard for Streamlit
```

## Before opening a pull request

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy

# The suite runs in two chunks. The whole thing in one process exhausts memory
# on a modest machine: the experiment tests generate and replay several
# synthetic datasets, each holding a feature table and three models.
python -m pytest --ignore=tests/test_experiments.py -p no:cacheprovider
python -m pytest tests/test_experiments.py -p no:cacheprovider
```

With `make` available, `make format` and `make check` wrap the same commands, and `make test-fast` /
`make test-experiments` are the two chunks. `make coverage` runs the suite in one process with
coverage — useful, but it is the configuration that runs out of memory, so expect to need headroom.

The experiment suite is the slow one (roughly a minute): it generates a synthetic dataset per seed
and replays three model arms over it. If you are iterating on anything else, `test-fast` is the one
to run.

- **Type hints are required** on all new code; mypy runs in strict mode.
- **Tests are required.** A new invariant means a new test that fails without the change. When
  adding a validator, test both the acceptance and the rejection.
- Add new test data through the factories in `tests/conftest.py` rather than literal observations,
  so the suite adapts in one place when the schema grows.
- Docstrings follow the Google convention and say *why*, not just *what*. Reference the governing
  `docs/SPEC.md` section where it clarifies intent.

## Phases

Work lands in small, separately committed phases — see the roadmap in the README. Please keep a pull
request within one phase; a change that needs modelling code and ingestion code at once is usually
two pull requests.

## Where things live

| directory | contents |
|---|---|
| `src/urbansense/` | the library: ingestion through adaptation, plus the API and dashboard |
| `src/urbansense/api/` | the read-only FastAPI layer. Root `api/` is a pointer to it |
| `src/urbansense/dashboard/` | the Streamlit app. Root `dashboard/` is a pointer to it |
| `scripts/` | the real interface — every one runs standalone with plain `python` |
| `configs/` | problem definitions, feature sets, adaptation rules, experiment pre-registration |
| `reports/`, `quarantine/`, `models/artifacts/` | generated, gitignored |

The API and the dashboard live under `src/` rather than in the root directories the spec sketches,
because `src/` is what `ruff` and `mypy` already cover and it makes both importable for tests. The
root directories hold READMEs pointing at the real locations.

**The dashboard's data layer must not import Streamlit.** `loaders.py` and `views.py` are plain
Python so their output can be tested without launching a server, and CI does not install Streamlit
at all. `app.py` is the only module that may import it; a test asserts this.

## Docker

```bash
docker compose build     # installs dependencies and bakes the demo artifacts (~10 min)
docker compose up        # API on 127.0.0.1:8000, dashboard on 127.0.0.1:8501
```

Ports are published to `127.0.0.1` only, deliberately: neither service has authentication. Inside
the container the servers bind `0.0.0.0`, which they must — a process on the container's own
loopback is unreachable even from the sibling container — so the localhost guarantee lives in the
published mapping. Keep it that way.

## Privacy

Public, sample and synthetic data only, aggregated to zone level. No personal identifiers, no
individual tracking (SPEC section 45). Do not commit real data of any kind to `data/sample/`.

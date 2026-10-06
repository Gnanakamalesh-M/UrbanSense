# `dashboard/`

The dashboard lives in [`src/urbansense/dashboard/`](../src/urbansense/dashboard/), not here.

It was planned for this directory in Phase 0 and moved during Phase 9, for the same reasons the API
moved: `src/` is already covered by `ruff` and `mypy`, it ships with the package, and it makes the
data layer importable so the tests can exercise it without launching Streamlit.

```bash
python -m pip install -e ".[dashboard]"   # adds streamlit
python scripts/serve_dashboard.py         # 127.0.0.1:8501
python scripts/serve_dashboard.py --check # which artifacts are ready, then exit
```

Six pages — city overview, predictions, patterns, data health, model evolution, experiment — all
reading stored artifacts and recomputing nothing. Source-specific views come before the unified one
(SPEC section 20), and every page says that the data is synthetic.

The package is split so the interesting part is testable:

| module | role | imports streamlit |
|---|---|---|
| `loaders.py` | where data comes from: files directly, or the API over HTTP | no |
| `views.py` | what each page says: artifacts in, plain dicts out | no |
| `app.py` | how it looks | yes |

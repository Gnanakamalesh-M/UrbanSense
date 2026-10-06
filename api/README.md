# `api/`

The API lives in [`src/urbansense/api/`](../src/urbansense/api/), not here.

It was planned for this directory in Phase 0 and moved during Phase 8. `src/` is already covered by
`ruff` and `mypy`, it ships with the package, and it makes the app importable as `urbansense.api`
so the tests can drive it through `TestClient` without a path hack.

```bash
python scripts/serve_api.py          # 127.0.0.1:8000, then open /docs
python scripts/serve_api.py --check  # list which artifacts are available, then exit
```

Read-only, unauthenticated, localhost by default. See [`docs/api.md`](../docs/api.md) for the
endpoint reference and the reasoning behind that posture.

# CLAUDE.md — standing rules for this repository

These apply to every change, without needing to be restated.

## Git

- **Never run `git push`.** Never add or change a git remote. Never touch GitHub — no `gh`
  commands, no API calls, no PRs. The repository owner pushes manually.
- Local commits are fine and expected. Commit, then stop and let the owner push.
- When something depends on a remote (verifying CI, opening a PR), say the commits are ready
  locally and that CI will run on their next push. Do not offer to push.

## Engineering

- **Python 3.11** target, `src/` layout, type hints on all new code, **pytest**, **ruff**.
- `make check` (ruff + mypy strict + pytest) must pass before a commit.
- Local machine has Python 3.10 and 3.14, not 3.11; CI pins 3.11.

## Data rules

These are the project's reason for existing. A change that breaks one is wrong even if the tests
pass — raise it rather than working around it.

- **Raw data is immutable.** Never silently overwrite or delete anything under `data/raw/`.
  Corrections arrive as new files. Rejected, invalid, conflicting and suspicious records go to
  `quarantine/` with their payload exactly as received.
- **Every observation keeps its provenance.** `provenance` is required and never defaulted. If you
  cannot say where a value came from, it does not enter the dataset.
- **Never treat missing as 0.** Use `value=None`. A filled value must set `is_imputed=True` and
  declare its `imputation_method`.
- **Temporal splits only; no future leakage.** Train on the past, test on the future. No random
  shuffling of time series. No feature may use data at or after the prediction target time.
- **Nothing is destroyed.** Duplicates are *linked* (`duplicate_of`), conflicts are *recorded*
  (`conflicts_with`) — never resolved by overwriting.

## Scope

- **Traffic is the only problem type in v1**, and nothing in the code special-cases it. A new
  problem domain is a YAML file in `configs/problems/`, not a code branch. Keep everything
  config-driven so flood and waste can be added later.
- Public, sample and synthetic data only, aggregated to zone level. No personal identifiers.

## Working rhythm

- **One phase at a time**, tested, committed in small separate commits. **Stop at the end of each
  phase** and report how to verify it. Do not roll into the next phase unasked.
- The phase roadmap is in `README.md`; the full vision is `docs/SPEC.md`.

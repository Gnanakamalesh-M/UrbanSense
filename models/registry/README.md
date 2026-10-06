# `models/registry/`

Model version metadata (SPEC sections 31, 32): training period, features, algorithm,
hyperparameters, metrics, dataset version, creation timestamp, and status — one of
`champion`, `challenger`, `rejected`, `archived`.

A rejected candidate is recorded, not discarded: knowing which updates failed evaluation and why is
part of the experimental record.

The registry index is tracked in git; the weights it points at are not.

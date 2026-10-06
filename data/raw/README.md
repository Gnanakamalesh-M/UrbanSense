# `data/raw/` — immutable

Raw source data lands here and is **never overwritten and never deleted** (SPEC section 4A).
Nothing in the codebase may write over an existing file in this directory.

- Ingestion appends; corrections arrive as new files, not edits.
- A record that fails validation does **not** get fixed in place or dropped — it goes to
  [`../../quarantine/`](../../quarantine/) with its payload exactly as received.
- Derived data belongs in `data/processed/`, which is regenerable from here.

Contents are gitignored. The directory itself is tracked so the layout survives a clone.

# `quarantine/`

Where questionable data is **preserved**, not destroyed (SPEC sections 14, 18).

Invalid values, conflicting records, suspected duplicates, low-confidence document extractions,
schema failures and suspicious measurements are written here as
`urbansense.schemas.QuarantineRecord` entries, each carrying:

- the **raw payload exactly as received** — a record that fails schema validation is precisely the
  one a reviewer needs to see, so it is never normalized on the way in;
- the reason, the detector that raised it, and a human-readable explanation;
- the parsed observation, when parsing succeeded and the issue was a content judgement.

Nothing in this directory is deleted by the pipeline. Entries are dispositioned by review and
marked resolved; the original stays.

Contents are gitignored — real quarantined records are data, not source.

# Document processing

> Status: **not implemented.** Planned alongside Phase 2, since document output must pass the same
> integrity gate as every other source. The schema support it depends on exists today
> (`DocumentProvenance`, `ExtractionKind`).

## Pipeline (SPEC section 5)

```
Document -> Parsing -> Text/Table Extraction -> Semantic Understanding
 -> Entity/Value Extraction -> Schema Mapping -> Validation
 -> Confidence Scoring -> Human Review if necessary -> Document Dataset
```

Supported formats: PDF, DOCX, TXT, Markdown, CSV, JSON, XLSX.

## Worked example

Input text:

> "Traffic in Zone B increased to approximately 8,200 vehicles per hour between 6 PM and 8 PM."

Extracted:

```
location   = Zone B
metric     = traffic_volume     # mapped from "vehicles per hour"
value      = 8200
unit       = vehicles/hour
start_time = 18:00
end_time   = 20:00
```

## What is already enforced by the schema

Every document-derived observation must carry `DocumentProvenance`: the document name and content
hash, page or section, the `original_text` the value came from, the extraction confidence, the
extraction timestamp and the extractor version. The schema rejects a `source_type=DOCUMENT`
observation without it.

**The hallucination guard.** `extraction_kind` separates `EXTRACTED_FACT` — the value is present in
`original_text` — from `INFERRED`, where a model derived it. An `INFERRED` record cannot be created
with `validation_status=VALID`; it must be `NEEDS_REVIEW`. So an extractor reading "traffic
increased" cannot produce a record asserting "increased by 27%" as fact.

Note the limit of this guard: it prevents an inferred value from being *treated* as a fact. It
cannot prevent an extractor from misreading a number that genuinely appears in the source — that is
what confidence scoring and review are for.

## Still to build

- Format parsers and table extraction.
- Terminology mapping onto canonical metrics. The alias tables already live in
  `configs/problems/*.yaml`, so this is config lookup rather than new logic.
- Confidence scoring, and routing below `validation.min_document_confidence` to review.
- Duplicate-document detection by content hash (SPEC section 9) — the hash field exists for this.
- The human review queue: approve / edit / reject (SPEC section 35).

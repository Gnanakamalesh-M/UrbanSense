"""Document parsing and extraction pipeline (SPEC sections 4C, 5, 34).

    Document -> Parsing -> Text/Table Extraction -> Semantic Understanding
    -> Entity/Value Extraction -> Schema Mapping -> Validation
    -> Confidence Scoring -> Human Review -> Document Dataset

Extraction is never trusted blindly: every value carries its source document,
page/section, original text, confidence and extraction timestamp, and is marked
EXTRACTED_FACT or INFERRED so a number that is not in the source can never be
promoted to a fact (hallucination guard).

Planned (Phase 2+): PDF/DOCX/TXT/MD/CSV/JSON/XLSX parsers, terminology mapping
to canonical metrics, confidence scoring, human-review queue.
"""

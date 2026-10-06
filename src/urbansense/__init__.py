"""UrbanSense — adaptive urban problem intelligence platform.

See ``docs/SPEC.md`` for the full vision. The codebase is built in phases; each
subpackage below owns one stage of the lifecycle described in SPEC section 2:

    COLLECT -> SEPARATE -> VALIDATE -> NORMALIZE -> DETECT DUPLICATES/CONFLICTS
    -> RECONCILE -> UNIFIED DATA -> DISCOVER PATTERNS -> DETECT ANOMALIES
    -> PREDICT -> EXPLAIN -> OBSERVE OUTCOME -> ANALYZE ERRORS -> DETECT DRIFT
    -> ADAPT MODEL -> VALIDATE -> VERSION -> LEARN

Phase 0 implements only :mod:`urbansense.schemas` and :mod:`urbansense.config`.
Every other subpackage is a documented stub.
"""

__version__ = "0.1.0"

"""Read-only HTTP layer over the stored artifacts (SPEC section 44).

The pipeline writes files; this package serves them. Nothing here ingests,
trains, scores or recomputes anything on a request -- the endpoints open
artifacts that earlier phases produced, so inspecting the system costs a file
read rather than a pipeline run.

**Read-only, unauthenticated, localhost by default.** Only ``GET`` routes are
registered: there is no endpoint that writes, and a ``POST`` is a 405 rather
than a route that happens to be missing. Because there is no authentication,
:mod:`scripts.serve_api` binds ``127.0.0.1`` and ``/health`` says so in its own
response -- an open service that does not announce it is unauthenticated is the
more dangerous kind.
"""

from urbansense.api.app import create_app
from urbansense.api.export import ExportResult, export_api_data, export_is_present
from urbansense.api.store import ArtifactStore

__all__ = [
    "ArtifactStore",
    "ExportResult",
    "create_app",
    "export_api_data",
    "export_is_present",
]

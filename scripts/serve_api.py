"""Serve the read-only API on localhost (SPEC section 44).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/serve_api.py
    python scripts/serve_api.py --port 8080 --rebuild
    python scripts/serve_api.py --check

Then open http://127.0.0.1:8000/docs for the interactive schema.

**Binds 127.0.0.1 and nothing else.** There is no authentication on any
endpoint, so the default must not be reachable from the network. ``--host`` can
override it, and prints a warning when it does, because exposing an
unauthenticated service should be a decision rather than a side effect of a
flag.

Every endpoint reads an artifact that some earlier job wrote. Twelve of the
thirteen families already had files; the unified data layer did not, because it
is rebuilt from CSV in about half a minute. ``--rebuild`` materializes it into
``data/processed/`` first, and the server then only ever opens files -- which is
what keeps "read-only" true of the process and not only of the HTTP verbs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.api import ArtifactStore, create_app, export_api_data, export_is_present
from urbansense.api.app import DEFAULT_HOST, DEFAULT_PORT
from urbansense.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Serve the read-only UrbanSense API over the stored artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "There is no authentication and there are no write endpoints. The "
            "default bind is localhost-only; /health reports both facts in its "
            "own response so a client is never left guessing."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"address to bind (default: {DEFAULT_HOST}, localhost only)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="repository root to read artifacts from (default: the current directory)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-export the unified layer before serving, even if one exists",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report which artifacts are available and exit without serving",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Export if needed, report what is available, and serve."""
    args = build_parser().parse_args(argv)
    store = ArtifactStore(root=args.root)

    if args.rebuild or not export_is_present(store.export_dir):
        reason = "--rebuild" if args.rebuild else "no export found"
        print(f"{reason}: running the pipeline once to materialize the unified layer")
        try:
            result = export_api_data(destination=store.export_dir)
        except (ConfigError, FileNotFoundError) as error:
            print(f"cannot build the export: {error}", file=sys.stderr)
            return 2
        print(f"  {result.describe()}")
        store.clear()

    print("\nartifacts")
    print("-" * 72)
    for family, present in store.availability.items():
        if present:
            print(f"  {family:<14} ready")
        else:
            print(f"  {family:<14} missing -- {store.missing_note(family)}")
    print(
        "\nA missing family is not an error: its endpoint answers 200 with an "
        "empty page and the command above."
    )

    if args.check:
        return 0

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install it with: pip install uvicorn", file=sys.stderr)
        return 2

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"\nWARNING: binding {args.host} exposes an unauthenticated, "
            "read-only service beyond this machine. There is no auth layer to "
            "fall back on.",
            file=sys.stderr,
        )

    print(f"\nserving on http://{args.host}:{args.port}")
    print(f"  interactive schema: http://{args.host}:{args.port}/docs")
    print("  read-only: every route is a GET; no endpoint writes anything")
    print("  no authentication")
    uvicorn.run(create_app(store), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

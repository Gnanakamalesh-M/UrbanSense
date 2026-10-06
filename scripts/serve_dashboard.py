"""Serve the demonstration dashboard on localhost (SPEC section 43).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --port 8600
    python scripts/serve_dashboard.py --api-url http://127.0.0.1:8000
    python scripts/serve_dashboard.py --check

Then open http://127.0.0.1:8501.

**Reads stored artifacts and recomputes nothing.** By default it opens the files
directly, so this works on a fresh clone with no server running. Pass
``--api-url`` to read the same things over HTTP from the read-only API instead --
which is what ``docker-compose`` does, so the two containers genuinely exercise
the API layer.

**Binds 127.0.0.1 and nothing else**, for the same reason the API does: there is
no authentication anywhere in this project. ``--host`` can override it and warns
when it does.

A family with no artifact is not an error. Each page says so and names the
command that would produce it; ``--check`` prints that list and exits without
starting anything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.dashboard.loaders import build_loader

#: Streamlit's default, kept so the printed URL matches the usual one.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501

#: The app module, found relative to this file so the script works from any
#: working directory.
APP_PATH = Path(__file__).resolve().parents[1] / "src" / "urbansense" / "dashboard" / "app.py"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Serve the read-only UrbanSense dashboard over the stored artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every figure on every page is read from an artifact the pipeline "
            "already wrote. Nothing is retrained, re-scored or recomputed per "
            "page load, and all of the data is synthetic."
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
        "--api-url",
        default=None,
        help=(
            "read over HTTP from the API at this URL instead of from disk, "
            "e.g. http://127.0.0.1:8000"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report which artifacts are available and exit without serving",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report what is available, then hand off to Streamlit."""
    args = build_parser().parse_args(argv)

    loader = build_loader(root=args.root, api_url=args.api_url)
    source = f"the API at {args.api_url}" if args.api_url else f"stored files under {args.root}"
    print(f"reading {source}")

    print("\nartifacts")
    print("-" * 72)
    availability = loader.availability()
    if not availability:
        print("  none reported", end="")
        if args.api_url:
            print(f" -- is the API running at {args.api_url}?")
        else:
            print()
    for family, present in sorted(availability.items()):
        if present:
            print(f"  {family:<14} ready")
        else:
            print(f"  {family:<14} missing -- {loader.missing_note(family)}")
    print(
        "\nA missing family is not an error: its page says so and names the "
        "command that would produce it."
    )

    if args.check:
        return 0

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "\nstreamlit is not installed. Install the dashboard extra with:\n"
            '  python -m pip install -e ".[dashboard]"\n'
            "  (or: python -m pip install -r requirements-dashboard.txt)",
            file=sys.stderr,
        )
        return 2

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"\nWARNING: binding {args.host} exposes an unauthenticated dashboard "
            "beyond this machine. There is no auth layer to fall back on.",
            file=sys.stderr,
        )

    # Passed through the environment rather than as script arguments: Streamlit
    # owns the command line it hands to the app, and its own --
    # separator for script args is version-dependent. The environment is stable
    # across versions and is what app.py reads.
    environment = dict(os.environ)
    environment["URBANSENSE_ROOT"] = str(args.root)
    environment["URBANSENSE_API_URL"] = args.api_url or ""

    print(f"\nserving on http://{args.host}:{args.port}")
    print("  read-only: the dashboard never writes an artifact or trains a model")
    print("  all data is synthetic, generated by this repository")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

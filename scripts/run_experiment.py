"""Run the static vs scheduled-retrain vs adaptive experiment (SPEC section 38).

Runs standalone with nothing but Python -- no ``make`` required:

    python scripts/run_experiment.py
    python scripts/run_experiment.py --seeds 20260101 --replay-days 60 --no-write
    python scripts/run_experiment.py --config configs/experiments/static_vs_adaptive.yaml

Three arms replay the **same** stream and are scored on the **same** rows:

``static``
    The frozen pre-drift champion. Never retrained.

``scheduled``
    Retrained every N days on the expanding window and deployed
    unconditionally. **This is the fair baseline**, and the reason the
    experiment exists: Phase 7 showed the adaptive system beating a frozen
    model, which is a weak claim, because the cheap alternative nobody had
    tried is retraining on a calendar.

``adaptive``
    The Phase 7 drift to candidate to promotion-rules system.

The parameters are **pre-registered** in ``configs/experiments/`` and copied
into every report. ``--seeds`` and ``--replay-days`` narrow a run for a quick
check; narrowing is visible in the report's own header, so a shortened run
cannot be mistaken for the registered one.

Each seed is generated into a temporary directory and deleted afterwards.
``data/sample`` is never written: other phases' tests assert against it.

**Nothing here reads ``ground_truth.json``.** Error is measured against observed
values; where the drift was planted is not an input to any arm or any metric.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbansense.adaptation import load_adaptation_config
from urbansense.config import ConfigError, load_settings
from urbansense.experiments import (
    ExperimentConfigError,
    ExperimentRunner,
    build_report,
    load_experiment_config,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compare a frozen model, scheduled retraining and the adaptive system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The pre-registered configuration is the default. Overriding seeds or "
            "the replay length is fine for a quick check -- the report records "
            "what was actually run -- but no parameter should be changed in "
            "response to a result, which is the point of pre-registering them."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment configuration (default: configs/experiments/static_vs_adaptive.yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="generator seeds to run, overriding the pre-registered list",
    )
    parser.add_argument(
        "--replay-days",
        type=int,
        default=None,
        help="shorten the replay to this many days, for a faster check",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory to generate datasets into (default: a temporary one, deleted afterwards)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the comparison without writing a report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment and print the comparison table."""
    args = build_parser().parse_args(argv)

    try:
        config = load_experiment_config(args.config)
        settings = load_settings()
        adaptation = load_adaptation_config(config.adaptation_config)
    except (ConfigError, ExperimentConfigError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    overrides: dict[str, object] = {}
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.replay_days is not None:
        overrides["replay_days"] = args.replay_days
    if overrides:
        config = config.model_copy(update=overrides)
        print("note: overriding the pre-registered configuration for this run:")
        for key, value in overrides.items():
            print(f"  {key} = {value}")

    print(config.describe())
    print(
        f"generating {len(config.seeds)} dataset(s); data/sample is never written. "
        "Each seed takes a couple of minutes."
    )

    # A temporary workspace by default, so a run leaves nothing behind. The
    # datasets are regenerable from their seeds, which is what makes deleting
    # them safe -- the seed is in the report.
    temporary = args.workspace is None
    workspace = (
        Path(tempfile.mkdtemp(prefix="urbansense-experiment-")) if temporary else (args.workspace)
    )

    try:
        runner = ExperimentRunner(
            config=config, workspace=workspace, settings=settings, adaptation=adaptation
        )
        runs = []
        for index, seed in enumerate(config.seeds, start=1):
            print(f"\n[{index}/{len(config.seeds)}] seed {seed}")
            run = runner.run_seed(seed)
            runs.append(run)
            for name, metrics in run.metrics.items():
                print(
                    f"  {name:<10} MAE {metrics.mae:>8,.1f}  "
                    f"F1 {metrics.congestion.f1:>5.3f}  {metrics.cost.describe()}"
                )
    finally:
        if temporary:
            shutil.rmtree(workspace, ignore_errors=True)

    report = build_report(config, runs)
    print("\n" + "=" * 72)
    print(report.render())

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    json_path, markdown_path = write_report(report)
    print(f"\nwrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

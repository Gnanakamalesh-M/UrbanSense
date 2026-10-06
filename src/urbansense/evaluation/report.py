"""Rendering results as a plain table (SPEC section 38).

    Report results honestly. Do not cherry-pick favorable results.

So the renderer takes every model and every slice it is given and prints all of
them, in a fixed order, with no "best model" highlighting and no filtering. A
report that only showed the winning configuration would be the cherry-picking
the spec warns against, and a renderer that *could* filter would eventually be
asked to.

Relative MAE is shown next to absolute MAE because the zones differ by a factor
of four in volume: an MAE of 400 is poor on a residential collector and good on
a six-lane arterial, and the absolute number alone invites the wrong comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

from urbansense.evaluation.metrics import EvaluationResult, SliceMetrics


def _format_value(value: float) -> str:
    """Render a float, or a dash when it is not a number."""
    if value != value:  # NaN
        return "-"
    return f"{value:,.1f}"


def _format_fraction(value: float) -> str:
    """Render a fraction as a percentage, or a dash."""
    if value != value:  # NaN
        return "-"
    return f"{value:.1%}"


def render_table(results: Sequence[EvaluationResult], *, title: str = "") -> str:
    """Render one results table: every model against every slice.

    Rows are slices and columns are models, so the comparison a reader actually
    wants -- "which model is better here?" -- is a horizontal scan.
    """
    if not results:
        return "no results"

    models = [result.model_name for result in results]
    slice_names = ["overall"]
    for result in results:
        for item in result.slices:
            if item.name not in slice_names:
                slice_names.append(item.name)

    def metrics_for(result: EvaluationResult, name: str) -> SliceMetrics | None:
        if name == "overall":
            return result.overall
        return result.slice_named(name)

    label_width = max(len(name) for name in slice_names) + 2
    column_width = max(14, max(len(name) for name in models) + 2)

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))

    header = "slice".ljust(label_width) + "rows".rjust(8)
    for model in models:
        header += model.rjust(column_width)
    lines.append(header)
    lines.append("-" * len(header))

    for name in slice_names:
        first = next(
            (metrics_for(result, name) for result in results if metrics_for(result, name)),
            None,
        )
        rows = 0 if first is None else first.rows

        mae_line = name.ljust(label_width) + f"{rows:,}".rjust(8)
        rmse_line = "".ljust(label_width) + "".rjust(8)
        relative_line = "".ljust(label_width) + "".rjust(8)
        for result in results:
            cell = metrics_for(result, name)
            if cell is None:
                mae_line += "-".rjust(column_width)
                rmse_line += "".rjust(column_width)
                relative_line += "".rjust(column_width)
                continue
            mae_line += _format_value(cell.mae).rjust(column_width)
            rmse_line += f"({_format_value(cell.rmse)})".rjust(column_width)
            relative_line += _format_fraction(cell.relative_mae).rjust(column_width)

        lines.append(mae_line)
        lines.append(rmse_line)
        lines.append(relative_line)
        lines.append("")

    lines.append("each slice shows three lines: MAE, (RMSE), MAE as % of the mean actual")
    return "\n".join(lines)


def render_comparison(results: Sequence[EvaluationResult], *, baseline: str, candidate: str) -> str:
    """State plainly whether one model beat another, overall and per slice.

    Reports every slice including the ones where the candidate lost. A comparison
    that only listed improvements would be a sales pitch rather than an
    evaluation, and the slices where a model is weak are the ones worth knowing
    about.
    """
    by_name = {result.model_name: result for result in results}
    if baseline not in by_name or candidate not in by_name:
        return f"cannot compare {candidate!r} against {baseline!r}: missing results"

    base = by_name[baseline]
    other = by_name[candidate]

    lines = [f"{candidate} vs {baseline}:"]

    def line(name: str, left: SliceMetrics | None, right: SliceMetrics | None) -> str:
        if left is None or right is None:
            return f"  {name}: not comparable"
        if left.mae != left.mae or right.mae != right.mae:
            return f"  {name}: not comparable (no finite MAE)"
        change = right.mae - left.mae
        direction = "better" if change < 0 else ("worse" if change > 0 else "equal")
        share = abs(change) / left.mae if left.mae else float("nan")
        return (
            f"  {name}: MAE {right.mae:,.1f} vs {left.mae:,.1f} "
            f"-> {_format_fraction(share)} {direction}"
        )

    lines.append(line("overall", base.overall, other.overall))
    for item in other.slices:
        lines.append(line(item.name, base.slice_named(item.name), item))

    worse = [
        item.name
        for item in other.slices
        if (peer := base.slice_named(item.name)) is not None
        and item.mae == item.mae
        and peer.mae == peer.mae
        and item.mae > peer.mae
    ]
    if worse:
        lines.append(f"  weaker than {baseline} on: {', '.join(worse)}")
    else:
        lines.append(f"  not weaker than {baseline} on any reported slice")
    return "\n".join(lines)

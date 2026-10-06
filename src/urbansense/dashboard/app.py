"""The Streamlit pages (SPEC section 43).

**The only module in the package that imports streamlit.** Everything it
displays comes from :mod:`urbansense.dashboard.views`, which is plain Python and
carries the tests -- so the numbers on these pages are checked without a browser
or a server.

Run it through the script rather than directly, so the localhost bind and the
backend choice are applied:

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --api-url http://127.0.0.1:8000

Six pages, each reading stored artifacts and recomputing nothing. Charts are
``st.bar_chart``, ``st.line_chart`` and ``st.map``; tables are
``st.dataframe``. Nothing custom, because the interesting part of this project
is the pipeline's honesty and not its front end.

Two conventions hold on every page. **Source-specific views precede the unified
one** (SPEC section 20): per-source health is shown before any aggregate, since
a healthy total hides a failed feed. And an artifact that has not been produced
yet renders as an explicit "not computed" panel naming the command that would
produce it, rather than as an empty chart that looks like a measurement of
nothing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from urbansense import __version__
from urbansense.dashboard import views
from urbansense.dashboard.loaders import DashboardData, build_loader

PAGES = (
    "City Overview",
    "Predictions",
    "Patterns",
    "Data Health",
    "Model Evolution",
    "Experiment",
)


@st.cache_resource
def _loader(root: str, api_url: str) -> DashboardData:
    """Build the backend once per session.

    Cached on the two strings that define it rather than on the object, because
    ``st.cache_resource`` keys on arguments and a loader holds an artifact cache
    that should survive a rerun. Streamlit reruns the whole script on every
    widget interaction, so without this each click would re-read every file.
    """
    return build_loader(root=Path(root), api_url=api_url or None)


def _note(panel_note: str | None, family: str) -> None:
    """Render the "nothing computed yet" explanation for an absent artifact."""
    st.info(
        f"**No {family} artifact yet.** This is a normal state for a fresh "
        f"clone, not an error.\n\n"
        + (f"```\n{panel_note}\n```" if panel_note else "Run the pipeline scripts to produce it.")
    )


def _metric(label: str, value: object, *, suffix: str = "") -> None:
    """One headline figure, or an em dash when the artifact is absent.

    An absent figure is shown as "--" rather than 0. A zero is a measurement and
    a blank is the absence of one, and a dashboard that conflated them would be
    lying in the most ordinary way available to it.
    """
    if value is None:
        st.metric(label, "--")
    elif isinstance(value, float):
        st.metric(label, f"{value:,.3f}{suffix}")
    elif isinstance(value, int):
        st.metric(label, f"{value:,}{suffix}")
    else:
        st.metric(label, f"{value}{suffix}")


def _chart_frame(
    rows: list[dict[str, Any]],
    index: str,
    columns: list[str],
) -> Any:  # noqa: ANN401 - pandas ships no py.typed, so a DataFrame is already Any
    """Rows as something ``st.bar_chart`` can index.

    pandas arrives with streamlit, so it is used here and nowhere else in the
    package -- keeping it out of :mod:`~urbansense.dashboard.views` is what lets
    the tests run without the dashboard extra installed.

    The return is annotated ``Any`` because pandas ships no ``py.typed`` marker:
    to mypy a DataFrame already *is* ``Any``, so a more specific annotation here
    would claim a precision the type checker cannot verify.
    """
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty or index not in frame:
        return frame
    keep = [column for column in columns if column in frame]
    return frame.set_index(index)[keep]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def city_overview(data: DashboardData) -> None:
    """Risk by zone, anomalies, data health and model status."""
    st.header("City Overview")
    headline = views.overview_headline(data)

    columns = st.columns(4)
    with columns[0]:
        _metric("Champion model", headline["champion"])
    with columns[1]:
        _metric("Anomalies flagged", headline["anomalies"])
    with columns[2]:
        _metric("Drift firings", headline["drift_firings"])
    with columns[3]:
        _metric("Conflicts recorded", headline["conflicts"])

    columns = st.columns(3)
    with columns[0]:
        _metric("Unified rows", headline["unified_rows"])
    with columns[1]:
        _metric("Quarantined", headline["quarantined"])
    with columns[2]:
        _metric("Missing values", headline["missing_values"])
    st.caption(
        "Missing values are counted, never filled. A quarantined record is "
        "retained with its raw payload, not deleted."
    )

    panel = views.zone_risk(data)
    if panel.is_empty:
        _note(panel.note, "prediction")
        return

    st.subheader("Risk by zone")
    st.caption(
        "Risk is the mean calibrated congestion probability over the "
        "**recorded** predictions for each zone -- the predictions the system "
        "actually made, not fresh inference. A zone with no recorded prediction "
        "shows a blank rather than a zero."
    )

    points = views.map_points(data)
    if points:
        st.map(_chart_frame(points, "lat", ["lon", "size"]).reset_index(), size="size")
    else:
        st.caption(
            "No zone coordinates available. Over the API the dashboard receives "
            "no geometry, so the map is empty by design; the table below is "
            "unaffected."
        )

    st.bar_chart(_chart_frame(panel.rows, "location_id", ["risk"]))
    st.dataframe(panel.rows, use_container_width=True, hide_index=True)


def predictions(data: DashboardData) -> None:
    """One zone and date, with probability, severity and the conditions."""
    st.header("Predictions")
    st.caption(
        "These are **logged predictions, not fresh inference**. Each was made "
        "by whichever model was champion at the time and is shown beside what "
        "actually happened, where that is known. An unresolved prediction keeps "
        "a blank actual and a blank error."
    )

    zones = views.prediction_zones(data)
    if not zones:
        _note(data.missing_note("predictions"), "prediction")
        return

    left, right = st.columns(2)
    with left:
        zone = st.selectbox("Zone", zones)
    dates = views.prediction_dates(data, zone=zone)
    with right:
        date = st.selectbox("Target date", ["(all)", *dates])

    panel = views.prediction_rows(data, zone=zone, date=None if date == "(all)" else date)

    columns = st.columns(4)
    with columns[0]:
        _metric("Predictions", panel.extra["total"])
    with columns[1]:
        _metric("Resolved", panel.extra["resolved"])
    with columns[2]:
        _metric("MAE", panel.extra["mae"])
    with columns[3]:
        _metric("Mean probability", panel.extra["mean_probability"])

    if panel.is_empty:
        st.warning("No recorded predictions for that zone and date.")
        return

    st.subheader("Predicted against actual")
    st.line_chart(_chart_frame(panel.rows, "target_time", ["predicted", "actual"]))

    st.subheader("Calibrated congestion probability")
    st.line_chart(_chart_frame(panel.rows, "target_time", ["probability"]))
    st.caption(
        "Probability is isotonic-calibrated on train/validation only, and "
        "severity bands come from configuration. Neither is fitted on the rows "
        "shown here."
    )
    st.dataframe(panel.rows, use_container_width=True, hide_index=True)


def patterns(data: DashboardData) -> None:
    """Discovered patterns, with the multiple-testing context."""
    st.header("Patterns")
    panel = views.pattern_rows(data)
    if panel.is_empty:
        _note(panel.note, "pattern")
        return

    columns = st.columns(4)
    with columns[0]:
        _metric("Patterns", len(panel))
    with columns[1]:
        _metric("Hypotheses tested", panel.extra["tests_run"])
    with columns[2]:
        _metric("FDR level", panel.extra["fdr_level"])
    with columns[3]:
        _metric("Cells too sparse", panel.extra["cells_too_sparse"])
    st.caption(
        "The hypothesis count is shown beside the findings on purpose: one "
        "pattern out of several hundred tests is a different claim from one out "
        "of ten. Strengths are effect sizes, not p-values, and confidence is "
        "1 - p rather than the probability a pattern is real."
    )

    if panel.extra["by_status"]:
        st.subheader("By status")
        st.bar_chart(
            _chart_frame(
                [
                    {"status": key, "count": value}
                    for key, value in panel.extra["by_status"].items()
                ],
                "status",
                ["count"],
            )
        )

    st.dataframe(panel.rows, use_container_width=True, hide_index=True)

    st.subheader("Anomalies")
    anomalies = views.anomaly_rows(data)
    if anomalies.is_empty:
        _note(anomalies.note, "anomaly")
        return
    st.caption(
        "Flagged, never judged. `is_error` is null on every record: a flagged "
        "reading may be a sensor fault, an event or a genuine extreme, and this "
        "evidence cannot separate them. The leading explanation is the "
        "best-supported candidate, which is not the same as the cause."
    )
    st.dataframe(anomalies.rows, use_container_width=True, hide_index=True)


def data_health(data: DashboardData) -> None:
    """Per-source health first, then the unified counts."""
    st.header("Data Health")

    # Source-specific before unified (SPEC section 20).
    st.subheader("Per source")
    st.caption(
        "Shown before any aggregate: a healthy total hides an unhealthy feed, "
        "and knowing which stream a number came from is the point."
    )
    sources = views.source_health(data)
    if sources.is_empty:
        _note(sources.note, "data-quality")
    else:
        st.dataframe(sources.rows, use_container_width=True, hide_index=True)
        unhealthy = sources.extra.get("unhealthy") or []
        if unhealthy:
            st.warning(f"Not healthy: {', '.join(unhealthy)}")

    st.subheader("Unified counts")
    summary = views.data_health(data)
    columns = st.columns(4)
    with columns[0]:
        _metric("Rows in", summary["rows_in"])
    with columns[1]:
        _metric("Valid", summary["valid"])
    with columns[2]:
        _metric("Quarantined", summary["quarantined"])
    with columns[3]:
        _metric("Missing values", summary["missing_values"])

    columns = st.columns(3)
    with columns[0]:
        _metric("Unified rows", summary["unified_rows"])
    with columns[1]:
        _metric("Duplicates linked", summary["duplicate_linked"])
    with columns[2]:
        _metric("Conflicted observations", summary["conflicted_observations"])
    st.caption(summary["missing_policy"])

    if summary["quarantine_by_reason"]:
        st.subheader("Quarantine by reason")
        st.bar_chart(
            _chart_frame(
                [
                    {"reason": key, "count": value}
                    for key, value in sorted(summary["quarantine_by_reason"].items())
                ],
                "reason",
                ["count"],
            )
        )

    st.subheader("Conflicts")
    conflicts = views.conflict_rows(data)
    if conflicts.is_empty:
        _note(conflicts.note, "conflict")
    else:
        st.caption(
            "Recorded, never resolved by overwriting. Each group keeps every "
            "candidate value and who claimed it; the disputed slot stays absent "
            "from the unified layer rather than being guessed."
        )
        st.dataframe(conflicts.rows, use_container_width=True, hide_index=True)

    st.subheader("Unified layer sample")
    unified = views.unified_sample(data)
    if unified.is_empty:
        _note(unified.note, "unified-export")
    else:
        st.caption(
            "Every row carries where its value came from -- the question this "
            f"project exists to answer. {unified.extra['missing_values']} of "
            "these rows carry no measurement, and stay null."
        )
        st.dataframe(unified.rows, use_container_width=True, hide_index=True)


def model_evolution(data: DashboardData) -> None:
    """Versions, champion, drift events and candidate decisions."""
    st.header("Model Evolution")

    timeline = views.model_timeline(data)
    if timeline.is_empty:
        _note(timeline.note, "model registry")
    else:
        columns = st.columns(3)
        with columns[0]:
            _metric("Versions", timeline.extra["versions"])
        with columns[1]:
            _metric("Champion", timeline.extra["champion"])
        with columns[2]:
            _metric("Champions (must be 1)", timeline.extra["champion_count"])
        if timeline.extra["champion_count"] != 1:
            st.error(
                "Exactly one champion is an invariant of the registry. "
                f"Found {timeline.extra['champion_count']}."
            )
        st.caption(
            "Rejected challengers keep their rows. Knowing which updates failed, "
            "and by how much, is the experimental record."
        )
        st.dataframe(timeline.rows, use_container_width=True, hide_index=True)

    st.subheader("Drift evaluations")
    drift = views.drift_rows(data)
    if drift.is_empty:
        _note(drift.note, "drift")
    else:
        columns = st.columns(3)
        with columns[0]:
            _metric("Evaluations", drift.extra["evaluations"])
        with columns[1]:
            _metric("Firings", drift.extra["firings"])
        with columns[2]:
            _metric("Episodes", drift.extra["episodes"])
        st.caption(
            "Quiet evaluations are kept. A false-alarm rate cannot be read from "
            "firings alone, so a calm detector should look calm rather than idle."
        )
        st.dataframe(drift.rows, use_container_width=True, hide_index=True)

    st.subheader("Candidate decisions")
    candidates = views.candidate_rows(data)
    if candidates.is_empty:
        _note(candidates.note, "candidate")
    else:
        columns = st.columns(3)
        with columns[0]:
            _metric("Candidates", candidates.extra["candidates"])
        with columns[1]:
            _metric("Promoted", candidates.extra["promoted"])
        with columns[2]:
            _metric("Rejected", candidates.extra["rejected"])
        st.caption(
            "A drift event never retrains on its own: it raises a candidate, "
            "which is then judged against config-declared promotion rules. A "
            "rejected candidate is the system working."
        )
        st.dataframe(candidates.rows, use_container_width=True, hide_index=True)


def experiment(data: DashboardData) -> None:
    """The three-arm comparison, as the stored report states it."""
    st.header("Experiment: static vs scheduled retrain vs adaptive")

    panel = views.experiment_summary(data)
    if panel.is_empty:
        _note(panel.note, "experiment")
        return

    extra = panel.extra
    st.caption(
        f"From `{extra['report_file']}`, pre-registered {extra['pre_registered']}, "
        f"{len(extra['seeds'])} seed(s), scheduled retrain every "
        f"{extra['retrain_every_days']} days. The verdict below is read from the "
        "report, not recomputed here."
    )

    conclusion = extra.get("conclusion")
    if conclusion:
        st.warning(f"**Reported verdict.** {conclusion}.")
    if extra.get("ranking"):
        st.markdown(f"Ranked by measured MAE: **{' < '.join(extra['ranking'])}**.")

    st.subheader("Accuracy and cost")
    st.caption(
        "Cost sits beside accuracy on purpose: an accuracy table on its own "
        "lets the most expensive arm look best for free."
    )
    st.dataframe(panel.rows, use_container_width=True, hide_index=True)
    st.bar_chart(_chart_frame(panel.rows, "arm", ["mae"]))
    st.bar_chart(_chart_frame(panel.rows, "arm", ["training_rows"]))

    st.subheader("Where each arm lost")
    regressions = extra.get("regressions") or {}
    if regressions:
        for arm, slices in regressions.items():
            if slices:
                st.markdown(f"- **{arm}** was beaten on {len(slices)}: {', '.join(slices)}")
            else:
                st.markdown(f"- **{arm}** was not beaten on any slice")
    slices_panel = views.experiment_slices(data)
    if not slices_panel.is_empty:
        st.dataframe(slices_panel.rows, use_container_width=True, hide_index=True)

    if extra.get("limitations"):
        st.subheader("Limitations")
        for item in extra["limitations"]:
            st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the sidebar and the selected page."""
    st.set_page_config(page_title="UrbanSense", page_icon="🚦", layout="wide")

    root = os.environ.get("URBANSENSE_ROOT", ".")
    api_url = os.environ.get("URBANSENSE_API_URL", "")
    data = _loader(root, api_url)

    st.sidebar.title("UrbanSense")
    st.sidebar.caption(f"v{__version__} -- read-only demonstration")
    page = st.sidebar.radio("Page", PAGES)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Reading from:** {'API at ' + api_url if api_url else 'stored files on disk'}"
    )
    st.sidebar.caption(
        "Nothing is recomputed per page load; every figure is read from an artifact."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Artifacts**")
    for item in views.readiness(data):
        st.sidebar.markdown(f"{'ready' if item['ready'] else 'missing'} -- `{item['family']}`")

    st.warning(views.SYNTHETIC_BANNER)

    renderers = {
        "City Overview": city_overview,
        "Predictions": predictions,
        "Patterns": patterns,
        "Data Health": data_health,
        "Model Evolution": model_evolution,
        "Experiment": experiment,
    }
    renderers[str(page)](data)

    st.markdown("---")
    st.caption(
        "Synthetic, zone-level data only. No personal identifiers, no individual "
        "tracking, no real sensor feed (SPEC section 45). Results are simulation "
        "outcomes, not measurements of a city and not forecasts for one."
    )


# ``streamlit run`` executes this file as a script, so the guard is what runs the
# app. It also means importing the module -- which the tests and CI's compile
# check do -- renders nothing.
if __name__ == "__main__":
    main()

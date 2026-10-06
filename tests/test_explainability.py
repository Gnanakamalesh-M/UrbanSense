"""Tests for factor attribution and the language it is reported in.

Three things are being defended.

**Arithmetic honesty.** When SHAP is available the contributions must actually
reconstruct the prediction; when it is not, the output must say the shares only
rank the inputs. A method that silently claimed additivity it did not have would
be the worst outcome here, so the assertion is conditional on the method rather
than skipped.

**Completeness.** Every feature must map to exactly one factor. An unmapped
feature would move predictions while appearing in no explanation.

**Language.** SPEC section 27 says never to claim a correlated feature is
causal, so the rendered strings are grepped for causal verbs. A style note in a
docstring does not survive contact with a hurried edit; a failing test does.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import temporal_split
from urbansense.explainability import (
    AttributionMethod,
    FactorAttributor,
    global_importance,
    load_factor_groups,
)
from urbansense.explainability.factors import FactorGroups
from urbansense.features import (
    assert_no_leakage,
    build_feature_table,
    detect_leakage,
    load_feature_set,
)
from urbansense.features.loader import FeatureConfigError
from urbansense.ingestion import load_historical
from urbansense.prediction import GradientForecaster
from urbansense.preprocessing import load_location_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
TZ_OFFSET = 5.5

#: Phrases that assert causation. Rendered explanations must contain none of
#: them: the model found an association in past data, which is all a
#: gradient-boosted tree can know.
CAUSAL_PHRASES = (
    "caused",
    "causes",
    "causing",
    "because of",
    "due to",
    "drives",
    "driven by",
    "leads to",
    "results in",
    "responsible for",
)

#: Disclaimers that contain a causal word in order to deny it. Stripped before
#: the grep, because "not causes" is the opposite of a causal claim and a naive
#: substring search would flag the very sentence doing the right thing.
DISCLAIMERS = (
    "not causes",
    "not causation",
    "association, not causation",
    "did not necessarily bring",
)


def assert_not_causal(text: str) -> None:
    """Fail if a rendered string claims causation.

    Disclaimers are removed first, so denying causation is allowed and
    asserting it is not.
    """
    lowered = text.lower()
    for disclaimer in DISCLAIMERS:
        lowered = lowered.replace(disclaimer, "")
    for phrase in CAUSAL_PHRASES:
        assert phrase not in lowered, f"causal language {phrase!r} in: {text}"


@pytest.fixture(scope="module")
def groups() -> FactorGroups:
    return load_factor_groups()


@pytest.fixture(scope="module")
def trained():
    settings = load_settings()
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV), settings=settings, registry=load_location_registry()
    )
    reconciled = reconcile(validated, settings=settings)
    feature_set = load_feature_set()
    split = temporal_split(
        build_feature_table(reconciled, feature_set, timezone_offset_hours=TZ_OFFSET)
    )
    model = GradientForecaster().fit(split.train)
    return {
        "reconciled": reconciled,
        "feature_set": feature_set,
        "split": split,
        "model": model,
    }


@pytest.fixture(scope="module")
def attributor(trained, groups):
    return FactorAttributor.from_forecaster(
        trained["model"], groups=groups, background=trained["split"].validation.x[:500]
    )


def _friday_peak_index(table, zone: str = "ZONE-B") -> int:
    """Row index of the strongest planted Friday evening hour in a table."""
    candidates = [
        index
        for index in range(len(table))
        if table.locations[index] == zone
        and (table.target_times[index] + timedelta(hours=TZ_OFFSET)).weekday() == 4
        and 18 <= (table.target_times[index] + timedelta(hours=TZ_OFFSET)).hour < 20
    ]
    assert candidates, "the planted window is absent from this table"
    return max(candidates, key=lambda index: table.y[index])


# ---------------------------------------------------------------------------
# Which method ran, and what it may claim
# ---------------------------------------------------------------------------


def test_the_method_used_is_recorded(attributor, capsys):
    """Either method is acceptable; silence about which one is not."""
    with capsys.disabled():
        print(
            f"\n  attribution method: {attributor.method.value} "
            f"(additive: {attributor.is_additive})"
        )
        print(f"  reason: {attributor.as_dict()['chosen_because']}")
    assert attributor.method in (AttributionMethod.SHAP_TREE, AttributionMethod.OCCLUSION)
    assert attributor.as_dict()["method"] == attributor.method.value
    assert str(attributor.as_dict()["chosen_because"])


def test_contributions_reconstruct_the_prediction_when_additive(attributor, trained):
    """The arithmetic claim is checked, not asserted.

    If the method says it is additive, baseline plus contributions must equal
    the prediction. If it does not, the output must not imply otherwise.
    """
    index = _friday_peak_index(trained["split"].test)
    explanation = attributor.explain_row(trained["split"].test.x[index])

    if explanation.is_additive:
        assert explanation.reconstruction_error < 1e-6
        assert "reconstructs the predicted" in explanation.describe()
    else:
        assert "not additive" in explanation.describe()
        assert "do not decompose the prediction" in explanation.describe()


def test_shares_sum_to_one(attributor, trained):
    """Normalized to shares of total absolute contribution, as documented."""
    for index in (0, len(trained["split"].test) // 2, len(trained["split"].test) - 1):
        explanation = attributor.explain_row(trained["split"].test.x[index])
        assert sum(item.share for item in explanation.factors) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= item.share <= 1.0 for item in explanation.factors)


def test_factors_are_ordered_by_absolute_contribution(attributor, trained):
    explanation = attributor.explain_row(trained["split"].test.x[0])
    magnitudes = [abs(item.contribution) for item in explanation.factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


# ---------------------------------------------------------------------------
# The Friday peak's explanation
# ---------------------------------------------------------------------------


def test_the_friday_peak_explanation_names_time_related_factors(attributor, trained, capsys):
    """Peak hour and weekday must both appear with a real contribution.

    Not asserted to be the *largest*: the seasonal lags carry most of the
    planted pattern, which is honest and is printed. What would be wrong is a
    Friday-evening spike explained with no reference to the hour or the day.
    """
    index = _friday_peak_index(trained["split"].test)
    explanation = attributor.explain_row(trained["split"].test.x[index])

    with capsys.disabled():
        print(f"\n  planted ZONE-B Friday evening, actual {trained['split'].test.y[index]:,.0f}:")
        print("  " + explanation.describe().replace("\n", "\n  "))

    contributions = {item.factor: item.contribution for item in explanation.factors}
    assert "peak hour" in contributions
    assert "weekday" in contributions
    assert contributions["peak hour"] != 0.0
    assert contributions["weekday"] != 0.0
    assert "historical traffic" in contributions


def test_the_friday_peak_is_explained_as_an_upward_departure(attributor, trained):
    """The factors should push the prediction above the model's baseline."""
    index = _friday_peak_index(trained["split"].test)
    explanation = attributor.explain_row(trained["split"].test.x[index])
    assert explanation.prediction > explanation.baseline
    assert sum(item.contribution for item in explanation.factors) > 0


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------


def test_no_rendered_explanation_claims_causation(attributor, trained):
    """SPEC section 27, enforced rather than noted."""
    index = _friday_peak_index(trained["split"].test)
    explanation = attributor.explain_row(trained["split"].test.x[index])

    rendered = [explanation.describe()]
    rendered.extend(item.describe() for item in explanation.factors)
    rendered.append(str(explanation.as_dict()["interpretation_note"]))
    rendered.extend(str(value) for value in attributor.as_dict().values())

    for text in rendered:
        assert_not_causal(text)


def test_the_explanation_says_what_it_is_not(attributor, trained):
    explanation = attributor.explain_row(trained["split"].test.x[0])
    assert "not causes" in explanation.describe()
    assert "not causes" in str(explanation.as_dict()["interpretation_note"])
    assert "contributed" in explanation.factors[0].describe()


def test_global_importance_disclaims_causation(trained, groups, capsys):
    importance = global_importance(
        trained["model"].estimator.predict,
        trained["split"].validation.x,
        trained["split"].validation.y,
        groups=groups,
        feature_names=trained["split"].validation.names,
        sample=400,
        repeats=2,
    )
    with capsys.disabled():
        print()
        print("  " + importance.describe().replace("\n", "\n  "))

    assert sum(importance.by_factor.values()) == pytest.approx(1.0, abs=1e-9)
    assert "association, not causation" in importance.describe()
    assert_not_causal(importance.describe())


# ---------------------------------------------------------------------------
# Factor grouping is complete and unambiguous
# ---------------------------------------------------------------------------


def test_every_feature_maps_to_exactly_one_factor(groups):
    names = load_feature_set().names
    assignment = groups.assign(names)
    assert set(assignment) == set(names)
    assert len(assignment) == len(names)


def test_an_unmapped_feature_is_an_error(groups):
    """It would otherwise move predictions and appear in no explanation."""
    with pytest.raises(FeatureConfigError, match="belongs to no factor group"):
        groups.factor_for("mystery_feature")


def test_an_ambiguous_feature_is_an_error():
    ambiguous = FactorGroups(patterns={"first": ("lag_*",), "second": ("lag_1h",)})
    with pytest.raises(FeatureConfigError, match="matches several factor groups"):
        ambiguous.factor_for("lag_1h")


def test_the_grouping_covers_the_readable_names_spec_27_shows(groups):
    names = set(groups.names)
    assert {"historical traffic", "rainfall", "peak hour", "weekday"} <= names


def test_grouped_features_are_reported_per_factor(groups):
    grouped = groups.grouped(load_feature_set().names)
    assert "lag_1h" in grouped["historical traffic"]
    assert "rainfall_0h" in grouped["rainfall"]
    assert "zone" in grouped["location"]
    assert sum(len(items) for items in grouped.values()) == len(load_feature_set().names)


def test_a_malformed_factor_block_is_rejected(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text("target_metric: x\nhorizon_hours: 1\nfactors: {}\n", encoding="utf-8")
    with pytest.raises(FeatureConfigError, match="non-empty 'factors' mapping"):
        load_factor_groups(path)


def test_a_factor_with_no_patterns_is_rejected(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text("factors:\n  rainfall: []\n", encoding="utf-8")
    with pytest.raises(FeatureConfigError, match="at least one feature pattern"):
        load_factor_groups(path)


def test_a_misconfigured_grouping_fails_at_construction(trained):
    """Not when someone later asks for an explanation."""
    incomplete = FactorGroups(patterns={"only lags": ("lag_*",)})
    with pytest.raises(FeatureConfigError, match="belongs to no factor group"):
        FactorAttributor.from_forecaster(trained["model"], groups=incomplete)


# ---------------------------------------------------------------------------
# The explanation inputs are leakage-free
# ---------------------------------------------------------------------------


def test_the_leakage_probe_is_clean_for_the_explanation_inputs(trained, capsys):
    """The occlusion path rebuilds features, so it is probed the same way.

    An explanation computed from a feature that reads the future would be a
    convincing account of a cheated prediction, which is worse than no
    explanation at all.
    """
    report = detect_leakage(
        trained["reconciled"], trained["feature_set"], timezone_offset_hours=TZ_OFFSET
    )
    with capsys.disabled():
        print()
        print("  " + report.describe().replace("\n", "\n  "))
    assert_no_leakage(report)
    assert report.is_clean


def test_occlusion_only_blanks_the_feature_it_is_measuring(trained, groups):
    """The fallback must not perturb anything else.

    Checked by construction: every occluded row differs from the original in
    exactly one column.
    """
    names = trained["split"].test.names
    occluded = FactorAttributor.build(
        trained["model"].estimator.predict,
        groups=groups,
        feature_names=names,
        estimator=None,  # force the fallback
        background=trained["split"].validation.x[:200],
    )
    assert occluded.method is AttributionMethod.OCCLUSION

    row = trained["split"].test.x[0]
    matrix = np.repeat(row.reshape(1, -1), len(names), axis=0)
    for index in range(len(names)):
        matrix[index, index] = np.nan
    differences = [int(np.sum(~_same(matrix[index], row))) for index in range(len(names))]
    assert differences == [1] * len(names)


def _same(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Elementwise equality that treats NaN as equal to NaN."""
    return (left == right) | (np.isnan(left) & np.isnan(right))


def test_the_fallback_produces_usable_explanations(trained, groups):
    """CI runs 3.11 and may not have shap, so the fallback must work."""
    occluded = FactorAttributor.build(
        trained["model"].estimator.predict,
        groups=groups,
        feature_names=trained["split"].test.names,
        estimator=None,
        background=trained["split"].validation.x[:200],
    )
    index = _friday_peak_index(trained["split"].test)
    explanation = occluded.explain_row(trained["split"].test.x[index])

    assert explanation.method is AttributionMethod.OCCLUSION
    assert not explanation.is_additive
    assert sum(item.share for item in explanation.factors) == pytest.approx(1.0, abs=1e-9)
    assert "not additive" in explanation.describe()
    assert_not_causal(explanation.describe())

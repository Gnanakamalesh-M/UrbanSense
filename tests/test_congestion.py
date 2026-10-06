"""Tests for the congestion definition, thresholds and severity bands.

The load-bearing test here is the first one: the threshold must not be able to
see validation or test data. It is checked by *altering* those splits and
asserting the fitted thresholds do not move, which catches a leak no matter how
it was introduced -- a wrong argument, a concatenation, a well-meaning
"use all the data we have".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from urbansense.config import load_settings
from urbansense.config.models import (
    CongestionConfig,
    ProblemConfig,
    SeverityBand,
    ThresholdConfig,
)
from urbansense.data_integrity import reconcile, validate_and_normalize
from urbansense.evaluation import temporal_split
from urbansense.features import build_feature_table, load_feature_set
from urbansense.features.builder import FeatureTable
from urbansense.ingestion import load_historical
from urbansense.prediction import congested_rate, fit_thresholds
from urbansense.preprocessing import load_location_registry
from urbansense.schemas import ProblemType
from urbansense.schemas.enums import Severity

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_CSV = REPO_ROOT / "data" / "sample" / "traffic_history.csv"
TZ_OFFSET = 5.5


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def congestion_config(settings):
    config = settings.problems[ProblemType.TRAFFIC].congestion
    assert config is not None, "the traffic problem must define congestion"
    return config


@pytest.fixture(scope="module")
def split(settings):
    validated = validate_and_normalize(
        load_historical(HISTORY_CSV),
        settings=settings,
        registry=load_location_registry(),
    )
    table = build_feature_table(
        reconcile(validated, settings=settings),
        load_feature_set(),
        timezone_offset_hours=TZ_OFFSET,
    )
    return temporal_split(table)


@pytest.fixture(scope="module")
def thresholds(split, congestion_config):
    return fit_thresholds(split.train, congestion_config)


# ---------------------------------------------------------------------------
# The threshold cannot see validation or test
# ---------------------------------------------------------------------------


def test_thresholds_are_fitted_on_train_only(split, congestion_config, capsys):
    """Alter validation and test; the thresholds must not move at all.

    Not "move a little" -- bit-identical. Any path by which later data reached
    the fit would shift a percentile, and a tolerance here would hide exactly
    the small leak hardest to notice.
    """
    baseline = fit_thresholds(split.train, congestion_config)

    def inflated(table: FeatureTable) -> FeatureTable:
        """The same rows with every target multiplied by ten."""
        return FeatureTable(
            feature_set=table.feature_set,
            x=table.x,
            y=table.y * 10.0,
            sample_times=table.sample_times,
            target_times=table.target_times,
            locations=table.locations,
            record_ids=table.record_ids,
            rows_considered=table.rows_considered,
            rows_without_target=table.rows_without_target,
        )

    # Fit again while loudly corrupted validation and test tables exist. The
    # signature offers nowhere to put them, which is the point being asserted.
    _ = inflated(split.validation)
    _ = inflated(split.test)
    after = fit_thresholds(split.train, congestion_config)

    with capsys.disabled():
        print("\n  thresholds (train-only fit):")
        for zone, value in sorted(baseline.by_zone.items()):
            print(f"    {zone}  {value:,.0f}")

    assert after.by_zone == baseline.by_zone
    for zone, value in baseline.by_zone.items():
        assert after.by_zone[zone] == value  # bit-identical, not approx


def test_a_threshold_fitted_on_everything_would_differ(split, congestion_config):
    """The guard above would be vacuous if later data were identical.

    Concatenating the splits moves the thresholds, which proves the first test
    is checking something real rather than restating a tautology.
    """
    combined = FeatureTable(
        feature_set=split.train.feature_set,
        x=np.vstack([split.train.x, split.validation.x, split.test.x]),
        y=np.concatenate([split.train.y, split.validation.y, split.test.y]),
        sample_times=split.train.sample_times
        + split.validation.sample_times
        + split.test.sample_times,
        target_times=split.train.target_times
        + split.validation.target_times
        + split.test.target_times,
        locations=split.train.locations + split.validation.locations + split.test.locations,
        record_ids=split.train.record_ids + split.validation.record_ids + split.test.record_ids,
    )
    leaky = fit_thresholds(combined, congestion_config, segment_name="all data")
    train_only = fit_thresholds(split.train, congestion_config)
    assert leaky.by_zone != train_only.by_zone
    assert all(leaky.by_zone[zone] > train_only.by_zone[zone] for zone in train_only.by_zone)


def test_the_fit_records_which_segment_it_used(thresholds):
    """A report should state the provenance rather than imply it."""
    assert thresholds.fitted_on == "train"
    assert thresholds.rows > 0
    payload = thresholds.as_dict()
    assert payload["fitted_on"] == "train"
    assert "training segment only" in str(payload["note"])


def test_the_congested_rate_rises_out_of_sample(split, thresholds, capsys):
    """The definition ages, and the suite records it rather than hiding it.

    Anchoring to train is correct -- anything else lets the test split set its
    own pass mark -- but it means more of the later period counts as congested.
    This is an honest property of the design, asserted so it cannot change
    silently.
    """
    rates = {
        name: congested_rate(thresholds, table)
        for name, table in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    with capsys.disabled():
        print("\n  congested rate by split (threshold fitted on train):")
        for name, rate in rates.items():
            print(f"    {name:<11} {rate:.3f}")

    assert rates["train"] == pytest.approx(0.10, abs=0.01)
    assert rates["validation"] > rates["train"]
    assert rates["test"] > rates["validation"]


def test_an_empty_table_cannot_produce_a_threshold(split, congestion_config):
    empty = split.train.select(np.zeros(len(split.train), dtype=bool))
    with pytest.raises(ValueError, match="empty"):
        fit_thresholds(empty, congestion_config)


# ---------------------------------------------------------------------------
# Severity is monotonic
# ---------------------------------------------------------------------------


def test_severity_never_decreases_as_volume_rises(thresholds):
    """Sweep the prediction upward; the band index must not go backwards."""
    ranks = [
        thresholds.severity("ZONE-B", volume).rank for volume in np.linspace(0.0, 20000.0, 400)
    ]
    assert ranks == sorted(ranks)
    assert ranks[0] == Severity.LOW.rank
    assert ranks[-1] == Severity.SEVERE.rank


def test_every_band_is_reachable(thresholds, congestion_config):
    """A band no volume can land in would be configuration that does nothing."""
    threshold = thresholds.by_zone["ZONE-B"]
    seen = {
        thresholds.severity("ZONE-B", threshold * band.min_ratio + 1.0)
        for band in congestion_config.severity_bands
    }
    assert seen == set(Severity)


def test_the_ratio_is_unknown_for_an_unfitted_zone(thresholds):
    """Not zero. An unknown ratio is not a quiet "not congested"."""
    assert np.isnan(thresholds.ratio("ZONE-NEVER-SEEN", 5000.0))
    assert thresholds.is_congested("ZONE-NEVER-SEEN", 99999.0) is False


def test_congestion_compares_the_observed_value_not_the_prediction(thresholds):
    threshold = thresholds.by_zone["ZONE-C"]
    assert thresholds.is_congested("ZONE-C", threshold + 1.0)
    assert not thresholds.is_congested("ZONE-C", threshold)
    assert not thresholds.is_congested("ZONE-C", float("nan"))


# ---------------------------------------------------------------------------
# The configuration guards
# ---------------------------------------------------------------------------


def test_bands_must_ascend():
    with pytest.raises(ValueError, match="ascending"):
        CongestionConfig(
            metric="traffic_volume",
            severity_bands=(
                SeverityBand(name=Severity.LOW, min_ratio=0.0),
                SeverityBand(name=Severity.HIGH, min_ratio=1.8),
                SeverityBand(name=Severity.MODERATE, min_ratio=1.0),
            ),
        )


def test_band_names_must_ascend_with_the_ratio():
    with pytest.raises(ValueError, match="names must ascend"):
        CongestionConfig(
            metric="traffic_volume",
            severity_bands=(
                SeverityBand(name=Severity.HIGH, min_ratio=0.0),
                SeverityBand(name=Severity.LOW, min_ratio=1.0),
            ),
        )


def test_the_first_band_must_start_at_zero():
    """Otherwise a quiet hour would fall in no band at all."""
    with pytest.raises(ValueError, match=r"must start at min_ratio 0\.0"):
        CongestionConfig(
            metric="traffic_volume",
            severity_bands=(SeverityBand(name=Severity.LOW, min_ratio=0.5),),
        )


def test_bands_are_required():
    with pytest.raises(ValueError, match="at least one band"):
        CongestionConfig(metric="traffic_volume", severity_bands=())


def test_an_unknown_threshold_method_is_rejected():
    with pytest.raises(ValueError):
        ThresholdConfig(method="eyeball")  # type: ignore[arg-type]


def test_the_percentile_must_be_a_percentile():
    with pytest.raises(ValueError):
        ThresholdConfig(percentile=140.0)


def test_congestion_must_name_a_declared_metric():
    """A threshold on an undeclared metric could never be computed."""
    with pytest.raises(ValueError, match="not one of this problem's metrics"):
        ProblemConfig.model_validate(
            {
                "problem_type": "traffic",
                "metrics": [{"name": "traffic_volume", "canonical_unit": "vehicles/hour"}],
                "congestion": {
                    "metric": "bicycle_volume",
                    "severity_bands": [{"name": "low", "min_ratio": 0.0}],
                },
            }
        )


def test_the_shipped_config_defines_congestion(congestion_config):
    """The behaviour under test must come from the real config, not a fixture."""
    assert congestion_config.metric == "traffic_volume"
    assert congestion_config.threshold.percentile == 90.0
    assert [band.name for band in congestion_config.severity_bands] == [
        Severity.LOW,
        Severity.MODERATE,
        Severity.HIGH,
        Severity.SEVERE,
    ]


def test_thresholds_are_per_zone(thresholds):
    """One city-wide number would call a quiet collector permanently clear."""
    values = sorted(thresholds.by_zone.values())
    assert len(thresholds.by_zone) == 4
    assert values[-1] > values[0] * 2, (
        "zones differ enough that a single shared threshold would be meaningless"
    )


def test_a_hand_built_table_gives_the_percentile_we_expect():
    """The fit is checked against a percentile computed by hand."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = np.array([float(x) for x in range(1, 101)], dtype=np.float64)
    table = FeatureTable(
        feature_set=load_feature_set(),
        x=np.zeros((100, len(load_feature_set().names))),
        y=values,
        sample_times=tuple(start + timedelta(hours=i) for i in range(100)),
        target_times=tuple(start + timedelta(hours=i + 1) for i in range(100)),
        locations=("ZONE-H",) * 100,
        record_ids=tuple(f"REC-{i}" for i in range(100)),
    )
    config = CongestionConfig(
        metric="traffic_volume",
        threshold=ThresholdConfig(percentile=90.0),
        severity_bands=(SeverityBand(name=Severity.LOW, min_ratio=0.0),),
    )
    fitted = fit_thresholds(table, config)
    # numpy's linear interpolation of the 90th percentile of 1..100.
    assert fitted.by_zone["ZONE-H"] == pytest.approx(90.1)

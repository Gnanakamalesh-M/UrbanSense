"""Tests for the YAML configuration layer.

These pin the "config-driven, not traffic-hardcoded" property: v1 runs traffic
only, but it does so because of a YAML file, and a second domain can be added
without touching Python.
"""

from __future__ import annotations

import pytest
import yaml

from urbansense.config import (
    ConfigError,
    available_problem_files,
    load_problem,
    load_settings,
)
from urbansense.schemas.enums import AggregationLevel, ProblemType

# ---------------------------------------------------------------------------
# Real repository configuration
# ---------------------------------------------------------------------------


def test_repository_config_loads(config_dir):
    settings = load_settings(config_dir)
    assert settings.project_name == "UrbanSense"
    assert settings.temporal.display_timezone == "Asia/Kolkata"


def test_traffic_is_the_only_enabled_problem(config_dir):
    settings = load_settings(config_dir)
    assert settings.enabled_problems == (ProblemType.TRAFFIC,)


def test_traffic_metrics_are_defined(config_dir):
    settings = load_settings(config_dir)
    traffic = settings.problems[ProblemType.TRAFFIC]
    names = {metric.name for metric in traffic.metrics}
    assert {"traffic_volume", "average_speed", "congestion_index"} <= names

    volume = traffic.metric("traffic_volume")
    assert volume is not None
    assert volume.canonical_unit == "vehicles/hour"
    assert volume.non_negative is True
    assert volume.default_aggregation is AggregationLevel.HOURLY


def test_env_can_be_overridden(config_dir):
    assert load_settings(config_dir, env="ci").env == "ci"


def test_paths_point_at_the_expected_tree(config_dir):
    paths = load_settings(config_dir).paths
    assert paths.raw.as_posix() == "data/raw"
    assert paths.quarantine.as_posix() == "quarantine"


# ---------------------------------------------------------------------------
# Terminology mapping (SPEC section 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        "traffic_volume",
        "vehicle count",
        "vehicle_count",
        "Vehicle Count",
        "  number of vehicles  ",
        "vehicles observed",
        "traffic-count",
        "FLOW",
    ],
)
def test_aliases_resolve_to_the_canonical_metric(config_dir, spelling):
    settings = load_settings(config_dir)
    metric = settings.resolve_metric(ProblemType.TRAFFIC, spelling)
    assert metric is not None
    assert metric.name == "traffic_volume"


def test_unknown_metric_name_does_not_resolve(config_dir):
    settings = load_settings(config_dir)
    assert settings.resolve_metric(ProblemType.TRAFFIC, "cosmic_rays") is None


def test_resolving_against_a_disabled_domain_returns_none(config_dir):
    """Flood is not loaded in v1, so nothing resolves against it."""
    settings = load_settings(config_dir)
    assert settings.resolve_metric(ProblemType.FLOOD, "rainfall") is None


# ---------------------------------------------------------------------------
# The template proves the design is config-driven
# ---------------------------------------------------------------------------


def test_template_is_excluded_from_loadable_problems(config_dir):
    files = available_problem_files(config_dir)
    assert (config_dir / "problems" / "traffic.yaml") in files
    assert all(not path.name.startswith("_") for path in files)


def test_template_is_valid_yaml_and_a_valid_problem(config_dir, tmp_path):
    """A new domain drops in with no code change -- proven by loading the template.

    The template is copied to its declared name because the loader insists the
    filename and the ``problem_type`` field agree.
    """
    template = config_dir / "problems" / "_template.yaml"
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert data["problem_type"] == "flood"
    assert data["enabled"] is False

    target = tmp_path / "flood.yaml"
    target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    problem = load_problem(target)
    assert problem.problem_type is ProblemType.FLOOD
    assert problem.metric("rainfall") is not None


def test_a_second_domain_can_be_enabled_without_code_changes(config_dir, tmp_path):
    problems = tmp_path / "problems"
    problems.mkdir()
    for name in ("base.yaml",):
        (tmp_path / name).write_text(
            (config_dir / name)
            .read_text(encoding="utf-8")
            .replace("problems:\n  - traffic\n", "problems:\n  - traffic\n  - flood\n"),
            encoding="utf-8",
        )
    (problems / "traffic.yaml").write_text(
        (config_dir / "problems" / "traffic.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (problems / "flood.yaml").write_text(
        (config_dir / "problems" / "_template.yaml")
        .read_text(encoding="utf-8")
        .replace("enabled: false", "enabled: true"),
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)
    assert set(settings.enabled_problems) == {ProblemType.TRAFFIC, ProblemType.FLOOD}
    rainfall = settings.resolve_metric(ProblemType.FLOOD, "precipitation")
    assert rainfall is not None and rainfall.name == "rainfall"


# ---------------------------------------------------------------------------
# Strict loading — nothing is skipped silently
# ---------------------------------------------------------------------------


def _write_base(tmp_path, body: str) -> None:
    (tmp_path / "base.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "problems").mkdir(exist_ok=True)


def test_missing_base_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path)


def test_unknown_problem_name_is_an_error(tmp_path):
    _write_base(tmp_path, "problems:\n  - streetlights\n")
    with pytest.raises(ConfigError, match="unknown problem"):
        load_settings(tmp_path)


def test_missing_problem_file_is_an_error_not_a_skip(tmp_path):
    _write_base(tmp_path, "problems:\n  - traffic\n")
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path)


def test_duplicate_problem_entry_is_an_error(tmp_path):
    _write_base(tmp_path, "problems:\n  - traffic\n  - traffic\n")
    with pytest.raises(ConfigError, match="more than once"):
        load_settings(tmp_path)


def test_problems_must_be_a_list(tmp_path):
    _write_base(tmp_path, "problems: traffic\n")
    with pytest.raises(ConfigError, match="must be a list"):
        load_settings(tmp_path)


def test_unknown_base_key_is_an_error(tmp_path):
    _write_base(tmp_path, "problems: []\ntypo_key: 1\n")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_settings(tmp_path)


def test_malformed_yaml_is_an_error(tmp_path):
    _write_base(tmp_path, "problems: [traffic\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(tmp_path)


def test_non_mapping_base_is_an_error(tmp_path):
    _write_base(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a mapping"):
        load_settings(tmp_path)


def test_filename_and_problem_type_must_agree(tmp_path):
    """A mismatch is a typo in one of the two; guessing would misconfigure a domain."""
    path = tmp_path / "waste.yaml"
    path.write_text("problem_type: flood\nenabled: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="but is named"):
        load_problem(path)


def test_canonical_unit_must_be_an_accepted_unit(tmp_path):
    path = tmp_path / "flood.yaml"
    path.write_text(
        "problem_type: flood\nmetrics:\n"
        "  - name: rainfall\n    canonical_unit: mm\n    accepted_units: [cm, inch]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"not .*listed in accepted_units"):
        load_problem(path)


def test_inverted_plausible_range_is_an_error(tmp_path):
    path = tmp_path / "flood.yaml"
    path.write_text(
        "problem_type: flood\nmetrics:\n"
        "  - name: rainfall\n    canonical_unit: mm\n"
        "    plausible_min: 100\n    plausible_max: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="plausible_max"):
        load_problem(path)


def test_duplicate_metric_names_are_an_error(tmp_path):
    path = tmp_path / "flood.yaml"
    path.write_text(
        "problem_type: flood\nmetrics:\n"
        "  - name: rainfall\n    canonical_unit: mm\n"
        "  - name: rainfall\n    canonical_unit: cm\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate metric names"):
        load_problem(path)


def test_available_problem_files_on_missing_directory(tmp_path):
    assert available_problem_files(tmp_path) == ()

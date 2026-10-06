"""The documentation and the deployment files (SPEC sections 39, 42).

Documentation rots quietly. A link that stops resolving, a doc the spec names
that nobody wrote, a requirements file that drifts from ``pyproject.toml`` --
none of these break a test suite, so none of them get noticed until a reader
hits them. These tests make that class of decay fail the build.

Four things are checked:

**Every relative link resolves.** Across ``README.md``, ``docs/*.md``,
``CONTRIBUTING.md`` and the directory READMEs. A broken link in a portfolio
project is read as carelessness, fairly.

**Every doc SPEC section 42 names exists.**

**The deployment files parse and keep their posture.** ``docker-compose.yml``
must declare both services and publish every port to ``127.0.0.1`` only, and the
``Dockerfile`` must run as a non-root user. Neither service has
authentication, so a published port on every interface would be a real
exposure rather than a style question.

**The requirements files agree with ``pyproject.toml``.** They already drifted
once -- ``httpx`` was added to the ``[dev]`` extra in Phase 8 and missed in
``requirements-dev.txt`` -- so the agreement is asserted rather than trusted.

These are checks on the repository, not on the library, so they read files from
the project root rather than importing anything.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

#: Project root, two levels up from this file.
ROOT = Path(__file__).resolve().parents[1]

#: The documents SPEC section 42 names under ``docs/``.
SPEC_DOCS = (
    "architecture.md",
    "data.md",
    "document-processing.md",
    "reconciliation.md",
    "patterns.md",
    "modeling.md",
    "drift.md",
    "experiments.md",
    "limitations.md",
)


#: Markdown files whose links are checked.
def _documents() -> list[Path]:
    """Every markdown file in the repository worth link-checking."""
    found = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    found.extend(sorted((ROOT / "docs").glob("*.md")))
    found.extend(
        ROOT / name / "README.md"
        for name in ("api", "dashboard", "scripts", "notebooks", "data/sample")
        if (ROOT / name / "README.md").is_file()
    )
    return [path for path in found if path.is_file()]


#: ``[text](target)`` where target is not a URL, an anchor or a mail link.
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)\s]+)\)")


def _link_targets(path: Path) -> list[str]:
    """Relative link targets in one markdown file, code fences excluded.

    Fenced blocks are stripped first: they hold shell commands and example
    output, and a path inside one is an illustration rather than a link.
    """
    text = path.read_text(encoding="utf-8")
    outside_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return LINK.findall(outside_fences)


class TestRelativeLinksResolve:
    """No broken relative links anywhere in the documentation."""

    def test_at_least_one_document_is_checked(self) -> None:
        """A guard on the guard: an empty sweep would pass silently."""
        documents = _documents()
        assert len(documents) >= 10, f"only found {len(documents)} markdown files"

    @pytest.mark.parametrize("document", _documents(), ids=lambda path: path.name)
    def test_every_relative_link_points_at_something(self, document: Path) -> None:
        """Each target exists on disk, anchors stripped."""
        broken: list[str] = []
        for target in _link_targets(document):
            # "file.md#section" -> "file.md". The fragment is not checked: a
            # heading can be reworded without the link being wrong, and
            # asserting on slugs would make every rename a test failure.
            bare = target.split("#", 1)[0]
            if not bare:
                continue
            if not (document.parent / bare).resolve().exists():
                broken.append(target)
        assert not broken, f"{document.name} links to missing paths: {broken}"

    def test_the_readme_links_to_the_docs_it_names(self) -> None:
        """The README is the entry point; its links are the ones readers use."""
        targets = {target.split("#", 1)[0] for target in _link_targets(ROOT / "README.md")}
        assert "docs/experiments.md" in targets
        assert "docs/limitations.md" in targets


class TestSpecDocumentsExist:
    """Everything SPEC section 42 lists is present."""

    @pytest.mark.parametrize("name", SPEC_DOCS)
    def test_the_document_exists_and_is_not_a_stub(self, name: str) -> None:
        """Present, and with enough in it to be worth linking to."""
        path = ROOT / "docs" / name
        assert path.is_file(), f"docs/{name} is named in SPEC section 42 but missing"
        assert len(path.read_text(encoding="utf-8").strip()) > 400, f"docs/{name} is a stub"

    def test_the_phase_8_and_9_docs_exist_too(self) -> None:
        """Written after SPEC section 42's list, and linked from the README."""
        assert (ROOT / "docs" / "api.md").is_file()
        assert (ROOT / "docs" / "prediction.md").is_file()


class TestDeploymentFilesParse:
    """The Docker files are well-formed and keep their localhost posture."""

    def test_the_compose_file_parses(self) -> None:
        """Valid YAML, which is the cheapest thing to get wrong."""
        payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert "services" in payload

    def test_both_services_are_declared(self) -> None:
        """The API and the dashboard."""
        payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        assert set(payload["services"]) == {"api", "dashboard"}

    def test_every_published_port_binds_localhost_only(self) -> None:
        """The load-bearing assertion in this class.

        Neither service has any authentication, so a plain ``"8000:8000"``
        mapping -- which binds every interface, usually including the LAN --
        would be a real exposure. The in-container bind is ``0.0.0.0`` by
        necessity: a process on the container's own loopback is unreachable even
        from the sibling container. The guarantee therefore lives here, in the
        published mapping, which is what this checks.
        """
        payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        for name, service in payload["services"].items():
            for mapping in service.get("ports", []):
                assert str(mapping).startswith("127.0.0.1:"), (
                    f"service {name} publishes {mapping} on every interface; "
                    "neither service has authentication"
                )

    def test_the_dashboard_reads_through_the_api(self) -> None:
        """So the compose demo exercises the API rather than bypassing it."""
        payload = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        command = payload["services"]["dashboard"]["command"]
        assert "--api-url" in command
        assert any("http://api:8000" in str(part) for part in command)

    def test_the_dockerfile_has_the_expected_shape(self) -> None:
        """A base image, an exposed pair of ports, and a command."""
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"^FROM python:3\.11", text, re.MULTILINE), "base image is not 3.11-slim"
        assert re.search(r"^EXPOSE .*8000", text, re.MULTILINE)
        assert re.search(r"^CMD ", text, re.MULTILINE)

    def test_the_container_does_not_run_as_root(self) -> None:
        """A service that reads artifacts has no reason to own the filesystem."""
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        users = re.findall(r"^USER\s+(\S+)", text, re.MULTILINE)
        assert users, "no USER directive: the container would run as root"
        assert users[-1] != "root", f"last USER is {users[-1]}"

    def test_the_image_bakes_the_artifacts(self) -> None:
        """Otherwise every page would open on "nothing computed yet".

        A fresh clone has ``data/sample`` but no ``reports/``, which is
        gitignored, so the build has to produce them or the container is a shell
        with nothing to show.
        """
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for script in ("discover_patterns.py", "train_baseline.py", "run_experiment.py"):
            assert script in text, f"the build never runs {script}"

    def test_the_build_fails_on_a_missing_artifact(self) -> None:
        """``serve_api.py --check`` exits non-zero, so a half-built image cannot ship."""
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "serve_api.py --check" in text

    def test_the_sample_data_is_copied_in(self) -> None:
        """The build needs no download, which is what makes it reproducible."""
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "data/sample" in text


def _floors(text: str) -> dict[str, str]:
    """Package floors from a requirements file, comments and -r lines skipped."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(>=|==)\s*([0-9.]+)", stripped)
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(3)
    return found


def _extra_floors(specifiers: list[str]) -> dict[str, str]:
    """Package floors from a list of pyproject dependency specifiers."""
    found: dict[str, str] = {}
    for item in specifiers:
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(>=|==)\s*([0-9.]+)", item)
        if match:
            found[match.group(1).lower().replace("_", "-")] = match.group(3)
    return found


@pytest.fixture(scope="module")
def pyproject() -> dict[str, object]:
    """The parsed pyproject."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


class TestRequirementsAgreeWithPyproject:
    """The reproducibility files must not drift from the real metadata.

    They already did once: ``httpx`` entered the ``[dev]`` extra in Phase 8 and
    was not added to ``requirements-dev.txt``, so anyone installing from the
    requirements files could not run the test suite.
    """

    def test_runtime_requirements_match(self, pyproject: dict[str, object]) -> None:
        """Same packages, same floors."""
        project = pyproject["project"]
        assert isinstance(project, dict)
        declared = _extra_floors(list(project["dependencies"]))
        listed = _floors((ROOT / "requirements.txt").read_text(encoding="utf-8"))
        assert declared == listed, (
            "requirements.txt and pyproject [dependencies] disagree: "
            f"{set(declared.items()) ^ set(listed.items())}"
        )

    def test_dev_requirements_match(self, pyproject: dict[str, object]) -> None:
        """Including httpx, the one that drifted."""
        project = pyproject["project"]
        assert isinstance(project, dict)
        extras = project["optional-dependencies"]
        assert isinstance(extras, dict)
        declared = _extra_floors(list(extras["dev"]))
        text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        listed = _floors(text)
        assert declared == listed, (
            "requirements-dev.txt and pyproject [dev] disagree: "
            f"{set(declared.items()) ^ set(listed.items())}"
        )
        assert "httpx" in listed, "httpx is the dependency that drifted once already"

    def test_dashboard_requirements_match(self, pyproject: dict[str, object]) -> None:
        """streamlit, and nothing else -- the chart libraries come bundled."""
        project = pyproject["project"]
        assert isinstance(project, dict)
        extras = project["optional-dependencies"]
        assert isinstance(extras, dict)
        declared = _extra_floors(list(extras["dashboard"]))
        listed = _floors((ROOT / "requirements-dashboard.txt").read_text(encoding="utf-8"))
        assert declared == listed
        assert set(declared) == {"streamlit"}

    def test_the_requirements_files_chain_rather_than_duplicate(self) -> None:
        """Each builds on requirements.txt, so a floor lives in one place."""
        for name in ("requirements-dev.txt", "requirements-dashboard.txt"):
            text = (ROOT / name).read_text(encoding="utf-8")
            assert "-r requirements.txt" in text, f"{name} does not include requirements.txt"

    def test_the_conda_environment_parses_and_pins_the_ci_python(self) -> None:
        """3.11, matching CI and the pyproject floor."""
        payload = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
        assert payload["name"] == "urbansense"
        dependencies = [item for item in payload["dependencies"] if isinstance(item, str)]
        assert "python=3.11" in dependencies

    def test_the_conda_environment_covers_the_runtime(self, pyproject: dict[str, object]) -> None:
        """Every runtime dependency appears, by conda or by pip."""
        payload = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
        names: set[str] = set()
        for item in payload["dependencies"]:
            if isinstance(item, str):
                names.add(re.split(r"[=><]", item)[0].lower())
            elif isinstance(item, dict):
                for pinned in item.get("pip", []):
                    names.add(re.split(r"[=><]", str(pinned))[0].lower())

        project = pyproject["project"]
        assert isinstance(project, dict)
        for specifier in project["dependencies"]:
            package = re.split(r"[=><\[]", str(specifier))[0].strip().lower()
            assert package in names, f"environment.yml is missing {package}"


class TestEnvExampleIsComplete:
    """Every documented override is discoverable in one place."""

    def test_the_file_covers_the_api_and_dashboard_settings(self) -> None:
        """Added in Phase 9 alongside the two servers."""
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for key in (
            "URBANSENSE_ENV",
            "URBANSENSE_DATA_DIR",
            "URBANSENSE_QUARANTINE_DIR",
            "URBANSENSE_LOG_LEVEL",
            "URBANSENSE_API_HOST",
            "URBANSENSE_API_PORT",
            "URBANSENSE_DASHBOARD_PORT",
            "URBANSENSE_API_URL",
            "URBANSENSE_ROOT",
        ):
            assert key in text, f".env.example does not mention {key}"

    def test_the_environment_variables_the_dashboard_reads_are_documented(self) -> None:
        """The app reads these two; a reader should not have to grep for them."""
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        app = (ROOT / "src" / "urbansense" / "dashboard" / "app.py").read_text(encoding="utf-8")
        for key in re.findall(r'os\.environ\.get\("([A-Z_]+)"', app):
            assert key in text, f"app.py reads {key} but .env.example does not mention it"

    def test_the_file_holds_no_secret_looking_values(self) -> None:
        """It is an example, and this project has no credentials to leak."""
        text = (ROOT / ".env.example").read_text(encoding="utf-8").lower()
        for word in ("password=", "secret=", "token=", "api_key="):
            assert word not in text


class TestLicenceAndContributing:
    """The files a reader checks before using the project."""

    def test_the_licence_is_mit_and_names_a_holder(self) -> None:
        """MIT, with a copyright line that actually names someone.

        It must not assert that the holder is still ``<YOUR NAME>``. The
        placeholder exists so the repository owner fills it in, and a test that
        demanded it stay would fail the moment they did the intended thing --
        which is exactly what happened. What matters is that the line is
        present and not blank.
        """
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "MIT License" in text
        assert "Permission is hereby granted" in text

        holder = re.search(r"^Copyright \(c\) (\d{4})\s+(.+)$", text, re.MULTILINE)
        assert holder, "no 'Copyright (c) <year> <holder>' line"
        assert holder.group(2).strip(), "the copyright holder is blank"

    def test_contributing_states_the_non_negotiable_rules(self) -> None:
        """The data rules are the project's reason for existing."""
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "immutable" in text.lower()
        assert "provenance" in text.lower()
        assert "temporal" in text.lower() or "leakage" in text.lower()

    def test_contributing_documents_the_split_test_run(self) -> None:
        """The whole suite in one process runs out of memory, so say so."""
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "test_experiments" in text


class TestScriptsAreDocumented:
    """Every script exists and is reachable from the docs."""

    def test_every_script_named_in_the_readme_exists(self) -> None:
        """A command a reader copies has to run."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        missing = [
            name
            for name in set(re.findall(r"scripts/([a-z_]+\.py)", text))
            if not (ROOT / "scripts" / name).is_file()
        ]
        assert not missing, f"README names scripts that do not exist: {missing}"

    def test_the_phase_8_and_9_scripts_are_in_the_readme(self) -> None:
        """They are the entry points to the newest work."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in ("run_experiment.py", "serve_api.py", "serve_dashboard.py"):
            assert name in text, f"README does not mention scripts/{name}"

    def test_every_script_has_a_module_docstring(self) -> None:
        """Each explains what it does and how to run it standalone."""
        for path in sorted((ROOT / "scripts").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert text.lstrip().startswith('"""'), f"{path.name} has no module docstring"

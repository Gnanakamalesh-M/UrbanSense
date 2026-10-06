# UrbanSense — developer tasks.
#
# PY can be overridden, e.g. `make test PY=py -3.11`. Targets work from Git Bash
# on Windows and from a POSIX shell.

PY ?= python

.PHONY: help install install-dashboard lint format typecheck test test-fast test-experiments coverage check clean demo-data demo-stream ingest-demo validate-demo reconcile-demo train-baseline discover-patterns predict-demo error-report replay-adaptive experiment serve-api serve-dashboard docker-build docker-up

help: ## Show available targets
	@grep -E "^[a-z-]+:.*?## " $(MAKEFILE_LIST) | sed -e "s/:.*## /\t/"

install: ## Install the package in editable mode with dev dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

install-dashboard: ## Install dev plus the dashboard extra (adds streamlit)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,dashboard]"

lint: ## Check lint rules and formatting without changing files
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format: ## Apply formatting and auto-fixable lint rules
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Run mypy over the package
	$(PY) -m mypy

# Split in two, because the whole suite in one process gets killed for memory on
# a modest machine: the experiment tests generate and replay several datasets.
test: test-fast test-experiments ## Run the whole suite, in two chunks

test-fast: ## Everything except the experiment suite
	$(PY) -m pytest --ignore=tests/test_experiments.py -p no:cacheprovider

test-experiments: ## The experiment suite alone (generates and replays seeds)
	$(PY) -m pytest tests/test_experiments.py -p no:cacheprovider

coverage: ## The whole suite in one process, with coverage (memory-hungry)
	$(PY) -m pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

# The scripts below are the real interface and run standalone with plain python;
# these targets are thin wrappers for convenience. `make` is never required.
demo-data: ## Generate the demo dataset into data/sample/ (add FORCE=1 to replace)
	$(PY) scripts/generate_demo_data.py --out data/sample $(if $(FORCE),--force,)

demo-stream: ## Replay held-out observations as a fast simulated live stream
	$(PY) scripts/simulate_stream.py --limit 20

ingest-demo: ## Ingest the demo dataset and summarize what came out
	$(PY) scripts/ingest_demo.py

validate-demo: ## Validate and normalize the demo dataset, writing quarantine/
	$(PY) scripts/validate_demo.py

reconcile-demo: ## Reconcile the demo dataset into the unified layer
	$(PY) scripts/reconcile_demo.py

train-baseline: ## Train and evaluate the baseline traffic forecasters
	$(PY) scripts/train_baseline.py

discover-patterns: ## Discover patterns, track evolution and scan for anomalies
	$(PY) scripts/discover_patterns.py

predict-demo: ## Predict congestion for one zone and date, with factors and sources
	$(PY) scripts/predict_demo.py

error-report: ## Break prediction errors down by condition, time, zone and slice
	$(PY) scripts/error_report.py

replay-adaptive: ## Replay a drift scenario and compare static against adaptive
	$(PY) scripts/replay_adaptive.py

experiment: ## Compare static, scheduled retraining and adaptive across seeds (~12 min)
	$(PY) scripts/run_experiment.py

serve-api: ## Serve the read-only API on 127.0.0.1:8000 (builds the export if missing)
	$(PY) scripts/serve_api.py

serve-dashboard: ## Serve the Streamlit dashboard on 127.0.0.1:8501
	$(PY) scripts/serve_dashboard.py

docker-build: ## Build the image (bakes the demo artifacts; takes several minutes)
	docker compose build

docker-up: ## Run the API and the dashboard on localhost via docker compose
	docker compose up


clean: ## Remove caches and build artifacts (never touches data/ or quarantine/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

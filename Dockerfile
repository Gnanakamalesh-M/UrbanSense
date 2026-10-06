# UrbanSense — one image, two services (SPEC section 39).
#
# The API and the dashboard run from the same image because the expensive part
# of the build is baking the demo artifacts, and building that twice would be
# pure waste. docker-compose picks which one to start.
#
# Three properties this image is built for:
#
#   Nothing is downloaded at runtime. Dependencies are installed and every
#   artifact is generated during the build, so a started container only ever
#   opens files. No network is needed once it is built.
#
#   It runs as a non-root user. The application reads artifacts and serves
#   them; it has no reason to own the filesystem.
#
#   It is useful the moment it starts. A fresh clone has data/sample but no
#   reports/ (gitignored), so without the bake below every dashboard page would
#   open on "nothing computed yet". The pipeline therefore runs once at build
#   time and the results ship inside the image.
#
# Python 3.11 matches the version CI pins and the floor pyproject declares.

FROM python:3.11-slim

# Fail fast and keep the image small: no .pyc files, unbuffered logs so
# `docker logs` shows progress during the long artifact bake.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency metadata first, so a code-only change does not reinstall
# scikit-learn and streamlit.
COPY pyproject.toml README.md ./
COPY src/urbansense/__init__.py src/urbansense/__init__.py
RUN python -m pip install --upgrade pip \
 && python -m pip install ".[dashboard]"

# Then the application and its committed inputs. data/sample is copied in
# deliberately: it holds the generated dataset and its ground truth, which is
# what makes the build reproducible without fetching anything.
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/sample/ data/sample/
COPY models/registry/ models/registry/
COPY Makefile ./

# Install the package itself now that the source is present.
RUN python -m pip install --no-deps -e .

# Bake the artifacts. Every page of the dashboard and every endpoint of the API
# reads one of these, and generating them here is what keeps the running
# container free of heavy work.
#
# The experiment runs one seed over a 60-day replay rather than the
# pre-registered five seeds over the full stream: five seeds take about twelve
# minutes, which is too long to ask of an image build. The report records what
# it actually ran, so the shipped numbers cannot be mistaken for the registered
# experiment -- for that, run `python scripts/run_experiment.py` in the
# container or on the host.
#
# `serve_api.py --check` is last on purpose: it exits non-zero if any artifact
# family is missing, so the build fails rather than shipping a half-populated
# image.
RUN python scripts/validate_demo.py \
 && python scripts/reconcile_demo.py \
 && python scripts/train_baseline.py \
 && python scripts/discover_patterns.py \
 && python scripts/predict_demo.py --zone ZONE-A \
 && python scripts/error_report.py \
 && python scripts/replay_adaptive.py \
 && python scripts/run_experiment.py --seeds 20260101 --replay-days 60 \
 && python scripts/serve_api.py --check

# Non-root. Created after the build steps so the baked artifacts can be handed
# over in one chown rather than the build running unprivileged and fighting for
# write access to /app.
RUN useradd --create-home --shell /usr/sbin/nologin urbansense \
 && chown -R urbansense:urbansense /app
USER urbansense

# 8000 the API, 8501 the dashboard. Publishing is docker-compose's job, and it
# binds these to 127.0.0.1 only -- neither service has any authentication.
EXPOSE 8000 8501

# Inside the container the servers must listen on 0.0.0.0, or nothing outside
# the container's own network namespace could reach them. That is not a weaker
# posture than the host default: the port is only reachable because
# docker-compose publishes it, and it publishes to 127.0.0.1.
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0"]

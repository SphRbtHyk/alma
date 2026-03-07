FROM python:3.12-slim

WORKDIR /app

# System build dependencies required by some ML packages (gcc, g++, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install an up-to-date Rust toolchain.
# python:3.12-slim ships Rust 1.86; tokenizers ≥0.21 requires ≥1.88.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# ------------------------------------------------------------------
# Install Python dependencies.
# We copy only the package manifest first so Docker can cache this
# layer and skip re-installing everything when only source changes.
# ------------------------------------------------------------------
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

RUN pip install -U pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0"

# ── spaCy models ──────────────────────────────────────────────────────────────
# Standard models from spaCy's official release channel.
# `python -m spacy download` resolves the correct version for the installed
# spaCy automatically.
RUN python -m spacy download en_core_web_lg \
    && python -m spacy download fr_core_news_lg \
    && python -m spacy download it_core_news_lg

# Community / HuggingFace-hosted models (not on the official spaCy hub).
RUN python -m pip install "la-core-web-lg @ https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl"
RUN python -m pip install "https://huggingface.co/chcaa/grc_odycy_joint_trf/resolve/main/grc_odycy_joint_trf-0.7.0-py3-none-any.whl"

# The backend source directory is mounted at runtime via docker-compose
# volume, so we do NOT COPY it here – that would defeat live-reload.
WORKDIR /app/web_app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

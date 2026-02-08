#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="agentic-rag"
CONDA_BIN="/home/nuku/miniconda3/bin/conda"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] conda not found at ${CONDA_BIN}" >&2
  exit 1
fi

exec "${CONDA_BIN}" run -n "${ENV_NAME}" python -m src.app.cli_chat "$@"

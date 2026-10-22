#!/usr/bin/env bash
set -euo pipefail
cd /workspace/resumeanalysis
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_XET=1
uv sync --python 3.12 --frozen --extra dev --extra data
if [ ! -f artifacts/e5/resumeanalysis_model.json ]; then
  .venv/bin/resumeanalysis download-e5
fi
if [ ! -f artifacts/run/model.pt ]; then
  if [ ! -f data/raw/candidate-matching-synthetic/manifest.json ]; then
    .venv/bin/resumeanalysis download-data
  fi
  .venv/bin/resumeanalysis prepare --max-jobs 250 --pairs-per-job 2 --seed 42
  .venv/bin/resumeanalysis train --config configs/cpu.json
fi
mkdir -p artifacts
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/resumeanalysis predict --input examples/pair.json > artifacts/setup_prediction.json

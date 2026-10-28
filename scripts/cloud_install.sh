#!/usr/bin/env bash
set -euo pipefail
resumeanalysis_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$resumeanalysis_root"
export HF_HUB_DISABLE_XET=1
uv sync --python 3.12 --frozen --extra dev --extra qwen --extra profile-llm --extra documents
if [ ! -f artifacts/e5/resumeanalysis_model.json ]; then
  .venv/bin/resumeanalysis download-e5
fi
mkdir -p artifacts
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/resumeanalysis recommend \
    --profile examples/recommendation_profile.json \
    --vacancies examples/recommendation_vacancies.json \
    --database artifacts/setup-preferences.sqlite > artifacts/setup_recommendations.json

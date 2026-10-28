#!/usr/bin/env bash
# Run from the repository root with the project Python environment active.
set -euo pipefail
python -m resumeanalysis_legacy.skill_data download
python -m resumeanalysis_legacy.skill_data guard \
  --input data/skill-foundation/raw/skillspan/2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b/data/json/test.json \
  --output artifacts/skill-foundation/skillspan-test-guard.json
python -m resumeanalysis_legacy.skill_data preprocess
python -m resumeanalysis_legacy.skill_data audit \
  --guard artifacts/skill-foundation/skillspan-test-guard.json
python -m resumeanalysis_legacy.skill_data coverage
python -m resumeanalysis_legacy.skill_data report

# ResumeAnalysis cloud startup

Each cloud task is already isolated. Use the existing checkout at
`/workspace/resumeanalysis`; do not create a Git worktree unless the user explicitly
requests one. Preserve user edits, source, tests, manifests, and lockfiles.

No database, web service, GPU, API key, or daemon is needed. Files and installed
dependencies can survive publication; do not assume shell activation or processes do.

1. Work from `/workspace/resumeanalysis`. Use `.venv/bin/python` and
   `.venv/bin/resumeanalysis`, or activate `.venv/bin/activate` for the current shell.
   CPU Python 3.12, torch, transformers and scikit-learn are installed from `uv.lock`.
2. Verify retained artifacts: `artifacts/e5/resumeanalysis_model.json`,
   `artifacts/e5/model.safetensors`, and `artifacts/run/model.pt`.
   If initial setup files are missing, run `bash scripts/cloud_install.sh`;
   this bootstraps only missing artifacts and does not overwrite an existing head.
   An incompatible existing head requires an explicit new training output path,
   not silently replacing user weights.
3. Readiness check from the checkout root:

   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     .venv/bin/resumeanalysis predict --input examples/pair.json
   ```

   Verify valid `VacancyAnalysis` JSON, Python/SQL in `matched_skills`, Airflow in
   `unknown_skills`, and null reserved fields. The saved smoke model returns relevant;
   do not treat its saturated score as a calibrated probability.
4. For development verification use
   `RESUMEANALYSIS_RUN_E5_TESTS=1 .venv/bin/pytest -q`,
   `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .`.
   For saved-model metrics use `.venv/bin/resumeanalysis evaluate`; retained processed
   data and logistic weights are required. Do not retrain just to start inference.
5. Follow README for new data and full training. E5/fit data downloads need Hugging Face
   and its CDN destinations, but inference is fully local with `local_files_only=True`.
   No secrets are required for the public artifacts. Use the existing platform GitHub
   proxy/authentication for Git operations, never request a token solely because a
   token variable is absent.

The present model is a CPU smoke experiment on Candidate Matching Synthetic, not a
validated employer-fit model. The RU/EN dictionary has low recall on full SkillSpan.
`med2425` is excluded because its card does not declare a license. See
`docs/evaluation.md` and `docs/research.md`; do not claim unrun NER/full benchmark checks.

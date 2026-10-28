# Historical matching and research

This optional workspace distribution, `resumeanalysis-legacy`, retains the old
matching and research code under `resumeanalysis_legacy`. It is excluded from the
normal runtime installation and the `resumeanalysis` wheel. Git history, original
licenses, saved checkpoints and historical reports are preserved.

Run commands from the repository root:

```bash
uv sync --frozen --group research --extra dev --extra qwen --extra profile-llm --extra documents
source .venv/bin/activate
resumeanalysis-legacy --help
resumeanalysis-legacy predict --input examples/pair.json
resumeanalysis-legacy-qwen --help
pytest -q legacy/tests
```

## Locations and compatibility

| Previous entry | Archived entry |
|---|---|
| `resumeanalysis predict`, `rank`, `train`, `evaluate`, `prepare`, `datasets`, `extractors`, `skillspan`, `analyze` | Same subcommand under `resumeanalysis-legacy` |
| `resumeanalysis-qwen` | `resumeanalysis-legacy-qwen` |
| `resumeanalysis.inference`, `models`, `features`, `postprocessing`, `training`, `data`, `experiments`, `skill_data`, `rerank` | Same module suffix under `resumeanalysis_legacy` |
| `resumeanalysis.qwen.predictor`, `prompts`, `cli`, classification `config` | Same suffix under `resumeanalysis_legacy` |
| `resumeanalysis.extraction.cli`, train-only `build_lexicon` | `resumeanalysis_legacy.extraction` |
| `resumeanalysis.integration.cover_letters` transport adapters | `resumeanalysis_legacy.integration.cover_letters` |
| Old `LocalInferenceClient`, old `AnalysisService` | `resumeanalysis_legacy.integration.applypilot`, `resumeanalysis_legacy.cover_letters.service` |
| Research configuration and scripts | `legacy/configs/`, `legacy/scripts/` |
| Extractor comparison notebook | `legacy/notebooks/skill_extractor_comparison.ipynb` |

Legacy features and all three scorers reuse the runtime's common evidence,
normalization and model-generation utilities. `matching/` owns shared rules;
`cover_letters/qwen_runtime.py` owns local Qwen generation. A general-purpose
PyTorch network and saved matching checkpoints remain available only through the
archived package. They are qualification experiments, not validated personal
interest models.

Examples:

```bash
resumeanalysis-legacy train --config legacy/configs/cpu.json --output artifacts/research-new-run
resumeanalysis-legacy predict --backend rerank --rerank-config legacy/configs/rerank.json \
  --input examples/pair.json
python -m resumeanalysis_legacy.skill_data --help
python legacy/scripts/benchmark_online_recommender.py --output artifacts/new-online-benchmark
```

Do not overwrite existing checkpoints when running a new experiment. Dataset
licenses, split controls and held-out-test rules still apply. No benchmark or
training is run automatically during normal installation or cloud setup.

Historical documents and reports retain their original observations. For commands
in older documents, use this path/namespace mapping. The old multi-resume
`analyze` service and Kafka adapters are retained for research compatibility; new
applications should use the single local `ApplyPilotClient` and explicit
selected-vacancy letter drafting.

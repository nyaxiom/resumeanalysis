"""Guard evaluation boundaries and production/benchmark input equivalence."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from resumeanalysis import cli
from resumeanalysis.cover_letters import cli as letter_cli
from resumeanalysis.cover_letters.generator import LocalCoverLetterGenerator
from resumeanalysis.cover_letters.schema import Resume
from resumeanalysis.qwen.config import QwenConfig
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location(
        "cover_letter_benchmark", Path("scripts/benchmark_cover_letters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_and_production_use_identical_structured_inputs(benchmark):
    calls = []
    runtime = SimpleNamespace(
        generate=lambda messages, schema: calls.append(messages) or "I use Python."
    )
    config = QwenConfig(
        device="cpu", quantization="none", dtype="float32", structured_generation=False
    )
    case = {
        "profile": {"skills": ["Python"], "experience_years": 2},
        "vacancy": {"title": "Developer", "min_experience_years": 1.5},
        "language": "en",
    }
    asyncio.run(
        LocalCoverLetterGenerator(config, runtime).generate(
            Vacancy.model_validate(case["vacancy"]),
            Resume(id=1, profile=Profile.model_validate(case["profile"])),
        )
    )
    assert calls == [benchmark.candidate_messages("applykit", case)]
    lines = calls[0][1]["content"].splitlines()
    profile = json.loads(lines[0].split("=", 1)[1])
    vacancy = json.loads(lines[1].split("=", 1)[1])
    assert type(profile["experience_years"]) is int
    assert vacancy["min_experience_years"] == 1.5


@pytest.mark.parametrize("selection", [None, {"signature": {}}])
def test_test_generation_rejects_missing_or_changed_selection_before_model_load(
    benchmark, monkeypatch, tmp_path, selection
):
    def forbidden(*args, **kwargs):
        raise AssertionError("The model must not load before the split guard passes")

    monkeypatch.setattr(benchmark, "TransformersGenerator", forbidden)
    arguments = ["benchmark", "--split", "test", "--output", str(tmp_path / "result.json")]
    if selection is not None:
        path = tmp_path / "selection.json"
        path.write_text(json.dumps(selection))
        arguments.extend(["--selection", str(path)])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="frozen validation|changed after validation"):
        benchmark.main()
    assert not (tmp_path / "result.json").exists()


def test_analyze_cli_retains_mlp_default_and_emits_selected_resume(monkeypatch, tmp_path, capsys):
    calls = []
    source = tmp_path / "request.json"
    source.write_text(
        json.dumps({"vacancy": {"title": "Developer"}, "resumes": [{"id": 7, "profile": {}}]})
    )
    result = VacancyAnalysis(
        relevant=False,
        relevance=0.1,
        matched_skills=[],
        missing_skills=[],
        unknown_skills=[],
        contradicted_skills=[],
        experience_gap_years=None,
        explanation="No candidate evidence",
    )

    def predictor(name, **kwargs):
        calls.append(name)
        return SimpleNamespace(predict=lambda profile, vacancy: result)

    monkeypatch.setattr(letter_cli, "TextEncoder", lambda path: object())
    monkeypatch.setattr(letter_cli, "create_predictor", predictor)
    monkeypatch.setattr(sys, "argv", ["resumeanalysis", "analyze", "--input", str(source)])
    assert cli.main() is None
    output = json.loads(capsys.readouterr().out)
    assert calls == ["mlp"]
    assert output["resume_id"] == 7 and output["relevance"] == 0.1
    assert output["cover_letter"] is None


def test_analyze_cli_rejects_malformed_request_before_model_load(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid requests must not initialize model assets")

    source = tmp_path / "request.json"
    source.write_text('{"vacancy": {}, "resumes": [{"id": true, "profile": {}}]}')
    monkeypatch.setattr(letter_cli, "TextEncoder", forbidden)
    monkeypatch.setattr(sys, "argv", ["resumeanalysis", "analyze", "--input", str(source)])
    with pytest.raises(ValueError):
        cli.main()


def test_historical_replay_rejects_changed_saved_prompts_before_inference(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "replay_cover_letters", Path("scripts/replay_cover_letter_prompts.py")
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    report, config = replay.read_report("evaluation/cover_letters/initial_4b_validation.json")
    assert config.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    report["results"][0]["messages"][0]["content"] += "Changed prompt"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="prompt hash"):
        replay.read_report(changed)

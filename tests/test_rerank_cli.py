"""CLI routing uses authored inputs and injected local backends, without model downloads."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from resumeanalysis import cli
from resumeanalysis.extraction.registry import is_legacy
from resumeanalysis.inference import backends
from resumeanalysis.rerank import assets
from resumeanalysis.schemas import VacancyAnalysis


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))
    return str(path)


def run_cli(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["resumeanalysis", *arguments])
    monkeypatch.setattr(cli.torch, "set_num_threads", lambda value: None)
    return cli.main()


def analysis():
    return VacancyAnalysis(
        relevant=False,
        relevance=0.6,
        matched_skills=[],
        missing_skills=["Python"],
        unknown_skills=["Python"],
        contradicted_skills=[],
        experience_gap_years=None,
        explanation="Uncalibrated document score; skill evidence needs verification.",
    )


@pytest.fixture
def settings(tmp_path):
    return write_json(
        tmp_path / "config.json",
        {
            "model_path": str(tmp_path / "no-weights"),
            "batch_size": 2,
            "max_length": 128,
            "long_document_policy": "error",
            "max_windows": 4,
            "overlap_tokens": 16,
        },
    )


def backend_spy(monkeypatch):
    calls = []

    class StubEncoder:
        identity = "authored-test-encoder"

        def __init__(self, model_path):
            calls.append(("encoder", model_path))

    class StubBackend:
        def predict(self, profile, vacancy):
            calls.append(("predict", profile, vacancy))
            return analysis()

        def rank(self, profile, vacancies, **kwargs):
            calls.append(("rank", profile, vacancies, kwargs))
            return [
                SimpleNamespace(
                    model_dump=lambda: {
                        "index": 0,
                        "analysis": analysis().model_dump(),
                        "rerank_score": 0.4,
                        "retrieval_score": 0.8,
                    }
                )
            ]

    def create(name, **kwargs):
        calls.append(("factory", name, kwargs))
        return StubBackend()

    monkeypatch.setattr(cli, "TextEncoder", StubEncoder)
    monkeypatch.setattr(backends, "create_predictor", create)
    return calls


def test_predict_rerank_preserves_pair_contract_and_legacy_cli_default(
    monkeypatch, tmp_path, settings, capsys
):
    calls = backend_spy(monkeypatch)
    source = write_json(
        tmp_path / "pair.json",
        {"profile": {"resume_text": "Изучаю Python"}, "vacancy": {"description": "Python needed"}},
    )
    result = run_cli(
        monkeypatch,
        ["predict", "--backend", "rerank", "--input", source, "--rerank-config", settings],
    )
    assert result is None  # Setuptools console entry must exit successfully.
    output = json.loads(capsys.readouterr().out)
    assert output == analysis().model_dump()
    factory = next(call for call in calls if call[0] == "factory")
    assert factory[1] == "rerank"
    assert is_legacy(factory[2]["extractor"])
    assert factory[2]["config"].model_path.endswith("no-weights")
    assert factory[2]["decision_logit_threshold"] == 0.0
    assert next(call for call in calls if call[0] == "predict")[1].resume_text == "Изучаю Python"


def test_rank_forwards_limits_v2_and_raw_threshold(monkeypatch, tmp_path, settings, capsys):
    calls = backend_spy(monkeypatch)
    source = write_json(
        tmp_path / "pool.json",
        {
            "profile": {"skills": ["Python"], "preferred_roles": ["Developer"]},
            "vacancies": [{"description": "Python or Java"}, {"description": "Docker"}],
        },
    )
    assert (
        run_cli(
            monkeypatch,
            [
                "rank",
                "--input",
                source,
                "--rerank-config",
                settings,
                "--preselect-k",
                "2",
                "--top-k",
                "1",
                "--decision-logit-threshold",
                "1.25",
            ],
        )
        is None
    )
    output = json.loads(capsys.readouterr().out)
    assert output[0]["index"] == 0
    assert output[0]["rerank_score"] == 0.4
    factory = next(call for call in calls if call[0] == "factory")
    assert not is_legacy(factory[2]["extractor"])
    assert factory[2]["decision_logit_threshold"] == 1.25
    rank = next(call for call in calls if call[0] == "rank")
    assert rank[1].preferred_roles == ["Developer"]
    assert rank[3] == {"preselect_k": 2, "top_k": 1}


def test_empty_rank_does_not_load_model_assets(monkeypatch, tmp_path, settings, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty candidate pool must not initialize models")

    monkeypatch.setattr(cli, "TextEncoder", forbidden)
    monkeypatch.setattr(backends, "create_predictor", forbidden)
    source = write_json(tmp_path / "pool.json", {"profile": {}, "vacancies": []})
    assert run_cli(monkeypatch, ["rank", "--input", source, "--rerank-config", settings]) is None
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize(
    "document",
    [
        {"profile": {}, "vacancies": [], "hidden": "unexpected"},
        {"profile": {}, "vacancies": {}},
        {"profile": {"skills": "Python"}, "vacancies": []},
        {"profile": {}, "vacancies": [{"remote": "false"}]},
        {"profile": {}, "vacancy": {}},
    ],
)
def test_rank_invalid_input_fails_before_loading_models(
    monkeypatch, tmp_path, settings, capsys, document
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed input must be rejected before initialization")

    monkeypatch.setattr(cli, "TextEncoder", forbidden)
    source = write_json(tmp_path / "invalid.json", document)
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, ["rank", "--input", source, "--rerank-config", settings])
    assert error.value.code == 2
    assert "Invalid reranker input" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments", [["--preselect-k", "0"], ["--top-k", "-1"], ["--decision-logit-threshold", "nan"]]
)
def test_invalid_rank_parameters_fail_before_configuration_or_models(
    monkeypatch, capsys, arguments
):
    with pytest.raises(SystemExit) as error:
        run_cli(monkeypatch, ["rank", "--input", "absent.json", *arguments])
    assert error.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_bad_configuration_and_duplicate_input_keys_are_rejected(
    monkeypatch, tmp_path, settings, capsys
):
    source = write_json(tmp_path / "pool.json", {"profile": {}, "vacancies": []})
    bad = write_json(tmp_path / "bad.json", {"batch_size": 0})
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["rank", "--input", source, "--rerank-config", bad])
    assert "valid reranker configuration" in capsys.readouterr().err
    (tmp_path / "pool.json").write_text('{"profile":{},"profile":{},"vacancies":[]}')
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["rank", "--input", source, "--rerank-config", settings])
    assert "Invalid reranker input" in capsys.readouterr().err


def test_lexical_smoke_remains_explicit(monkeypatch, tmp_path, settings, capsys):
    calls = backend_spy(monkeypatch)
    source = write_json(tmp_path / "pair.json", {"profile": {}, "vacancy": {}})
    run_cli(
        monkeypatch,
        [
            "predict",
            "--backend",
            "rerank",
            "--input",
            source,
            "--rerank-config",
            settings,
            "--smoke-lexical",
        ],
    )
    kwargs = next(call for call in calls if call[0] == "factory")[2]
    assert kwargs["encoder"].identity.startswith("lexical-smoke")
    assert not any(call[0] == "encoder" for call in calls)
    capsys.readouterr()


def test_download_command_is_explicit_and_successful(monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "download_reranker", lambda output: calls.append(output))
    assert run_cli(monkeypatch, ["download-reranker", "--output", "artifacts/fixture"]) is None
    assert calls == ["artifacts/fixture"]


def test_module_help_lists_ranking_without_initializing_assets():
    result = subprocess.run(
        [sys.executable, "-m", "resumeanalysis", "rank", "--help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0
    assert "--preselect-k" in result.stdout and "--rerank-config" in result.stdout
    assert not result.stderr

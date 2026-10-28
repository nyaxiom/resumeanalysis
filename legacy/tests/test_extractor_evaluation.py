import dataclasses
import json
import re
from pathlib import Path

import pytest
from resumeanalysis_legacy.data.skillspan import evaluate_skillspan as legacy_skillspan
from resumeanalysis_legacy.experiments.extractor_evaluation import (
    benchmark_skillspan,
    comparison_html,
    evaluate_fixtures,
    evaluate_role_radar,
    evaluate_skillspan,
    save_comparison,
)
from resumeanalysis_legacy.extraction.cli import configure_parser, run

from resumeanalysis.extraction.skill_extractor import Extraction, SkillExtractor, SkillMention


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def skillspan_row(tokens, knowledge, skills=None):
    return {
        "tokens": tokens,
        "tags_knowledge": knowledge,
        "tags_skill": skills or ["O"] * len(tokens),
    }


def test_full_skillspan_counts_equal_legacy_and_errors_are_deterministic(tmp_path):
    source = write_rows(
        tmp_path / "spans.jsonl",
        [
            skillspan_row(["Python", "and", "Docker"], ["B", "O", "O"]),
            skillspan_row(["an", "unrecognized", "competency"], ["O", "B", "I"]),
        ],
    )
    result = evaluate_skillspan(source, SkillExtractor(), include_records=True)
    baseline = legacy_skillspan(source)
    assert result["rows"] == baseline["rows"] == 2
    assert result["exact_span"] == baseline["exact_span"]
    assert result["token"] == baseline["token"]
    assert result["exact_span"]["true_positive"] == 1
    assert result["exact_span"]["false_positive"] == result["exact_span"]["false_negative"] == 1
    assert result["common_false_positive_spans"] == [{"text": "Docker", "count": 1}]
    assert result["common_missed_spans"] == [{"text": "unrecognized competency", "count": 1}]
    assert len(result["record_diagnostics"]) == 2
    assert result["source"]["revision"] is None
    assert "Caller-supplied" in result["source"]["dataset"]
    assert result["performance"]["process_cpu_seconds"] >= 0
    repeated = evaluate_skillspan(source, SkillExtractor())
    assert repeated["error_examples"] == result["error_examples"]


def test_exact_span_and_token_overlap_are_distinct(tmp_path):
    source = write_rows(tmp_path / "spans.jsonl", [skillspan_row(["Python", "3"], ["B", "O"])])
    result = evaluate_skillspan(source, SkillExtractor())
    assert result["exact_span"]["f1"] == 0
    assert result["token"]["recall"] == 1
    assert result["token"]["precision"] == 0.5


@pytest.mark.parametrize(
    "text",
    [
        "\n",
        "not json\n",
        '{"tokens": []}\n',
        '{"tokens":["Python"],"tags_skill":[],"tags_knowledge":["B"]}\n',
        '{"tokens":[3],"tags_skill":["O"],"tags_knowledge":["B"]}\n',
    ],
)
def test_invalid_skillspan_rows_are_not_silently_filtered(tmp_path, text):
    source = tmp_path / "bad.jsonl"
    source.write_text(text)
    with pytest.raises(ValueError):
        evaluate_skillspan(source, SkillExtractor())


def test_invalid_extractor_offsets_fail_instead_of_disappearing(tmp_path):
    class BadExtractor:
        def extract(self, text):
            return Extraction(
                mentions=[SkillMention("python", "not input", 0, 1, "confirmed", "mentioned")]
            )

    source = write_rows(tmp_path / "spans.jsonl", [skillspan_row(["Python"], ["B"])])
    with pytest.raises(ValueError, match="invalid source span"):
        evaluate_skillspan(source, BadExtractor())


def test_source_changes_during_skillspan_evaluation_are_detected(tmp_path):
    source = write_rows(tmp_path / "spans.jsonl", [skillspan_row(["Python"], ["B"])])

    class MutatingExtractor(SkillExtractor):
        def extract(self, text):
            source.write_text("changed source\n")
            return super().extract(text)

    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluate_skillspan(source, MutatingExtractor())


def test_controlled_fixture_languages_and_categories_are_separate(tmp_path):
    fixtures = [
        {
            "id": "en",
            "language": "en",
            "text": "Required: Python",
            "vacancy": True,
            "expected": {"required": ["python"], "optional": []},
        },
        {
            "id": "ru",
            "language": "ru",
            "text": "Желательно Docker",
            "vacancy": True,
            "expected": {"required": [], "optional": ["docker"]},
        },
    ]
    source = tmp_path / "fixtures.json"
    source.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = evaluate_fixtures(source, SkillExtractor())
    assert result["languages"]["en"]["categories"]["required"]["f1"] == 1
    assert result["languages"]["ru"]["categories"]["optional"]["f1"] == 1
    assert result["languages"]["en"]["rows"] == result["languages"]["ru"]["rows"] == 1
    assert result["languages"]["en"]["categories"]["mentioned"] is None
    assert all(not row["errors"] for row in result["records"])
    fixtures[1]["expected"]["unsupported"] = []
    source.write_text(json.dumps(fixtures))
    with pytest.raises(ValueError, match="schema"):
        evaluate_fixtures(source, SkillExtractor())


def test_alternative_groups_and_negated_categories_are_not_flattened(tmp_path):
    @dataclasses.dataclass
    class Group:
        skills: tuple

    @dataclasses.dataclass
    class RichExtraction(Extraction):
        alternative_groups: list = dataclasses.field(default_factory=list)
        negated_requirements: set = dataclasses.field(default_factory=set)

    class Extractor:
        def extract(self, text, vacancy=False):
            return RichExtraction(
                alternative_groups=[Group(("python", "node.js"))],
                negated_requirements={"kubernetes"},
            )

    fixture = [
        {
            "language": "en",
            "text": "Python or Node.js; no Kubernetes required",
            "expected": {
                "required": [],
                "negated_requirements": ["kubernetes"],
                "alternative_groups": [["node.js", "python"]],
            },
        }
    ]
    source = tmp_path / "fixtures.json"
    source.write_text(json.dumps(fixture))
    result = evaluate_fixtures(source, Extractor())
    assert not result["records"][0]["errors"]
    assert result["languages"]["en"]["categories"]["alternative_groups"]["f1"] == 1


def test_role_radar_scores_are_metadata_and_every_row_has_an_outcome(tmp_path):
    valid = {
        "profile": {"skills": ["Python"], "contact": {"email": "private@example.invalid"}},
        "vacancy": {"description": "Required: Python, Docker", "required_skills": ["NewSkill"]},
        "composite_score": 0.79,
        "label": 1,
        "metadata": {"private": "confidential"},
    }
    source = tmp_path / "pairs.jsonl"
    source.write_text(json.dumps(valid) + "\nnot JSON\n\n")
    result = evaluate_role_radar(source, SkillExtractor())
    assert result["input_records"] == 3 and result["count_reconciliation"]
    assert result["outcome_counts"] == {"diagnosed_with_unmapped_fields": 1, "rejected": 2}
    assert result["accuracy_metrics"] is None
    row = result["record_outcomes"][0]
    assert row["source_scores_metadata"] == {"composite_score": 0.79}
    assert row["effective_required_skills"] == ["docker", "newskill", "python"]
    assert row["required_not_in_explicit_profile"] == ["docker", "newskill"]
    assert "/profile/contact/email" in row["unmapped_fields"]
    assert "/label" in row["unmapped_fields"]
    assert "private@example" not in json.dumps(result)
    assert "confidential" not in json.dumps(result)
    assert all("source_sha256" in row for row in result["record_outcomes"])
    assert row["model_accuracy_metrics"] is None


def test_role_radar_missing_input_or_malformed_known_fields_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="unavailable"):
        evaluate_role_radar(tmp_path / "missing.jsonl", SkillExtractor())
    source = write_rows(
        tmp_path / "pairs.jsonl",
        [
            {"profile": {"skills": [123]}, "vacancy": {"description": "Python"}},
            {"profile": {}, "vacancy": {"title": "Python engineer"}},
            {"profile": {}, "vacancy": {"description": "Python"}, "score": "not a numeric score"},
        ],
    )
    result = evaluate_role_radar(source, SkillExtractor())
    assert result["outcome_counts"] == {"rejected": 3}
    assert result["count_reconciliation"] and result["accuracy_metrics"] is None


def test_shared_comparison_provenance_and_standalone_escaped_report(tmp_path):
    source = write_rows(tmp_path / "spans.jsonl", [skillspan_row(["Python"], ["B"])])
    result = benchmark_skillspan(source, ("legacy", "v2"), isolated=False)
    assert (
        result["backends"]["legacy"]["source"]["sha256"]
        == result["backends"]["v2"]["source"]["sha256"]
    )
    assert "Shared process" in result["memory_comparison_scope"]
    result["backends"]["<script>unsafe</script>"] = result["backends"].pop("legacy")
    markup = comparison_html(result)
    assert "<svg" in markup and "&lt;script&gt;unsafe" in markup
    assert "<script>unsafe" not in markup
    save_comparison(result, tmp_path / "report.json")
    assert (tmp_path / "report.html").is_file()
    with pytest.raises(ValueError, match="Unique"):
        benchmark_skillspan(source, ("legacy", "legacy"))


def test_cli_extract_compare_and_input_overwrite_refusal(tmp_path, capsys):
    import argparse

    parser = configure_parser(argparse.ArgumentParser())
    args = parser.parse_args(
        ["extract", "--extractor", "legacy", "--text", "Required: Python", "--vacancy"]
    )
    assert run(args, parser)["required"] == ["python"]
    assert "legacy" in capsys.readouterr().out
    source = write_rows(tmp_path / "spans.jsonl", [skillspan_row(["Python"], ["B"])])
    args = parser.parse_args(
        ["skillspan", "--extractor", "legacy", "--input", str(source), "--output", str(source)]
    )
    with pytest.raises(SystemExit):
        run(args, parser)
    args = parser.parse_args(
        ["compare", "--input", str(source), "--output", str(tmp_path / "isolated.json")]
    )
    result = run(args, parser)
    assert "independent subprocess" in result["memory_comparison_scope"]
    assert result["backends"]["legacy"]["exact_span"]["f1"] == 1


def test_notebook_is_clean_json_and_code_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks/skill_extractor_comparison.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert cells
    for index, cell in enumerate(cells):
        assert not cell["outputs"] and cell["execution_count"] is None
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    assert re.search("SKILL_EXTRACTOR_SMOKE", path.read_text())

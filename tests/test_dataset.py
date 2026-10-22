import pytest

from resumeanalysis.data.dataset import PairRecord, assert_no_leakage, grouped_split
from resumeanalysis.data.prepare import string_list
from resumeanalysis.data.skillspan import bio_spans
from resumeanalysis.schemas import Profile, Vacancy


def record(i, pid=None, jid=None, text=None):
    return PairRecord(
        candidate_id=pid or f"p{i}",
        vacancy_id=jid or f"v{i}",
        profile=Profile(resume_text=text or f"Candidate content {i}"),
        vacancy=Vacancy(description=f"Vacancy content {i}"),
        label=i % 2,
        provenance="test fixture",
    )


def test_grouped_split_keeps_transitive_entities_together():
    rows = [record(i) for i in range(30)]
    rows += [record(101, pid="p0", jid="v1"), record(102, pid="p1", jid="v2")]
    splits = grouped_split(rows)
    assert_no_leakage(splits)
    members = [{r.candidate_id for r in rs} for rs in splits.values()]
    assert any({"p0", "p1", "p2"}.issubset(s) for s in members)
    assert sum(map(len, splits.values())) == len(rows)


def test_duplicate_normalized_text_with_changed_ids_is_leakage():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(
            {"train": [record(1, text="PYTHON  SQL")], "test": [record(2, text="python sql")]}
        )


def test_unsplittable_graph_fails():
    with pytest.raises(ValueError, match="independent"):
        grouped_split([record(i, pid="same-candidate") for i in range(10)])


def test_source_arrays_are_not_assumed_comma_separated():
    assert string_list('["SQL", "Python"]') == ["SQL", "Python"]
    with pytest.raises(ValueError):
        string_list("SQL, Python")


def test_bio_spans():
    assert bio_spans(["a", "b", "c", "d"], ["B", "I", "O", "B"]) == {(0, 2), (3, 4)}

import pytest

from resumeanalysis.extraction.skill_extractor import SkillExtractor


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python 3, Postgres, GNU/Linux", {"python", "postgresql", "linux"}),
        ("Нужны Python, SQL и Linux", {"python", "sql", "linux"}),
        ("C++ C# node.js scikit-learn", {"c++", "c#", "node.js", "scikit-learn"}),
        ("JavaScript and NoSQL", {"javascript"}),
        ("NoSQL; GitHub; an airflowing stream", set()),
        ("Data Analyst / ML Engineer", set()),
        ("Torch tensors in Python", {"pytorch", "python"}),
        ("Torch in Lua", {"torch"}),
    ],
)
def test_extraction_boundaries(text, expected):
    assert SkillExtractor().extract(text).confirmed == expected


def test_negation_and_unknown_are_not_positive():
    result = SkillExtractor().extract(
        "No experience with Docker or SQL, but Python. Want to learn Airflow."
    )
    assert result.confirmed == {"python"}
    assert result.denied == {"docker", "sql"}
    assert result.uncertain == {"airflow"}


def test_russian_negation():
    result = SkillExtractor().extract("Нет опыта с Docker, но знаю Python. Не владею SQL.")
    assert result.confirmed == {"python"}
    assert result.denied == {"docker", "sql"}


def test_contradicting_evidence_becomes_unknown():
    result = SkillExtractor().extract("Python. No experience with Python.")
    assert "python" in result.uncertain
    assert "python" not in result.confirmed | result.denied


def test_requirement_sections_and_offsets():
    text = "Требования:\nPython и SQL\nЖелательно:\nAirflow, Docker"
    result = SkillExtractor().extract(text, vacancy=True)
    assert result.required == {"python", "sql"}
    assert result.optional == {"airflow", "docker"}
    for mention in result.mentions:
        assert text[mention.start : mention.end] == mention.text


def test_not_required():
    result = SkillExtractor().extract("Docker is not required.", vacancy=True)
    assert result.required == set()
    assert result.optional == {"docker"}


@pytest.mark.parametrize(
    ("text", "years"), [("2.5 years experience", 2.5), ("опыт 3 года", 3), ("", None)]
)
def test_experience(text, years):
    assert SkillExtractor().extract(text).experience_years == years

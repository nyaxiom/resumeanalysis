import pytest

from resumeanalysis.normalization.skill_normalizer import SkillNormalizer
from resumeanalysis.normalization.taxonomy import Taxonomy


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("Postgres", "postgresql"),
        ("GNU/Linux", "linux"),
        ("Python 3", "python"),
        ("Torch", "torch"),
        ("K8S", "kubernetes"),
        ("  SQL  ", "sql"),
    ],
)
def test_aliases(source, canonical):
    assert SkillNormalizer().normalize(source) == canonical


def test_ambiguous_torch_requires_context():
    normalizer = SkillNormalizer()
    assert normalizer.normalize("Torch", "Python tensors") == "pytorch"
    assert normalizer.normalize("Torch", "Lua") == "torch"


def test_related_skills_do_not_become_confirmed():
    norm = SkillNormalizer()
    assert norm.normalize_many(["Postgres"]) == {"postgresql"}
    assert 0 < norm.taxonomy.related_support({"postgresql"}, "sql") < 1
    assert norm.taxonomy.related_support({"sql"}, "postgresql") == 0


def test_import_official_esco_csv(tmp_path):
    path = tmp_path / "skills.csv"
    path.write_text(
        "conceptUri,preferredLabel,altLabels\nhttp://example.test/skill,Example skill,Example alias\n"
    )
    taxonomy = Taxonomy()
    old = taxonomy.identity
    taxonomy.import_esco(path)
    assert SkillNormalizer(taxonomy).normalize("Example alias") == "example skill"
    assert taxonomy.concept_uris["example skill"] == "http://example.test/skill"
    assert taxonomy.identity != old


def test_esco_ids_present():
    assert Taxonomy().concept_uris["python"].startswith("http://data.europa.eu/esco/skill/")

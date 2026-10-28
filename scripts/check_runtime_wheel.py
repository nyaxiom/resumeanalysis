"""Verify the built runtime wheel's contents and independence from legacy code."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    expected = {
        "recommendation",
        "extraction",
        "normalization",
        "embeddings",
        "profiles",
        "matching",
        "cover_letters",
        "integration",
    }
    with zipfile.ZipFile(args.wheel) as wheel, tempfile.TemporaryDirectory() as directory:
        names = wheel.namelist()
        packages = {
            p.split("/")[1] for p in names if p.startswith("resumeanalysis/") and p.count("/") > 1
        }
        assert packages == expected, packages
        assert not any(p.startswith("resumeanalysis_legacy/") for p in names)
        assert not any(p.endswith((".pt", ".safetensors", ".sqlite")) for p in names)
        assert "resumeanalysis/extraction/identity_compatibility.json" in names
        metadata = wheel.read(next(p for p in names if p.endswith(".dist-info/METADATA"))).decode()
        assert "Requires-Dist: resumeanalysis-legacy" not in metadata
        assert "Requires-Dist: pyarrow" not in metadata
        assert "Requires-Dist: lm-format-enforcer" not in metadata
        entrypoints = wheel.read(
            next(p for p in names if p.endswith(".dist-info/entry_points.txt"))
        ).decode()
        assert "resumeanalysis-qwen" not in entrypoints
        wheel.extractall(directory)
        program = """
import importlib, importlib.abc, pkgutil, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('resumeanalysis_legacy'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoLegacy())
import resumeanalysis
assert Path(resumeanalysis.__file__).is_relative_to(Path(sys.argv[1]))
for module in pkgutil.walk_packages(resumeanalysis.__path__, resumeanalysis.__name__ + '.'):
    if not module.name.endswith('.__main__'):
        importlib.import_module(module.name)
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis import RecommendationEngine, SQLiteFeedbackRepository, StructuredProfile, StructuredVacancy
repository = SQLiteFeedbackRepository(':memory:')
profile = StructuredProfile(user_id='wheel-smoke', facts={'skills':['Python']})
vacancy = StructuredVacancy(vacancy_id='fixture:1', facts={'required_skills':['Python']})
engine = RecommendationEngine(PreferenceFeatureBuilder(LexicalSmokeEncoder()), repository)
assert engine.recommend(profile.user_id, profile, [vacancy])[0].qualification.matched_skills == ['python']
engine.record_feedback(profile.user_id, vacancy.vacancy_id, 'dislike')
assert engine.recommend(profile.user_id, profile, [vacancy]) == []
repository.close()
"""
        subprocess.run([sys.executable, "-I", "-c", program, directory], check=True)
        print(
            f"Runtime wheel verified: eight product packages, {sum('.dist-info/licenses/' in p for p in names)} notices, no legacy imports"
        )


if __name__ == "__main__":
    main()

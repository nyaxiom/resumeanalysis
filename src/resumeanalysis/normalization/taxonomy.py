"""Hand-curated MVP vocabulary; related skills are NOT interchangeable.

ESCO imports retain official concept URIs; this seed vocabulary is not an ESCO export.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python", "python 3", "python3", "питон"),
    "sql": ("sql", "structured query language"),
    "postgresql": ("postgresql", "postgres", "postgre sql"),
    "mysql": ("mysql",),
    "linux": ("linux", "gnu/linux", "линукс"),
    "pytorch": ("pytorch", "py torch"),
    "docker": ("docker", "докер"),
    "kubernetes": ("kubernetes", "k8s"),
    "containers": ("containerization", "контейнеризация"),
    "airflow": ("airflow", "apache airflow"),
    "spark": ("spark", "apache spark"),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "scikit-learn": ("scikit-learn", "sklearn", "scikit learn"),
    "tensorflow": ("tensorflow",),
    "git": ("git",),
    "java": ("java",),
    "javascript": ("javascript", "java script", "js"),
    "typescript": ("typescript", "ts"),
    "c++": ("c++", "cpp"),
    "c#": ("c#", "csharp"),
    "node.js": ("node.js", "nodejs"),
    "bash": ("bash", "shell scripting"),
    "aws": ("aws", "amazon web services"),
    "azure": ("azure",),
    "gcp": ("gcp", "google cloud"),
    "excel": ("excel", "ms excel"),
    "tableau": ("tableau",),
    "power bi": ("power bi", "powerbi"),
    "machine learning": ("machine learning", "машинное обучение", "машинного обучения"),
    "deep learning": ("deep learning", "глубокое обучение"),
    "nlp": ("nlp", "natural language processing", "обработка естественного языка"),
    "etl": ("etl",),
    "statistics": ("statistics", "статистика", "статистики"),
    "communication": ("communication", "коммуникация"),
    "teamwork": ("teamwork", "team work", "командная работа"),
    "project management": ("project management", "управление проектами"),
    "data analysis": ("data analysis", "анализ данных"),
    "testing": ("unit testing", "тестирование"),
    "rest": ("rest api", "restful"),
    "fastapi": ("fastapi",),
    "django": ("django",),
    "redis": ("redis",),
    "mongodb": ("mongodb", "mongo db"),
    "ci/cd": ("ci/cd", "ci cd"),
}
RELATIONS = {
    "postgresql": {"sql": 0.6},
    "mysql": {"sql": 0.6},
    "pytorch": {"python": 0.35, "deep learning": 0.5},
    "tensorflow": {"python": 0.3, "deep learning": 0.5},
    "kubernetes": {"docker": 0.3, "containers": 0.6},
    "docker": {"containers": 0.6},
    "airflow": {"python": 0.25, "etl": 0.4},
    "pandas": {"python": 0.25, "data analysis": 0.4},
}
# ESCO v1.1.1 concept identifiers checked against Nesta's formatted ESCO table.
# European Commission, CC BY 4.0; attribution and provenance: THIRD_PARTY_NOTICES.md.
ESCO_CONCEPTS = {
    "python": "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d",
    "sql": "http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742",
    "postgresql": "http://data.europa.eu/esco/skill/a8d07b5a-c1a1-42c6-9d53-db9c7a2ca996",
    "mysql": "http://data.europa.eu/esco/skill/4da171e5-779c-4983-a76f-91c16751e99f",
}


@dataclass
class Taxonomy:
    aliases: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(ALIASES))
    relations: dict[str, dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in RELATIONS.items()}
    )
    concept_uris: dict[str, str] = field(default_factory=lambda: dict(ESCO_CONCEPTS))

    @property
    def identity(self) -> str:
        import hashlib
        import json

        content = json.dumps([self.aliases, self.relations, self.concept_uris], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def import_esco(self, skills_csv: str | Path) -> None:
        """Official skills_en.csv / skills_ru.csv: conceptUri, preferredLabel, altLabels.

        No occupation-to-skill inference. Download/license provenance belongs in data manifest.
        """
        with open(skills_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"conceptUri", "preferredLabel"}.issubset(reader.fieldnames or []):
                raise ValueError("Expected official ESCO skills CSV columns")
            for row in reader:
                label = row["preferredLabel"].strip().casefold()
                if label:
                    self.aliases[label] = tuple(
                        dict.fromkeys(
                            (
                                *self.aliases.get(label, ()),
                                label,
                                *filter(None, row.get("altLabels", "").splitlines()),
                            )
                        )
                    )
                    self.concept_uris[label] = row["conceptUri"]

    def related_support(self, possessed: set[str], target: str) -> float:
        return max((self.relations.get(s, {}).get(target, 0.0) for s in possessed), default=0)

"""Original curated vocabulary for v2, authored from the task's competency categories.

Not copied from EMSI/Lightcast, ESCO, third-party dictionaries or evaluation spans.
Related technologies are deliberately separate skills. Ambiguous short names are
recognized only through contextual guards in the v2 matcher.
"""

import re

from resumeanalysis.normalization.skill_normalizer import SkillNormalizer, clean
from resumeanalysis.normalization.taxonomy import Taxonomy

V2_ALIASES = {
    "react": ("react", "react.js", "reactjs"),
    "next.js": ("next.js", "nextjs"),
    "vue.js": ("vue.js", "vuejs", "vue"),
    "angular": ("angular", "angularjs"),
    "svelte": ("svelte",),
    "html": ("html", "html5"),
    "css": ("css", "css3", "cascading style sheets"),
    "sass": ("sass", "scss"),
    "tailwind css": ("tailwind css", "tailwind"),
    "webpack": ("webpack",),
    "vite": ("vite",),
    "redux": ("redux",),
    "express": ("express", "express.js", "expressjs"),
    "flask": ("flask",),
    "spring": ("spring", "spring boot"),
    ".net": (".net", "dotnet", "asp.net"),
    "ruby": ("ruby",),
    "ruby on rails": ("ruby on rails", "rails"),
    "php": ("php",),
    "laravel": ("laravel",),
    "golang": ("golang",),
    "rust": ("rust",),
    "scala": ("scala",),
    "kotlin": ("kotlin",),
    "swift": ("swift",),
    "objective-c": ("objective-c",),
    "c": ("c programming", "c language"),
    "r": ("r programming", "r language"),
    "perl": ("perl",),
    "matlab": ("matlab",),
    "powershell": ("powershell",),
    "graphql": ("graphql", "graph ql"),
    "rest": ("rest", "rest api", "restful api", "restful"),
    "grpc": ("grpc",),
    "api design": ("api design", "проектирование api"),
    "microservices": ("microservices", "microservice architecture", "микросервисы"),
    "sql server": ("sql server", "mssql", "ms sql server"),
    "oracle database": ("oracle database", "oracle sql"),
    "sqlite": ("sqlite",),
    "mariadb": ("mariadb",),
    "cassandra": ("cassandra", "apache cassandra"),
    "dynamodb": ("dynamodb", "dynamo db"),
    "elasticsearch": ("elasticsearch", "elastic search"),
    "opensearch": ("opensearch",),
    "clickhouse": ("clickhouse",),
    "snowflake": ("snowflake",),
    "bigquery": ("bigquery", "big query"),
    "redshift": ("redshift",),
    "databricks": ("databricks",),
    "cloud computing": ("cloud computing", "облачные вычисления"),
    "terraform": ("terraform",),
    "ansible": ("ansible",),
    "helm": ("helm",),
    "openshift": ("openshift",),
    "jenkins": ("jenkins",),
    "github actions": ("github actions",),
    "gitlab ci": ("gitlab ci", "gitlab ci/cd"),
    "prometheus": ("prometheus",),
    "grafana": ("grafana",),
    "nginx": ("nginx",),
    "apache kafka": ("apache kafka", "kafka"),
    "rabbitmq": ("rabbitmq",),
    "hadoop": ("hadoop", "apache hadoop"),
    "dbt": ("dbt",),
    "dask": ("dask",),
    "polars": ("polars",),
    "mlflow": ("mlflow",),
    "kubeflow": ("kubeflow",),
    "xgboost": ("xgboost",),
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "keras": ("keras",),
    "hugging face transformers": ("hugging face transformers", "huggingface transformers"),
    "opencv": ("opencv",),
    "computer vision": ("computer vision", "компьютерное зрение"),
    "data engineering": ("data engineering", "инженерия данных"),
    "data modeling": ("data modeling", "data modelling", "моделирование данных"),
    "data warehousing": ("data warehousing", "хранилища данных"),
    "data visualization": ("data visualization", "data visualisation", "визуализация данных"),
    "feature engineering": ("feature engineering",),
    "a/b testing": ("a/b testing", "ab testing"),
    "pytest": ("pytest",),
    "unittest": ("unittest",),
    "junit": ("junit",),
    "jest": ("jest",),
    "vitest": ("vitest",),
    "selenium": ("selenium",),
    "playwright": ("playwright",),
    "cypress": ("cypress",),
    "postman": ("postman",),
    "cucumber": ("cucumber",),
    "test automation": ("test automation", "automated testing", "автоматизация тестирования"),
    "integration testing": ("integration testing", "интеграционное тестирование"),
    "performance testing": ("performance testing", "нагрузочное тестирование"),
    "quality assurance": ("quality assurance", "обеспечение качества"),
    "tcp/ip": ("tcp/ip",),
    "network security": ("network security", "сетевая безопасность"),
    "oauth": ("oauth", "oauth2", "oauth 2.0"),
    "oauth2": ("oauth 2",),
    "agile": ("agile",),
    "scrum": ("scrum",),
    "kanban": ("kanban",),
    "jira": ("jira",),
    "confluence": ("confluence",),
    "technical writing": ("technical writing", "техническая документация"),
    "requirements analysis": ("requirements analysis", "анализ требований"),
    "stakeholder management": ("stakeholder management",),
    "customer service": ("customer service", "обслуживание клиентов"),
    "negotiation": ("negotiation", "negotiation skills", "ведение переговоров"),
    "presentation skills": ("presentation skills", "проведение презентаций"),
    "communication": (
        "communication skills",
        "written communication",
        "verbal communication",
        "навыки коммуникации",
    ),
    "teamwork": ("working in a team", "работа в команде", "умение работать в команде"),
    "leadership": ("leadership", "leadership skills", "лидерские качества"),
    "problem solving": ("problem solving", "problem-solving", "решение проблем"),
    "critical thinking": ("critical thinking", "критическое мышление"),
    "analytical thinking": ("analytical thinking", "analytical skills", "аналитическое мышление"),
    "time management": ("time management", "управление временем"),
    "attention to detail": (
        "attention to detail",
        "внимание к деталям",
        "внимательность к деталям",
    ),
    "adaptability": ("adaptability", "адаптивность"),
    "mentoring": ("mentoring", "наставничество"),
    "conflict resolution": ("conflict resolution", "разрешение конфликтов"),
}


def make_normalizer(extra: dict[str, tuple[str, ...]] | None = None) -> SkillNormalizer:
    taxonomy = Taxonomy()
    for vocabulary in (V2_ALIASES, extra or {}):
        for name, aliases in vocabulary.items():
            taxonomy.aliases[name] = tuple(
                dict.fromkeys((*taxonomy.aliases.get(name, ()), *aliases))
            )
    return SkillNormalizer(taxonomy)


def vocabulary_identity(normalizer: SkillNormalizer) -> str:
    return normalizer.taxonomy.identity


def phrase_aliases(phrases: list[str]) -> dict[str, tuple[str, ...]]:
    normalizer = make_normalizer()
    groups = {}
    for phrase in phrases:
        normalized = normalizer.normalize(re.sub(r"\s*([.\-])\s*", r"\1", clean(phrase)))
        groups[normalized] = (*groups.get(normalized, ()), phrase)
    return groups

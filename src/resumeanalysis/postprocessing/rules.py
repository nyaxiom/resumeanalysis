from resumeanalysis.features.feature_builder import FeatureDetail
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class Postprocessor:
    def __init__(self, threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")
        self.threshold = threshold

    def process(
        self, score: float, detail: FeatureDetail, profile: Profile, vacancy: Vacancy
    ) -> VacancyAnalysis:
        conflicts, unknowns = [], []
        if profile.remote_only:
            if vacancy.remote is False:
                conflicts.append("Требуется удалённая работа, вакансия её не предусматривает.")
            elif vacancy.remote is None:
                unknowns.append("Удалённый формат работы не подтверждён.")
        if profile.allowed_locations:
            if vacancy.location is None:
                unknowns.append("Местоположение вакансии не указано.")
            elif vacancy.location.casefold() not in {
                v.casefold() for v in profile.allowed_locations
            }:
                conflicts.append("Местоположение не соответствует предпочтениям.")
        parts = []
        if detail.matched:
            parts.append("Подтверждены навыки: " + ", ".join(sorted(detail.matched)) + ".")
        if detail.unknown:
            parts.append(
                "Нет подтверждения навыков (это не доказательство их отсутствия): "
                + ", ".join(sorted(detail.unknown))
                + "."
            )
        if detail.contradicted:
            parts.append(
                "Явно указано отсутствие навыков: " + ", ".join(sorted(detail.contradicted)) + "."
            )
        if detail.experience_gap is None:
            parts.append("Недостаточно данных для сравнения продолжительности опыта.")
        elif detail.experience_gap > 0:
            parts.append(f"До требования по опыту: {detail.experience_gap:g} лет.")
        else:
            parts.append("Указанное требование по продолжительности опыта выполнено.")
        if not detail.required:
            parts.append("Обязательные навыки явно не выделены.")
        parts.extend(conflicts + unknowns)
        return VacancyAnalysis(
            relevant=score >= self.threshold and not conflicts and not unknowns,
            relevance=score,
            matched_skills=sorted(detail.matched),
            missing_skills=sorted(detail.unknown | detail.contradicted),
            unknown_skills=sorted(detail.unknown),
            contradicted_skills=sorted(detail.contradicted),
            experience_gap_years=detail.experience_gap,
            explanation=" ".join(parts),
            preference_conflicts=conflicts,
            preference_unknowns=unknowns,
        )

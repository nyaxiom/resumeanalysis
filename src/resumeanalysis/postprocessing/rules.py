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
                conflicts.append("Remote work is required, but the vacancy does not offer it.")
            elif vacancy.remote is None:
                unknowns.append("Remote work has not been confirmed.")
        if profile.allowed_locations:
            if vacancy.location is None:
                unknowns.append("The vacancy location is not specified.")
            elif vacancy.location.casefold() not in {
                v.casefold() for v in profile.allowed_locations
            }:
                conflicts.append("The location does not match the allowed locations.")
        parts = []
        if detail.matched:
            parts.append("Confirmed skills: " + ", ".join(sorted(detail.matched)) + ".")
        if detail.unknown:
            parts.append(
                "Unconfirmed skills (this is not proof that the candidate lacks them): "
                + ", ".join(sorted(detail.unknown))
                + "."
            )
        if detail.contradicted:
            parts.append(
                "Explicitly denied skills: " + ", ".join(sorted(detail.contradicted)) + "."
            )
        if detail.experience_gap is None:
            parts.append("There is not enough information to compare experience duration.")
        elif detail.experience_gap > 0:
            parts.append(
                f"Experience falls short of the requirement by {detail.experience_gap:g} years."
            )
        else:
            parts.append("The stated experience-duration requirement is met.")
        if not detail.required:
            parts.append("Mandatory skills have not been explicitly identified.")
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

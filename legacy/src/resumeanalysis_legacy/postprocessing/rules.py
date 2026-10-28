from resumeanalysis.matching.evidence import SkillEvidence
from resumeanalysis.matching.rules import preference_checks
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class Postprocessor:
    def __init__(self, threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in [0, 1]")
        self.threshold = threshold

    def process(
        self, score: float, detail: SkillEvidence, profile: Profile, vacancy: Vacancy
    ) -> VacancyAnalysis:
        conflicts, unknowns = preference_checks(profile, vacancy)
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
        for group in detail.alternative_groups:
            qualifier = (
                "no option is required"
                if group["requirement"] == "negated"
                else "one option suffices"
            )
            parts.append(
                "Alternative "
                + group["requirement"]
                + " requirement ("
                + qualifier
                + "): "
                + " or ".join(group["skills"])
                + "; evidence status: "
                + group["status"]
                + "."
            )
        if not detail.required and not any(
            g["requirement"] == "required" for g in detail.alternative_groups
        ):
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

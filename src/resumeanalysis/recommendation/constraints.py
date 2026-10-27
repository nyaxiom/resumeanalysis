"""User hard constraints are independent of interest labels and qualifications."""

from resumeanalysis.postprocessing.rules import Postprocessor
from resumeanalysis.recommendation.schema import ConstraintStatus, Salary


def salary_compatibility(expected: Salary | None, offered: Salary | None) -> float | None:
    if expected is None or expected.minimum is None or offered is None:
        return None
    # No guessed exchange rates, gross/net conversion or salary-period conversion.
    for field in ("currency", "period", "gross"):
        left, right = getattr(expected, field), getattr(offered, field)
        if left is None or right is None or left != right:
            return None
    if offered.maximum is not None and offered.maximum < expected.minimum:
        return 0.0
    if offered.minimum is not None and offered.minimum >= expected.minimum:
        return 1.0
    # An overlapping range does not promise an offer above the candidate minimum.
    return None


def assess_constraints(profile, vacancy, detail) -> ConstraintStatus:
    hard = profile.preferences.hard
    legacy_profile = profile.facts.model_copy(
        update={"remote_only": hard.remote_only, "allowed_locations": hard.locations}
    )
    result = Postprocessor().process(0.5, detail, legacy_profile, vacancy.facts)
    conflicts, unknowns = list(result.preference_conflicts), list(result.preference_unknowns)
    if hard.salary is not None:
        compatible = salary_compatibility(hard.salary, vacancy.salary)
        if compatible == 0:
            conflicts.append("The stated salary range is below the required minimum.")
        elif compatible is None:
            unknowns.append("Salary compatibility needs confirmation, including units and basis.")
    return ConstraintStatus(
        status="blocked" if conflicts else "unknown" if unknowns else "eligible",
        conflicts=conflicts,
        unknowns=unknowns,
    )

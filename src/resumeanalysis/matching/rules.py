"""Shared explicit remote/location constraint checks, without a scoring backend."""

from resumeanalysis.schemas import Profile, Vacancy


def preference_checks(profile: Profile, vacancy: Vacancy) -> tuple[list[str], list[str]]:
    conflicts, unknowns = [], []
    if profile.remote_only:
        if vacancy.remote is False:
            conflicts.append("Remote work is required, but the vacancy does not offer it.")
        elif vacancy.remote is None:
            unknowns.append("Remote work has not been confirmed.")
    if profile.allowed_locations:
        if vacancy.location is None:
            unknowns.append("The vacancy location is not specified.")
        elif vacancy.location.casefold() not in {v.casefold() for v in profile.allowed_locations}:
            conflicts.append("The location does not match the allowed locations.")
    return conflicts, unknowns

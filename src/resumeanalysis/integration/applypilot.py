import asyncio
from typing import Protocol

from resumeanalysis.inference.backends import MatchingBackend, MatchingProfile
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class InferenceClient(Protocol):
    async def process(self, vacancy: Vacancy) -> VacancyAnalysis: ...


class ProfileProvider(Protocol):
    def get_profile(self) -> Profile | MatchingProfile: ...


class LocalInferenceClient:
    def __init__(self, backend: MatchingBackend, profiles: ProfileProvider):
        self.backend, self.profiles = backend, profiles

    async def process(self, vacancy: Vacancy) -> VacancyAnalysis:
        profile = self.profiles.get_profile()
        # E5's base Profile deliberately rejects preferred_roles. Do not silently
        # drop them here: applications should supply a base Profile for E5.
        return await asyncio.to_thread(self.backend.predict, profile, vacancy)

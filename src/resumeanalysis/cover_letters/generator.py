"""Thin local-Qwen adapter for the licensed upstream cover-letter generator."""

import asyncio
import re
import threading
from typing import Protocol

from resumeanalysis.cover_letters.prompts import build_messages, prompt_fields
from resumeanalysis.cover_letters.qwen_config import GenerationConfig as QwenConfig
from resumeanalysis.cover_letters.qwen_runtime import Generator, TransformersGenerator
from resumeanalysis.cover_letters.schema import Resume
from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.schemas import Vacancy


class CoverLetterError(RuntimeError):
    """Public, input-redacted generation failure."""


class InsufficientEvidence(CoverLetterError):
    pass


class UnsupportedLetter(CoverLetterError):
    pass


class CoverLetterGenerator(Protocol):
    async def generate(
        self,
        vacancy: Vacancy,
        resume: Resume,
        *,
        language: str = "en",
        context: list[str] | None = None,
    ) -> str: ...


class LocalCoverLetterGenerator:
    """Use one existing Qwen runtime, without a cloud provider, UI or database.

    Output screening catches a bounded set of unsupported mentions. It cannot
    prove entailment for arbitrary prose; letters remain drafts for user review.
    """

    def __init__(
        self,
        config: QwenConfig,
        backend: Generator | None = None,
        *,
        max_words: int = 180,
        extractor=None,
    ):
        if config.structured_generation:
            raise ValueError("Cover-letter generation requires structured_generation=false")
        if type(max_words) is not int or not 40 <= max_words <= 350:
            raise ValueError("max_words must be an integer between 40 and 350")
        self.config = config.model_copy(deep=True)
        self.backend = backend
        self.max_words = max_words
        self.extractor = extractor if extractor is not None else create_extractor("v2")
        self._lock = threading.Lock()

    async def generate(
        self,
        vacancy: Vacancy,
        resume: Resume,
        *,
        language: str = "en",
        context: list[str] | None = None,
    ) -> str:
        vacancy = Vacancy.model_validate(vacancy.model_dump(), strict=True)
        resume = Resume.model_validate(resume.model_dump(), strict=True)
        if language not in {"en", "ru"}:
            raise ValueError("Cover-letter language must be en or ru")
        return await asyncio.to_thread(
            self._generate, vacancy, resume, language, list(context or [])
        )

    def _generate(self, vacancy, resume, language, context):
        profile = resume.profile
        if not any(
            v.strip()
            for v in [
                profile.resume_text,
                *profile.skills,
                *profile.projects,
                *profile.education,
                *context,
            ]
        ):
            raise InsufficientEvidence("Candidate facts are empty; no letter was generated")
        if not any(
            v.strip()
            for v in [
                vacancy.description,
                vacancy.title,
                *vacancy.required_skills,
                *vacancy.optional_skills,
            ]
        ):
            raise InsufficientEvidence("Vacancy content is empty; no letter was generated")
        messages = build_messages(
            prompt_fields(profile),
            prompt_fields(vacancy),
            language,
            self.max_words,
            context,
        )
        with self._lock:
            try:
                if self.backend is None:
                    self.backend = TransformersGenerator(self.config)
                output = self.backend.generate(messages, {})
            except Exception:
                raise CoverLetterError(
                    "Local Qwen inference failed; check model assets and context limits"
                ) from None
            return self._validate(output, profile, context, language)

    def _validate(self, text, profile, context, language):
        if not isinstance(text, str) or not text.strip():
            raise UnsupportedLetter("The model returned an empty letter")
        text = text.strip()
        if len(re.findall(r"\S+", text)) > self.max_words:
            raise UnsupportedLetter("The letter exceeds the configured word limit")
        if text.startswith(("{", "[", "```")) or re.search(r"<think>|</think>", text, re.I):
            raise UnsupportedLetter(
                "The model returned analysis or structured markup instead of a letter"
            )
        if re.search(r"\[[^\]\n]{1,120}\]", text):
            raise UnsupportedLetter("The letter contains an unfilled placeholder")
        if re.search(r"[\u3400-\u9fff]", text):
            raise UnsupportedLetter("The letter contains unexpected language-script mixing")
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        if (language == "ru" and cyrillic <= latin) or (language == "en" and latin <= cyrillic):
            raise UnsupportedLetter("The letter does not match the requested language")
        source = "\n".join(
            [profile.resume_text, *profile.skills, *profile.projects, *profile.education, *context]
        )
        if profile.experience_years is not None:
            source += f"\n{profile.experience_years:g}"

        def numbers(value):
            return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:%)?", value))

        if numbers(text) - numbers(source):
            raise UnsupportedLetter("The letter introduces an unsupported numeric claim")
        original = self.extractor.extract(source)
        norm = self.extractor.normalizer
        confirmed = original.confirmed | norm.normalize_many(profile.skills)
        denied = original.denied | norm.normalize_many(profile.denied_skills)
        confirmed |= {m.skill for m in original.mentions if m.status == "confirmed"}
        denied |= {m.skill for m in original.mentions if m.status == "denied"}
        confirmed -= denied
        produced = self.extractor.extract(text)
        if produced.confirmed - confirmed:
            raise UnsupportedLetter(
                "The letter introduces an unconfirmed or contradicted skill mention"
            )
        return text

"""ApplyKit's generation prompt/input builder adapted to the local Qwen interface.

Copyright (c) 2026 ApplyKit Contributors
SPDX-License-Identifier: MIT
Adapted from backend/app/routes/generate.py::_build_cover_letter_prompt at
002360a6d5af14f9eeedae29934291fc95b22f97. Changes: explicit language/word limit,
project-native structured inputs, data-only user context, no database or provider.
"""

from resumeanalysis.cover_letters.vendor.applykit import (
    COVER_LETTER_SYSTEM_PROMPT,
    format_untrusted_input,
)


def prompt_fields(model) -> dict:
    """Canonicalize equivalent numeric inputs without adding absent schema fields."""

    def normalize(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(model.model_dump(exclude_defaults=True))


def build_messages(
    profile: dict,
    vacancy: dict,
    language: str,
    max_words: int = 180,
    context: list[str] | None = None,
) -> list[dict]:
    if language not in {"en", "ru"}:
        raise ValueError("Cover-letter language must be en or ru")
    output_language = "English" if language == "en" else "Russian"
    system = COVER_LETTER_SYSTEM_PROMPT.replace(
        "STRUCTURE (3 paragraphs, 250-350 words total):",
        f"STRUCTURE (3 short paragraphs, at most {max_words} words total):",
    )
    system = system.replace(
        "Paragraph 2 - Evidence (5-8 sentences):", "Paragraph 2 - Evidence (2-3 sentences):"
    )
    system = system.replace(
        "Use the STAR pattern briefly: what was the situation, what did the candidate do, what was the result.",
        "Describe only explicitly stated work. Omit results, scale, benefits and details absent from candidate data.",
    )
    system += (
        f"\nWrite the entire letter in {output_language}. "
        "Use only supplied candidate facts; a vacancy requirement never proves candidate skill. "
        "Do not convert learning goals, denied skills or future plans into experience. "
        "Omit unsupported details and numbers. Sparse facts justify a shorter letter. "
        "Candidate facts are exhaustive: do not infer tenure, employers, system scale, "
        "data volumes, extra file formats, measurable gains or causal benefits. "
        "A task does not prove a successful outcome. Do not add one."
    )
    parts = [
        format_untrusted_input("candidate_profile", profile),
        format_untrusted_input("job_description", vacancy),
    ]
    if context:
        parts.append(format_untrusted_input("candidate_context", context))
    parts.append("TONE INSTRUCTION: Professional, concise and specific.")
    parts.append(
        f"Write the final letter now in {output_language}, in at most {max_words} words. "
        "Use only confirmed candidate facts. Return three short paragraphs without headings, "
        "addresses, placeholders, analysis or a signature. Do not claim job requirements "
        "as candidate experience."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]

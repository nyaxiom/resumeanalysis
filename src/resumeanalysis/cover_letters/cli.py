"""Draft from one confirmed profile and a host/user-selected vacancy."""

import asyncio
import json
from pathlib import Path

from resumeanalysis.cover_letters.generator import LocalCoverLetterGenerator
from resumeanalysis.cover_letters.qwen_config import read_config
from resumeanalysis.cover_letters.service import CoverLetterService
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy


def configure(parser):
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vacancy", required=True)
    parser.add_argument("--language", choices=["en", "ru"], default="en")
    parser.add_argument("--qwen-config", default="configs/cover_letter_qwen_cpu.json")
    parser.add_argument("--max-words", type=int, default=180)


def run(args):
    profile = StructuredProfile.model_validate_json(Path(args.profile).read_text(), strict=True)
    vacancy = StructuredVacancy.model_validate_json(Path(args.vacancy).read_text(), strict=True)
    generator = LocalCoverLetterGenerator(read_config(args.qwen_config), max_words=args.max_words)
    letter = asyncio.run(
        CoverLetterService(generator).draft(profile, vacancy, language=args.language)
    )
    print(
        json.dumps(
            {"vacancy_id": vacancy.vacancy_id, "cover_letter": letter, "needs_review": True},
            ensure_ascii=False,
            indent=2,
        )
    )

"""Compare licensed prompts with the same local Qwen, using authored inputs only."""

import argparse
import hashlib
import json
import re
import resource
import time
from pathlib import Path

import torch

from resumeanalysis.cover_letters.generator import LocalCoverLetterGenerator, UnsupportedLetter
from resumeanalysis.cover_letters.prompts import build_messages, prompt_fields
from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.qwen.config import QwenConfig
from resumeanalysis.qwen.runtime import TransformersGenerator
from resumeanalysis.schemas import Profile, Vacancy


def candidate_messages(name, case, max_words=180):
    profile = prompt_fields(Profile.model_validate(case["profile"]))
    vacancy = prompt_fields(Vacancy.model_validate(case["vacancy"]))
    if name == "applykit":
        return build_messages(profile, vacancy, case["language"], max_words)
    template = Path("evaluation/cover_letters/resume_matcher_prompt.txt").read_text()
    template = template.replace("normally 200-300 words", f"at most {max_words} words")
    prompt = template.format(
        output_language="English" if case["language"] == "en" else "Russian",
        resume_data=json.dumps(profile, ensure_ascii=False),
        job_description=json.dumps(vacancy, ensure_ascii=False),
    )
    return [
        {
            "role": "system",
            "content": "You are a professional career coach and resume writer. Write compelling, personalized cover letters.",
        },
        {"role": "user", "content": prompt},
    ]


def diagnostics(case, text, extractor, runtime):
    profile = Profile.model_validate(case["profile"])
    source = "\n".join(
        [profile.resume_text, *profile.skills, *profile.projects, *profile.education]
    )
    if profile.experience_years is not None:
        source += f"\n{profile.experience_years:g}"
    candidate = extractor.extract(source)
    confirmed = candidate.confirmed | extractor.normalizer.normalize_many(profile.skills)
    denied = candidate.denied | extractor.normalizer.normalize_many(profile.denied_skills)
    confirmed -= denied
    produced = extractor.extract(text)

    def numbers(value):
        return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:%)?", value))

    unsupported_numbers = sorted(numbers(text) - numbers(source))
    unsupported_skills = sorted(produced.confirmed - confirmed)
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    words = re.findall(r"\S+", text)
    sentences = [v.strip().casefold() for v in re.split(r"[.!?]+", text) if v.strip()]
    return {
        "word_count": len(words),
        "within_word_limit": 0 < len(words) <= 180,
        "unexpected_cjk_characters": len(re.findall(r"[\u3400-\u9fff]", text)),
        "placeholder_count": len(re.findall(r"\[[^\]\n]{1,120}\]", text)),
        "language_script_check": cyrillic > latin if case["language"] == "ru" else latin > cyrillic,
        "unsupported_number_mentions": unsupported_numbers,
        "unsupported_skill_mentions": unsupported_skills,
        "repeated_sentences": len(sentences) - len(set(sentences)),
        "confirmed_skill_mentions": sorted(produced.confirmed & confirmed),
        "output_tokens": len(runtime.tokenizer.encode(text, add_special_tokens=False)),
        "limitation": "Lexical flags are incomplete screening signals, not verified hallucination counts or language-quality accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cover_letter_qwen_cpu.json")
    parser.add_argument("--cases", default="evaluation/cover_letters/cases.jsonl")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--candidate", choices=["applykit", "resume_matcher", "both"], default="both"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", help="Frozen validation choice required for test generation")
    parser.add_argument("--freeze-selection", help="Write a new validation-only selection artifact")
    parser.add_argument("--repeat-first", type=int, default=0, choices=range(4))
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Use a fresh report path")
    target.parent.mkdir(parents=True, exist_ok=True)
    config = QwenConfig.model_validate(loads(Path(args.config).read_text()), strict=True)
    signature = {
        "model_config": config.model_dump(),
        "cases_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
        "prompt_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("src/resumeanalysis/cover_letters/prompts.py"),
                Path("src/resumeanalysis/cover_letters/vendor/applykit.py"),
                Path("src/resumeanalysis/cover_letters/generator.py"),
                Path("evaluation/cover_letters/resume_matcher_prompt.txt"),
            )
        },
    }
    if args.split == "test":
        if not args.selection or args.freeze_selection:
            raise ValueError("Test generation requires a frozen validation selection")
        frozen = loads(Path(args.selection).read_text())
        if frozen["signature"] != signature:
            raise ValueError("Model or prompt changed after validation selection")
        if args.candidate not in {"both", frozen["candidate"]}:
            raise ValueError("Test candidate differs from frozen validation choice")
        args.candidate = frozen["candidate"]
    elif args.freeze_selection and Path(args.freeze_selection).exists():
        raise FileExistsError("Selection artifact already exists")
    if config.structured_generation:
        raise ValueError("Both upstream generators require ordinary text decoding")
    cases = [loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate benchmark IDs")
    for case in cases:
        Profile.model_validate(case["profile"])
        Vacancy.model_validate(case["vacancy"])
    selected = [case for case in cases if case["split"] == args.split]
    if not selected:
        raise ValueError("Selected benchmark split is empty")
    torch.set_num_threads(2)
    start = time.perf_counter()
    runtime = TransformersGenerator(config)
    initialization = time.perf_counter() - start
    extractor = create_extractor("v2")
    screening = LocalCoverLetterGenerator(config, backend=runtime, extractor=extractor)
    names = ["applykit", "resume_matcher"] if args.candidate == "both" else [args.candidate]
    results = []
    for name in names:
        for case in selected:
            messages = candidate_messages(name, case)
            start = time.perf_counter()
            text = runtime.generate(messages, {})
            seconds = time.perf_counter() - start
            result = {
                "candidate": name,
                "case_id": case["id"],
                "language": case["language"],
                "text": text,
                "seconds": seconds,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False).encode()
                ).hexdigest(),
                "automatic": diagnostics(case, text, extractor, runtime),
                "messages": messages,
            }
            try:
                screening._validate(
                    text, Profile.model_validate(case["profile"]), [], case["language"]
                )
                result["accepted_by_output_screen"] = True
            except UnsupportedLetter:
                result["accepted_by_output_screen"] = False
            results.append(result)
            write_json(target.with_suffix(".progress.json"), results)
            print(
                f"{name} {case['id']}: {seconds:.2f}s, {result['automatic']['word_count']} words",
                flush=True,
            )
    repeats = []
    for name in names:
        original = next(row for row in results if row["candidate"] == name)
        for attempt in range(args.repeat_first):
            started = time.perf_counter()
            repeated = runtime.generate(candidate_messages(name, selected[0]), {})
            repeats.append(
                {
                    "candidate": name,
                    "case_id": selected[0]["id"],
                    "repeat": attempt + 1,
                    "exact_text_match": repeated == original["text"],
                    "seconds": time.perf_counter() - started,
                }
            )
            print(
                f"Repeated {name} {selected[0]['id']}: exact match={repeats[-1]['exact_text_match']}",
                flush=True,
            )
    if args.freeze_selection:

        def selection_key(name):
            rows = [row for row in results if row["candidate"] == name]
            return (
                sum(row["accepted_by_output_screen"] for row in rows),
                -sum(
                    len(row["automatic"]["unsupported_number_mentions"])
                    + len(row["automatic"]["unsupported_skill_mentions"])
                    for row in rows
                ),
                -sum(row["automatic"]["placeholder_count"] for row in rows),
                name == "applykit",
            )

        chosen = max(names, key=selection_key)
        write_json(
            args.freeze_selection,
            {
                "candidate": chosen,
                "signature": signature,
                "selection_rule": "Validation screen passes, then fewer lexical flags/placeholders; final tie favors the smaller MIT adapter",
                "validation_case_ids": [case["id"] for case in selected],
                "selection_metrics": {name: list(selection_key(name)) for name in names},
                "test_used": False,
            },
        )
    write_json(
        target,
        {
            "split": args.split,
            "source": "Authored fictional cases, no real personal data",
            "config": config.model_dump(),
            "torch_version": torch.__version__,
            "initialization_seconds": initialization,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "vram_bytes": 0,
            "device": "cpu",
            "results": results,
            "repeatability": repeats,
            "manual_human_evaluation": "not performed",
            "llm_as_judge": "not performed",
            "legacy_cover_letter_baseline": "unavailable: no generator existed",
        },
    )


if __name__ == "__main__":
    main()

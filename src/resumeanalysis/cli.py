"""Runtime commands for recommendations, profiles and selected-vacancy letters."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main(argv=None):
    from resumeanalysis.cover_letters.cli import configure as configure_letters
    from resumeanalysis.recommendation.cli import configure as configure_recommendations

    parser = argparse.ArgumentParser(description="Local River job recommendations")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    configure_recommendations(sub)
    configure_letters(sub.add_parser("cover-letter", help="Draft a letter for a selected vacancy"))
    download = sub.add_parser("download-e5", help="Explicitly download the pinned E5 model")
    download.add_argument("--output", default="artifacts/e5")
    download = sub.add_parser("download-qwen", help="Explicitly download configured Qwen weights")
    download.add_argument("--config", default="configs/cover_letter_qwen_cpu.json")
    extract = sub.add_parser("extract", help="Extract skill evidence from a UTF-8 text file")
    extract.add_argument("--input", required=True)
    extract.add_argument("--vacancy", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    try:
        if args.command in {"recommend", "feedback", "preferences", "profile"}:
            from resumeanalysis.recommendation.cli import run

            run(args)
        elif args.command == "cover-letter":
            from resumeanalysis.cover_letters.cli import run

            run(args)
        elif args.command == "download-e5":
            from resumeanalysis.embeddings.assets import download_e5

            download_e5(args.output)
        elif args.command == "download-qwen":
            from resumeanalysis.cover_letters.qwen_assets import download
            from resumeanalysis.cover_letters.qwen_config import read_config

            print(json.dumps(download(read_config(args.config)), indent=2))
        elif args.command == "extract":
            from resumeanalysis.extraction.registry import create_extractor

            result = create_extractor("v2").extract(
                Path(args.input).read_text(encoding="utf-8"), vacancy=args.vacancy
            )
            print(json.dumps(asdict(result), default=sorted, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError):
        parser.error("Request failed; check input schemas, versions, local files and configuration")

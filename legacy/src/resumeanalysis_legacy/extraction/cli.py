"""Standalone and application-embedded CLI for reproducible extractor diagnostics."""

import argparse
import json
import time
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis_legacy.experiments.extractor_evaluation import (
    benchmark_skillspan,
    evaluate_fixtures,
    evaluate_role_radar,
    evaluate_skillspan,
    extraction_details,
    save_comparison,
)


def configure_parser(parser):
    sub = parser.add_subparsers(dest="extractor_command", required=True)
    p = sub.add_parser("build-lexicon")
    p.add_argument("--input", required=True, help="Exact pinned SkillSpan train.json")
    p.add_argument("--output", required=True)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=12)
    for name in ("extract", "skillspan", "role-radar", "fixtures", "_skillspan-worker"):
        p = sub.add_parser(name)
        p.add_argument("--extractor", choices=["legacy", "v2"], default="v2")
        p.add_argument("--lexicon-path")
        if name == "extract":
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument("--text")
            source.add_argument("--input", help="UTF-8 plain-text file")
            p.add_argument("--vacancy", action="store_true")
            p.add_argument("--output")
        else:
            p.add_argument("--input", required=True)
            p.add_argument("--output", required=True)
        if name in ("skillspan", "_skillspan-worker"):
            p.add_argument("--include-records", action="store_true")
        if name == "role-radar":
            p.add_argument(
                "--checkpoint", help="Optional MLP checkpoint matching the selected extractor"
            )
            p.add_argument("--e5-path", default="artifacts/e5")
        if name == "_skillspan-worker":
            p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("compare")
    p.add_argument("--input", required=True, help="SkillSpan-compatible BIO JSONL")
    p.add_argument("--extractors", nargs="+", choices=["legacy", "v2"], default=["legacy", "v2"])
    p.add_argument("--lexicon-path", help="Optional train-only phrase lexicon for v2")
    p.add_argument("--output", default="artifacts/experiments/skill_extractor/skillspan.json")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--include-records", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--input", required=True, help="Existing comparison JSON")
    p.add_argument("--output", required=True, help="HTML report path")
    p = sub.add_parser("ablate")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--legacy-checkpoint", default="artifacts/run/model.pt")
    p.add_argument("--v2-checkpoint", required=True)
    p.add_argument("--e5-path", default="artifacts/e5")
    p.add_argument("--lexicon-path")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    p = sub.add_parser("split-audit")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    return parser


def run(args, parser):
    from resumeanalysis.extraction.registry import create_extractor, extractor_identity

    command = args.extractor_command
    if getattr(args, "input", None) and getattr(args, "output", None):
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("Output must not overwrite input data")
    if command == "split-audit":
        from resumeanalysis_legacy.experiments.skillspan_splits import audit_skillspan_splits

        result = audit_skillspan_splits(args.train, args.validation, args.test, args.output_dir)
    elif command == "ablate":
        from resumeanalysis.embeddings.encoder import TextEncoder
        from resumeanalysis_legacy.experiments.extractor_ablation import run_ablation

        result = run_ablation(
            args.data,
            args.legacy_checkpoint,
            args.v2_checkpoint,
            TextEncoder(args.e5_path),
            args.output,
            lexicon_path=args.lexicon_path,
            limit=args.limit,
        )
    elif command == "build-lexicon":
        from resumeanalysis_legacy.extraction.phrase_lexicon import build_lexicon

        metadata = Path(args.output).with_suffix(".metadata.json")
        if metadata.resolve() == Path(args.output).resolve():
            parser.error("Lexicon filename must differ from its .metadata.json sidecar")
        if metadata.exists():
            raise FileExistsError("Use a new lexicon metadata artifact path")
        result = build_lexicon(
            args.input, args.output, min_frequency=args.min_frequency, max_tokens=args.max_tokens
        )
        write_json(metadata, result)
    elif command == "compare":
        if args.threads < 1:
            parser.error("threads must be positive")
        result = benchmark_skillspan(
            args.input,
            args.extractors,
            lexicon_path=args.lexicon_path,
            threads=args.threads,
            include_records=args.include_records,
        )
        save_comparison(result, args.output)
    elif command == "report":
        from resumeanalysis_legacy.experiments.extractor_evaluation import comparison_html

        result = loads(Path(args.input).read_text())
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(comparison_html(result))
    else:
        if args.lexicon_path and args.extractor != "v2":
            parser.error("A phrase lexicon is only supported by v2")
        if command == "_skillspan-worker":
            import torch

            if args.threads < 1:
                parser.error("threads must be positive")
            torch.set_num_threads(args.threads)
        start = time.perf_counter()
        extractor = create_extractor(args.extractor, lexicon_path=args.lexicon_path)
        initialization = time.perf_counter() - start
        if command == "extract":
            text = Path(args.input).read_text() if args.input else args.text
            result = extraction_details(extractor.extract(text, vacancy=args.vacancy), text)
        elif command in ("skillspan", "_skillspan-worker"):
            result = evaluate_skillspan(args.input, extractor, include_records=args.include_records)
            result["initialization_seconds"] = initialization
        elif command == "role-radar":
            backend = None
            if args.checkpoint:
                from resumeanalysis.embeddings.encoder import TextEncoder
                from resumeanalysis_legacy.inference.predictor import Predictor

                backend = Predictor(args.checkpoint, TextEncoder(args.e5_path), extractor=extractor)
            result = evaluate_role_radar(args.input, extractor, backend=backend)
            if backend is not None:
                from resumeanalysis_legacy.data.download import sha256_file

                result["model"] = {
                    "checkpoint_sha256": sha256_file(args.checkpoint),
                    "encoder_identity": backend.builder.encoder.identity,
                    "extractor_identity": backend.builder.extractor_identity,
                    "pipeline_version": backend.builder.pipeline_version,
                }
        else:
            result = evaluate_fixtures(args.input, extractor)
        result["extractor_identity"] = extractor_identity(extractor)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    if command != "_skillspan-worker":
        display = result
        if command == "ablate":
            display = {
                "output": args.output,
                "conditions": {
                    name: {k: v for k, v in condition.items() if k != "records"}
                    for name, condition in result["conditions"].items()
                },
                "claim": result["claim"],
            }
        print(json.dumps(display, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    parser = configure_parser(argparse.ArgumentParser(description="Skill extraction experiments"))
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()

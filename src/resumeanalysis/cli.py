import argparse
import json
import math
from pathlib import Path

import torch

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder, TextEncoder


def _run_rerank(args, parser):
    """Validate new backend inputs before loading any local model assets."""
    from resumeanalysis.extraction.registry import create_extractor
    from resumeanalysis.inference.backends import MatchingProfile, create_predictor
    from resumeanalysis.jsonio import loads
    from resumeanalysis.rerank.config import RerankConfig
    from resumeanalysis.schemas import Vacancy

    if not math.isfinite(args.decision_logit_threshold):
        parser.error("--decision-logit-threshold must be finite")
    if args.command == "rank" and (args.preselect_k < 1 or args.top_k < 1):
        parser.error("--preselect-k and --top-k must be positive")
    try:
        config = RerankConfig.model_validate(
            loads(Path(args.rerank_config).read_text()), strict=True
        )
    except (OSError, ValueError, TypeError):
        parser.error("Cannot read a valid reranker configuration; inspect --rerank-config")
    try:
        document = loads(Path(args.input).read_text())
        expected = {"profile", "vacancies" if args.command == "rank" else "vacancy"}
        if not isinstance(document, dict) or set(document) != expected:
            raise ValueError("Unexpected input fields")
        profile = MatchingProfile.model_validate(document["profile"], strict=True)
        if args.command == "rank":
            if not isinstance(document["vacancies"], list):
                raise ValueError("Vacancies must be an array")
            vacancies = [Vacancy.model_validate(v, strict=True) for v in document["vacancies"]]
        else:
            vacancy = Vacancy.model_validate(document["vacancy"], strict=True)
    except (OSError, ValueError, TypeError):
        parser.error(
            "Invalid reranker input; expected profile and "
            + ("a vacancies array" if args.command == "rank" else "one vacancy object")
        )
    extractor = create_extractor(args.extractor, lexicon_path=args.lexicon_path)
    if args.command == "rank" and not vacancies:
        return []
    encoder = LexicalSmokeEncoder() if args.smoke_lexical else TextEncoder(args.e5_path)
    predictor = create_predictor(
        "rerank",
        encoder=encoder,
        config=config,
        extractor=extractor,
        decision_logit_threshold=args.decision_logit_threshold,
    )
    if args.command == "rank":
        return [
            item.model_dump()
            for item in predictor.rank(
                profile, vacancies, preselect_k=args.preselect_k, top_k=args.top_k
            )
        ]
    return predictor.predict(profile, vacancy).model_dump()


def main():
    from resumeanalysis.data.adapter_cli import configure_parser
    from resumeanalysis.data.adapter_cli import run as datasets_run
    from resumeanalysis.extraction.cli import configure_parser as configure_extractors
    from resumeanalysis.extraction.cli import run as extractors_run

    parser = argparse.ArgumentParser(description="ResumeAnalysis: local matching and CPU training")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    configure_parser(sub.add_parser("datasets", help="Inspect, convert and audit JSONL datasets"))
    configure_extractors(
        sub.add_parser("extractors", help="Compare versioned local skill extractors")
    )
    from resumeanalysis.cover_letters.cli import configure as configure_letters

    configure_letters(
        sub.add_parser("analyze", help="Score resumes and draft a local cover letter")
    )
    for name, default in [
        ("download-e5", "artifacts/e5"),
        ("download-data", "data/raw/candidate-matching-synthetic"),
        ("download-reranker", "artifacts/reranker"),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--output", default=default)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", default="data/raw/candidate-matching-synthetic")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--max-jobs", type=int, default=250)
    p.add_argument("--pairs-per-job", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare-med2425")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--license-evidence", required=True)
    p.add_argument(
        "--potential-fit", choices=["exclude", "positive", "negative"], default="exclude"
    )
    for name in ["train", "evaluate", "predict"]:
        p = sub.add_parser(name)
        p.add_argument("--extractor", choices=["legacy", "v2"], default="legacy")
        p.add_argument("--lexicon-path", help="Optional pinned train-only v2 phrase artifact")
        p.add_argument("--e5-path", default="artifacts/e5")
        p.add_argument(
            "--smoke-lexical",
            action="store_true",
            help="Diagnostic hashing encoder, explicitly not E5",
        )
        if name == "predict":
            p.add_argument("--backend", choices=["mlp", "qwen", "rerank"], default="mlp")
            p.add_argument("--qwen-config", default="configs/qwen.json")
            p.add_argument("--rerank-config", default="configs/rerank.json")
            p.add_argument("--decision-logit-threshold", type=float, default=0.0)
            p.add_argument("--checkpoint", default="artifacts/run/model.pt")
            p.add_argument("--input", required=True)
        else:
            p.add_argument("--data", default="data/processed")
            p.add_argument("--output", default="artifacts/run")
        if name == "train":
            p.add_argument("--config", default="configs/cpu.json")
    p = sub.add_parser("rank", help="Opt-in exact E5 retrieval and offline cross-encoder ranking")
    p.add_argument("--input", required=True, help="JSON object with profile and vacancies array")
    p.add_argument("--preselect-k", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--e5-path", default="artifacts/e5")
    p.add_argument("--rerank-config", default="configs/rerank.json")
    p.add_argument("--decision-logit-threshold", type=float, default=0.0)
    p.add_argument("--extractor", choices=["legacy", "v2"], default="v2")
    p.add_argument("--lexicon-path", help="Optional pinned train-only v2 phrase artifact")
    p.add_argument(
        "--smoke-lexical", action="store_true", help="Explicit hashing diagnostic, not E5"
    )
    p = sub.add_parser("skillspan")
    p.add_argument("--input", required=True, help="Upstream SkillSpan test.json (JSONL)")
    p.add_argument("--output", default="artifacts/skillspan_metrics.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "analyze":
        from resumeanalysis.cover_letters.cli import run as run_letters

        run_letters(args)
        return
    if args.command == "datasets":
        return datasets_run(args, parser)
    if args.command == "extractors":
        extractors_run(args, parser)
        return
    if args.command == "download-e5":
        from resumeanalysis.data.download import download_e5

        download_e5(args.output)
        return
    if args.command == "download-data":
        from resumeanalysis.data.download import download_candidates

        download_candidates(args.output)
        return
    if args.command == "download-reranker":
        from resumeanalysis.rerank.assets import download_reranker

        download_reranker(args.output)
        return
    if args.command == "prepare":
        from resumeanalysis.data.prepare import prepare_candidates

        result = prepare_candidates(
            args.raw, args.output, args.max_jobs, args.pairs_per_job, args.seed
        )
    elif args.command == "prepare-med2425":
        from resumeanalysis.data.prepare import prepare_med2425

        result = prepare_med2425(
            args.input,
            args.output,
            license_evidence=args.license_evidence,
            potential_fit=args.potential_fit,
        )
    elif args.command == "skillspan":
        from resumeanalysis.data.skillspan import evaluate_skillspan

        result = evaluate_skillspan(args.input)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "rank" or (args.command == "predict" and args.backend == "rerank"):
        result = _run_rerank(args, parser)
    elif args.command == "predict" and args.backend == "qwen":
        from resumeanalysis.extraction.registry import create_extractor
        from resumeanalysis.jsonio import loads
        from resumeanalysis.qwen.cli import read_config
        from resumeanalysis.qwen.predictor import QwenPredictor

        pair = loads(Path(args.input).read_text())
        result = (
            QwenPredictor(
                read_config(args.qwen_config),
                extractor=create_extractor(args.extractor, lexicon_path=args.lexicon_path),
            )
            .predict(pair["profile"], pair["vacancy"])
            .model_dump()
        )
    else:
        from resumeanalysis.extraction.registry import create_extractor

        extractor = create_extractor(args.extractor, lexicon_path=args.lexicon_path)
        encoder = LexicalSmokeEncoder() if args.smoke_lexical else TextEncoder(args.e5_path)
        if args.command == "train":
            from resumeanalysis.training.train import TrainingConfig, train

            config = TrainingConfig(**json.loads(Path(args.config).read_text()))
            if (
                args.extractor != "legacy"
                and Path(args.output).resolve() == Path("artifacts/run").resolve()
            ):
                parser.error(
                    "V2 training requires a separate --output, such as artifacts/skill-extractor-v2/run"
                )
            result = train(args.data, args.output, encoder, config, extractor=extractor)
            result = {k: v for k, v in result.items() if k != "history"}
        elif args.command == "evaluate":
            from resumeanalysis.training.train import evaluate_saved

            result = evaluate_saved(args.data, args.output, encoder, extractor=extractor)
        else:
            from resumeanalysis.inference.predictor import Predictor

            pair = json.loads(Path(args.input).read_text())
            result = (
                Predictor(args.checkpoint, encoder, extractor=extractor)
                .predict(pair["profile"], pair["vacancy"])
                .model_dump()
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))

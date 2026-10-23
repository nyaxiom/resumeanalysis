import argparse
import json
from pathlib import Path

import torch

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder, TextEncoder


def main():
    from resumeanalysis.data.adapter_cli import configure_parser
    from resumeanalysis.data.adapter_cli import run as datasets_run

    parser = argparse.ArgumentParser(description="ResumeAnalysis: local matching and CPU training")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    configure_parser(sub.add_parser("datasets", help="Inspect, convert and audit JSONL datasets"))
    for name, default in [
        ("download-e5", "artifacts/e5"),
        ("download-data", "data/raw/candidate-matching-synthetic"),
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
        p.add_argument("--e5-path", default="artifacts/e5")
        p.add_argument(
            "--smoke-lexical",
            action="store_true",
            help="Diagnostic hashing encoder, explicitly not E5",
        )
        if name == "predict":
            p.add_argument("--backend", choices=["mlp", "qwen"], default="mlp")
            p.add_argument("--qwen-config", default="configs/qwen.json")
            p.add_argument("--checkpoint", default="artifacts/run/model.pt")
            p.add_argument("--input", required=True)
        else:
            p.add_argument("--data", default="data/processed")
            p.add_argument("--output", default="artifacts/run")
        if name == "train":
            p.add_argument("--config", default="configs/cpu.json")
    p = sub.add_parser("skillspan")
    p.add_argument("--input", required=True, help="Upstream SkillSpan test.json (JSONL)")
    p.add_argument("--output", default="artifacts/skillspan_metrics.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "datasets":
        return datasets_run(args, parser)
    if args.command == "download-e5":
        from resumeanalysis.data.download import download_e5

        download_e5(args.output)
        return
    if args.command == "download-data":
        from resumeanalysis.data.download import download_candidates

        download_candidates(args.output)
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
    elif args.command == "predict" and args.backend == "qwen":
        from resumeanalysis.jsonio import loads
        from resumeanalysis.qwen.cli import read_config
        from resumeanalysis.qwen.predictor import QwenPredictor

        pair = loads(Path(args.input).read_text())
        result = (
            QwenPredictor(read_config(args.qwen_config))
            .predict(pair["profile"], pair["vacancy"])
            .model_dump()
        )
    else:
        encoder = LexicalSmokeEncoder() if args.smoke_lexical else TextEncoder(args.e5_path)
        if args.command == "train":
            from resumeanalysis.training.train import TrainingConfig, train

            config = TrainingConfig(**json.loads(Path(args.config).read_text()))
            result = train(args.data, args.output, encoder, config)
            result = {k: v for k, v in result.items() if k != "history"}
        elif args.command == "evaluate":
            from resumeanalysis.training.train import evaluate_saved

            result = evaluate_saved(args.data, args.output, encoder)
        else:
            from resumeanalysis.inference.predictor import Predictor

            pair = json.loads(Path(args.input).read_text())
            result = (
                Predictor(args.checkpoint, encoder)
                .predict(pair["profile"], pair["vacancy"])
                .model_dump()
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))

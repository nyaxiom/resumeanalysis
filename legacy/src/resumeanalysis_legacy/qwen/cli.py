"""Qwen-specific utilities and isolated-process model comparisons."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis_legacy.qwen.config import QwenConfig


def read_config(path=None):
    return (
        QwenConfig.model_validate(loads(Path(path).read_text()), strict=True)
        if path
        else QwenConfig()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optional local Qwen backend and controlled experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="legacy/configs/qwen.json")
        if name == "predict":
            p.add_argument("--input", required=True)
            p.add_argument("--extractor", choices=["legacy", "v2"], default="legacy")
            p.add_argument("--lexicon-path")
    p = sub.add_parser("augment")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--output", required=True)
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    p.add_argument("--seed", type=int, default=42)
    for name in ("measure", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data/processed")
        p.add_argument("--split", choices=["validation", "test"], default="test")
        p.add_argument("--qwen-config", default="legacy/configs/qwen.json")
        p.add_argument("--checkpoint", default="artifacts/run/model.pt")
        p.add_argument("--e5-path", default="artifacts/e5")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--warmup", type=int, default=1)
        p.add_argument("--limit", type=int, help="Diagnostic ordered subset, not a full benchmark")
        p.add_argument("--output", required=True)
        if name == "measure":
            p.add_argument("--backend", choices=["mlp", "qwen"], required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        from resumeanalysis.cover_letters.qwen_assets import download

        result = download(read_config(args.config))
    elif args.command == "predict":
        from resumeanalysis.extraction.registry import create_extractor
        from resumeanalysis_legacy.qwen.predictor import QwenPredictor

        pair = loads(Path(args.input).read_text())
        result = (
            QwenPredictor(
                read_config(args.config),
                extractor=create_extractor(args.extractor, lexicon_path=args.lexicon_path),
            )
            .predict(pair["profile"], pair["vacancy"])
            .model_dump()
        )
    elif args.command == "augment":
        from resumeanalysis_legacy.experiments.partial_profiles import augment_dataset

        result = augment_dataset(args.data, args.output, args.ratios, args.seed)
    elif args.command == "measure":
        import torch

        from resumeanalysis_legacy.data.download import sha256_file
        from resumeanalysis_legacy.experiments.comparison import (
            comparison_manifest,
            evaluate_backend,
        )
        from resumeanalysis_legacy.inference.backends import create_predictor

        if args.threads < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
            parser.error("threads/limit must be positive; warmup must be nonnegative")
        torch.set_num_threads(args.threads)
        records, metadata = comparison_manifest(args.data, args.split)
        records = records[: args.limit]
        start = time.perf_counter()
        if args.backend == "mlp":
            from resumeanalysis.embeddings.encoder import TextEncoder

            backend = create_predictor(
                "mlp", checkpoint=args.checkpoint, encoder=TextEncoder(args.e5_path)
            )
            identity = {
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "encoder_identity": backend.builder.encoder.identity,
            }
        else:
            config = read_config(args.qwen_config)
            backend = create_predictor("qwen", config=config)
            identity = config.model_dump()
        initialization = time.perf_counter() - start
        result = evaluate_backend(
            backend, records, split=args.split, seed=args.seed, warmup=args.warmup
        )
        result.update(
            {
                "backend": args.backend,
                "identity": identity,
                "initialization_seconds": initialization,
                "dataset": metadata,
                "limit": args.limit,
            }
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    else:
        from resumeanalysis_legacy.experiments.comparison import combine_reports

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        reports = {}
        for name in ("mlp", "qwen"):
            report_path = output.with_name(output.stem + "." + name + ".json")
            if report_path.exists() or output.exists():
                raise FileExistsError("Use fresh comparison output paths")
            cmd = [
                sys.executable,
                "-m",
                "resumeanalysis_legacy.qwen.cli",
                "measure",
                "--backend",
                name,
                "--data",
                args.data,
                "--split",
                args.split,
                "--qwen-config",
                args.qwen_config,
                "--checkpoint",
                args.checkpoint,
                "--e5-path",
                args.e5_path,
                "--seed",
                str(args.seed),
                "--threads",
                str(args.threads),
                "--warmup",
                str(args.warmup),
                "--output",
                str(report_path),
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            reports[name] = loads(report_path.read_text())
        result = combine_reports(reports)
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

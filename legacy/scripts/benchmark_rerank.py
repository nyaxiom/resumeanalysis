"""Run five CPU methods in separate processes and combine identical-record evidence."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from resumeanalysis_legacy.experiments.rerank_evaluation import (
    METHODS,
    benchmark_summary,
    benchmark_worker,
    combine_benchmarks,
)

from resumeanalysis.jsonio import loads, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="artifacts/rerank-benchmark")
    parser.add_argument("--e5-path", default="artifacts/e5")
    parser.add_argument("--model-path", default="artifacts/reranker")
    parser.add_argument("--checkpoint", default="artifacts/run/model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.threads) < 1:
        parser.error("Batch, length and threads must be positive")
    if args.worker:
        report = benchmark_worker(
            args.worker,
            data=args.data,
            output=args.output,
            e5_path=args.e5_path,
            model_path=args.model_path,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            max_length=args.max_length,
            threads=args.threads,
        )
        print(f"{args.worker}: F1={report['test_pair_metrics']['macro_f1']:.4f}", flush=True)
        return
    output = Path(args.output)
    if output.exists():
        parser.error("Use a fresh output directory; previous measurements are preserved")
    output.mkdir(parents=True)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for method in METHODS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            method,
            "--data",
            args.data,
            "--output",
            str(output / method),
            "--e5-path",
            args.e5_path,
            "--model-path",
            args.model_path,
            "--checkpoint",
            args.checkpoint,
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--threads",
            str(args.threads),
        ]
        with (output / (method + ".log")).open("x") as log:
            subprocess.run(
                command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        print(f"Completed {method}", flush=True)
    reports = {method: loads((output / method / "report.json").read_text()) for method in METHODS}
    comparison = combine_benchmarks(args.data, reports)
    write_json(output / "comparison.json", comparison)
    write_json(output / "summary.json", benchmark_summary(comparison))
    print(f"Wrote {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()

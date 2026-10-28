"""Replay the exact saved messages/configuration from a published local comparison.

This reproduces historical prompt versions without changing production prompts
or selecting a model. Only authored benchmark data belongs in public reports.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from resumeanalysis_legacy.qwen.config import QwenConfig

from resumeanalysis.cover_letters.qwen_runtime import TransformersGenerator
from resumeanalysis.jsonio import loads, write_json


def read_report(path):
    report = loads(Path(path).read_text())
    config = QwenConfig.model_validate(report["config"], strict=True)
    if config.structured_generation:
        raise ValueError("A plain-text generation report is required")
    for row in report["results"]:
        digest = hashlib.sha256(json.dumps(row["messages"], ensure_ascii=False).encode())
        if digest.hexdigest() != row["prompt_sha256"]:
            raise ValueError("Saved messages do not match the recorded prompt hash")
    return report, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Use a fresh report path")
    report, config = read_report(args.report)
    torch.set_num_threads(2)
    started = time.perf_counter()
    runtime = TransformersGenerator(config)
    initialization = time.perf_counter() - started
    rows = []
    for original in report["results"]:
        started = time.perf_counter()
        text = runtime.generate(original["messages"], {})
        rows.append(
            {
                "candidate": original["candidate"],
                "case_id": original["case_id"],
                "prompt_sha256": original["prompt_sha256"],
                "text": text,
                "seconds": time.perf_counter() - started,
                "exact_text_match": text == original["text"],
            }
        )
        write_json(target.with_suffix(".progress.json"), rows)
    write_json(
        target,
        {
            "source_report": args.report,
            "source_report_sha256": hashlib.sha256(Path(args.report).read_bytes()).hexdigest(),
            "config": config.model_dump(),
            "torch_version": torch.__version__,
            "initialization_seconds": initialization,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "results": rows,
            "purpose": "Reproduction only; no new selection or independent quality measurement",
        },
    )


if __name__ == "__main__":
    main()

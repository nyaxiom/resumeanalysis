import argparse
import json
import warnings
from pathlib import Path

from resumeanalysis.jsonio import loads
from resumeanalysis_legacy.data.adapters import (
    AdapterSpec,
    canonical_spec,
    inspect_rows,
    source_file_policy,
    source_rows,
)
from resumeanalysis_legacy.data.conversion import QualityPolicy, audit_converted, convert_dataset


def configure_parser(parser):
    sub = parser.add_subparsers(dest="dataset_command", required=True)
    for name in ("inspect", "convert"):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--adapter", choices=["canonical", "mapped"], default="canonical")
        p.add_argument("--mapping", help="Explicit adapter specification JSON, required for mapped")
        p.add_argument("--dataset", required=True)
        p.add_argument("--revision", required=True)
        if name == "convert":
            p.add_argument("--output", required=True)
            p.add_argument("--provenance", required=True)
            mode = p.add_mutually_exclusive_group()
            mode.add_argument("--strict", action="store_true", help="Default")
            mode.add_argument("--permissive", action="store_true")
            p.add_argument(
                "--retain-source",
                action="store_true",
                help="Explicitly permit local raw snapshots; may contain personal data",
            )
            p.add_argument("--quality-policy", help="Dataset-specific threshold/rationale JSON")
    p = sub.add_parser("audit")
    p.add_argument("--input", required=True)
    return parser


def run(args, parser):
    if args.dataset_command == "audit":
        result = audit_converted(args.input)
    else:
        if args.adapter == "mapped" and not args.mapping:
            parser.error("mapped adapter requires --mapping")
        spec = (
            AdapterSpec.model_validate(loads(Path(args.mapping).read_text()), strict=True)
            if args.mapping
            else canonical_spec()
        )
        if args.dataset_command == "inspect":
            result = inspect_rows(source_rows(args.input, args.dataset, args.revision), spec)
            result["source_file_policy"] = source_file_policy(args.input)
        else:
            policy = (
                QualityPolicy.model_validate(
                    loads(Path(args.quality_policy).read_text()), strict=True
                )
                if args.quality_policy
                else None
            )
            result = convert_dataset(
                args.input,
                args.output,
                spec,
                dataset=args.dataset,
                revision=args.revision,
                provenance=args.provenance,
                permissive=args.permissive,
                retain_source=args.retain_source,
                policy=policy,
            )
            if args.permissive and any(
                result["outcome_counts"][k] for k in ("converted_with_loss", "excluded", "rejected")
            ):
                warnings.warn(
                    "Permissive conversion has documented omissions/rejections; inspect per-record reports",
                    stacklevel=1,
                )
            result = {k: v for k, v in result.items() if k != "record_outcomes"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = configure_parser(
        argparse.ArgumentParser(
            prog="resumeanalysis datasets", description="Loss-aware JSONL dataset adapters"
        )
    )
    run(parser.parse_args(argv), parser)

"""Explicit network downloads; all other commands use pinned local files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Skill dataset foundation; sealed SkillSpan test")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download reviewed, checksum-pinned original assets")
    p.add_argument("--catalog", default="configs/skill_datasets.json")
    p.add_argument("--raw", default="data/skill-foundation/raw")
    p = sub.add_parser("guard", help="Build text-only SkillSpan heldout contamination guard")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("--catalog", default="configs/skill_datasets.json")
    p.add_argument("--raw", default="data/skill-foundation/raw")
    p.add_argument("--output", default="data/skill-foundation/normalized")
    p = sub.add_parser("audit")
    p.add_argument("--processed", default="data/skill-foundation/normalized")
    p.add_argument("--guard", required=True)
    p.add_argument("--output", default="artifacts/skill-foundation/audit")
    p.add_argument("--near-threshold", type=float, default=0.85)
    p = sub.add_parser("coverage")
    p.add_argument("--audit", default="artifacts/skill-foundation/audit")
    p.add_argument("--output", default="artifacts/skill-foundation/coverage")
    p.add_argument("--min-frequency", type=int, default=2)
    p = sub.add_parser("schema")
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--catalog", default="configs/skill_datasets.json")
    p.add_argument("--processed", default="data/skill-foundation/normalized")
    p.add_argument("--audit", default="artifacts/skill-foundation/audit")
    p.add_argument("--coverage", default="artifacts/skill-foundation/coverage")
    p.add_argument("--output", default="docs/skill_data")
    args = parser.parse_args()
    if args.command == "download":
        from resumeanalysis.skill_data.sources import download

        result = download(args.catalog, args.raw)
    elif args.command == "guard":
        from resumeanalysis.skill_data.guard import build_guard

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = build_guard(args.input, args.output)
    elif args.command == "preprocess":
        from resumeanalysis.skill_data.preprocess import preprocess

        result = preprocess(args.catalog, args.raw, args.output)
        result = {"status": result["status"], "files": len(result["files"])}
    elif args.command == "audit":
        from resumeanalysis.skill_data.audit import audit

        result = audit(args.processed, args.guard, args.output, threshold=args.near_threshold)
        result = {
            "status": result["status"],
            "totals": result["totals"],
            "quarantine_reasons": result["quarantine_reasons"],
        }
    elif args.command == "coverage":
        from resumeanalysis.skill_data.coverage import run_coverage

        result = run_coverage(args.audit, args.output, min_frequency=args.min_frequency)
        result = {name: value["token_exact"] for name, value in result["results"].items()}
    elif args.command == "schema":
        from resumeanalysis.jsonio import write_json
        from resumeanalysis.skill_data.schema import SkillRecord

        write_json(args.output, SkillRecord.model_json_schema())
        result = {"schema": str(args.output)}
    else:
        from resumeanalysis.skill_data.reports import publish_reports

        result = publish_reports(
            args.catalog, args.processed, args.audit, args.coverage, args.output
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

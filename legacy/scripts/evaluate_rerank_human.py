"""Freeze validation choices, then evaluate independently adjudicated human pools."""

import argparse

from resumeanalysis_legacy.experiments.rerank_human_evaluation import (
    evaluate_test,
    freeze_validation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-validation")
    freeze.add_argument("--annotations", required=True)
    freeze.add_argument("--predictions", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    freeze.add_argument("--preselect-k", type=int, nargs="+", default=[1, 5, 10, 20])
    test = commands.add_parser("evaluate-test")
    test.add_argument("--annotations", required=True)
    test.add_argument("--predictions", required=True)
    test.add_argument("--validation-predictions", required=True)
    test.add_argument("--frozen", required=True)
    test.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze-validation":
        freeze_validation(
            args.annotations,
            args.predictions,
            args.output,
            k_values=args.k,
            preselect_values=args.preselect_k,
        )
    else:
        evaluate_test(
            args.annotations,
            args.predictions,
            args.frozen,
            args.validation_predictions,
            args.output,
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

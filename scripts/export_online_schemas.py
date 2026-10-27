"""Export the versioned integration schemas from their tested Pydantic definitions."""

import argparse
from pathlib import Path

from resumeanalysis.jsonio import write_json
from resumeanalysis.recommendation.schema import (
    FeedbackEvent,
    Recommendation,
    SourceDocument,
    StructuredProfile,
    StructuredVacancy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/online_schemas")
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    for schema in (
        StructuredProfile,
        StructuredVacancy,
        Recommendation,
        FeedbackEvent,
        SourceDocument,
    ):
        write_json(target / f"{schema.__name__}.json", schema.model_json_schema())


if __name__ == "__main__":
    main()

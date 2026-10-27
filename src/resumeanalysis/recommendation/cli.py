"""Local recommendation, feedback, replay and profile commands."""

import json
from pathlib import Path
from types import SimpleNamespace

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder, TextEncoder
from resumeanalysis.jsonio import loads
from resumeanalysis.recommendation.engine import RecommendationConfig, RecommendationEngine
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.learner import LearnerConfig
from resumeanalysis.recommendation.repository import SQLiteFeedbackRepository
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy


def configure(sub):
    parser = sub.add_parser("recommend", help="Rank jobs with a local per-user online learner")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--vacancies", required=True)
    parser.add_argument("--database", default="artifacts/preferences.sqlite")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--e5-path", default="artifacts/e5")
    parser.add_argument("--smoke-lexical", action="store_true", help="Explicit diagnostic; not E5")
    parser.add_argument("--embedding-components", type=int, choices=[0, 16], default=0)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)
    parser.add_argument("--learner-config")
    for command in ("feedback", "preferences"):
        parser = sub.add_parser(command, help="Manage persisted explicit user preferences")
        parser.add_argument("--database", default="artifacts/preferences.sqlite")
        parser.add_argument("--user-id", required=True)
        if command == "feedback":
            parser.add_argument("--vacancy-id", required=True)
            parser.add_argument("--action", choices=["like", "dislike"], required=True)
            parser.add_argument("--reason")
            parser.add_argument("--event-id")
            parser.add_argument("--recommendation-id")
            parser.add_argument("--corrects-event-id")
        else:
            parser.add_argument("action", choices=["metrics", "reset", "replay"])
            parser.add_argument("--include-before-reset", action="store_true")
    parser = sub.add_parser("profile", help="Parse, extract, edit or confirm a local profile")
    parser.add_argument("action", choices=["parse", "extract", "structured", "edit", "confirm"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-id")
    parser.add_argument("--changes")
    parser.add_argument("--llm-config")
    parser.add_argument("--confirm", action="store_true")


def _read(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def _profile(args):
    from resumeanalysis.profiles.documents import LocalDocumentParser
    from resumeanalysis.profiles.extraction import (
        ProfileExtractionConfig,
        StructuredProfileExtractor,
    )

    target = Path(args.output)
    if target.exists():
        raise ValueError("Use a new profile output path to preserve the original")
    if args.action in {"parse", "extract"}:
        source = LocalDocumentParser().parse_file(args.input, source_id="uploaded-document")
        if args.action == "parse":
            result = source
        else:
            if not args.user_id:
                raise ValueError("Profile extraction requires --user-id")
            config = (
                ProfileExtractionConfig.model_validate(_read(args.llm_config), strict=True)
                if args.llm_config
                else None
            )
            extractor = StructuredProfileExtractor(config)
            try:
                result = extractor.extract_documents(args.user_id, [source])
            finally:
                extractor.close()
    elif args.action == "structured":
        if not args.user_id:
            raise ValueError("Structured profile creation requires --user-id")
        result = StructuredProfileExtractor.from_structured(args.user_id, _read(args.input))
    else:
        profile = StructuredProfile.model_validate(_read(args.input), strict=True)
        changes = _read(args.changes) if args.action == "edit" and args.changes else {}
        if args.action == "edit" and not args.changes:
            raise ValueError("Profile edit requires --changes")
        result = StructuredProfileExtractor.correct(
            profile, changes, confirm=args.confirm or args.action == "confirm"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return {"output": str(target), "review_status": getattr(result, "review_status", None)}


def run(args):
    if args.command == "profile":
        result = _profile(args)
    else:
        if args.command == "recommend":
            profile = StructuredProfile.model_validate(_read(args.profile), strict=True)
            vacancies = [
                StructuredVacancy.model_validate(v, strict=True) for v in _read(args.vacancies)
            ]
            if profile.review_status == "needs_review":
                raise ValueError("Confirm extracted profile facts before recommendation")
            configuration = RecommendationConfig(exploration_fraction=args.exploration_fraction)
            if not 1 <= args.limit <= 100:
                raise ValueError("Recommendation limit must be between 1 and 100")
            if not vacancies:
                print("[]")
                return
            encoder = LexicalSmokeEncoder() if args.smoke_lexical else TextEncoder(args.e5_path)
            builder = PreferenceFeatureBuilder(
                encoder, embedding_components=args.embedding_components
            )
            learner = (
                LearnerConfig.model_validate(_read(args.learner_config), strict=True)
                if args.learner_config
                else LearnerConfig()
            )
        repository = SQLiteFeedbackRepository(args.database)
        try:
            if args.command != "recommend":
                saved = repository.stored_configuration(args.user_id)
                if saved is None:
                    raise ValueError("No recommendation history exists for this user")
                # Feedback/replay use persisted snapshots and need no E5/model download.
                builder = SimpleNamespace(
                    identity=saved["feature_identity"],
                    schema_version=saved["feature_schema_version"],
                )
                learner = LearnerConfig.model_validate(saved["learner_config"], strict=True)
                configuration = RecommendationConfig()
            engine = RecommendationEngine(
                builder, repository, learner_config=learner, config=configuration
            )
            if args.command == "recommend":
                result = [
                    r.model_dump(mode="json")
                    for r in engine.recommend(profile.user_id, profile, vacancies, args.limit)
                ]
            elif args.command == "feedback":
                result = engine.record_feedback(
                    args.user_id,
                    args.vacancy_id,
                    args.action,
                    args.reason,
                    event_id=args.event_id,
                    recommendation_id=args.recommendation_id,
                    corrects_event_id=args.corrects_event_id,
                ).model_dump(mode="json")
            else:
                if args.action == "reset":
                    engine.reset(args.user_id)
                elif args.action == "replay":
                    engine.replay(args.user_id, include_before_reset=args.include_before_reset)
                result = engine.online_metrics(args.user_id)
        finally:
            repository.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

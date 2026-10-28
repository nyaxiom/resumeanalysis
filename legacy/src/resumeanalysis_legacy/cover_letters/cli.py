"""One explicit local command for scoring, resume selection and letter drafts."""

import asyncio
import json
from pathlib import Path

from resumeanalysis.cover_letters.generator import LocalCoverLetterGenerator
from resumeanalysis.cover_letters.schema import AnalysisRequest
from resumeanalysis.embeddings.encoder import TextEncoder
from resumeanalysis.jsonio import loads
from resumeanalysis_legacy.cover_letters.service import AnalysisService
from resumeanalysis_legacy.inference.backends import create_predictor
from resumeanalysis_legacy.qwen.cli import read_config


def configure(parser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--qwen-config", default="legacy/configs/cover_letter_qwen_cpu.json")
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--relevance-backend", choices=["mlp", "rerank"], default="mlp")
    parser.add_argument("--checkpoint", default="artifacts/run/model.pt")
    parser.add_argument("--e5-path", default="artifacts/e5")
    parser.add_argument("--rerank-config", default="legacy/configs/rerank.json")


def run(args):
    request = AnalysisRequest.model_validate(loads(Path(args.input).read_text()), strict=True)
    config = read_config(args.qwen_config)
    generator = LocalCoverLetterGenerator(config, max_words=args.max_words)
    encoder = TextEncoder(args.e5_path)
    if args.relevance_backend == "mlp":
        relevance = create_predictor("mlp", encoder=encoder, checkpoint=args.checkpoint)
    else:
        from resumeanalysis_legacy.rerank.config import RerankConfig

        rerank = RerankConfig.model_validate(
            loads(Path(args.rerank_config).read_text()), strict=True
        )
        relevance = create_predictor("rerank", encoder=encoder, config=rerank)
    result = asyncio.run(AnalysisService(relevance, generator).analyze(request))
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

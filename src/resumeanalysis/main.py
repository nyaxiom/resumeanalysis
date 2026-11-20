"""Three synchronous routes run inference off FastAPI's event loop."""

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from resumeanalysis.config import Settings
from resumeanalysis.cover_letter import ContextTooLong, CoverLetters, InvalidLetter
from resumeanalysis.embeddings import Embeddings
from resumeanalysis.recommender import Recommender, UnknownVacancy
from resumeanalysis.schemas import (
    CoverLetterRequest,
    CoverLetterResponse,
    FeedbackRequest,
    FeedbackResponse,
    RankRequest,
    RankResponse,
)
from resumeanalysis.storage import Storage


def create_app(settings: Settings | None = None, *, embeddings=None, letter_backend=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        storage = Storage(settings.database_path)
        app.state.recommender = Recommender(
            storage,
            embeddings if embeddings is not None else Embeddings(settings),
            settings,
        )
        app.state.letters = CoverLetters(settings, letter_backend)
        try:
            yield
        finally:
            storage.close()

    app = FastAPI(title="ResumeAnalysis", version="0.3.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()
                ]
            },
        )

    @app.post("/rank", response_model=RankResponse)
    def rank(request: RankRequest):
        try:
            return app.state.recommender.rank(request)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            raise HTTPException(
                503, "Ranking unavailable; check model assets and storage"
            ) from None

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest):
        try:
            return app.state.recommender.feedback(request)
        except UnknownVacancy as exc:
            raise HTTPException(404, str(exc)) from None
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            raise HTTPException(
                503, "Feedback unavailable; check model assets and storage"
            ) from None

    @app.post("/generate-cover-letter", response_model=CoverLetterResponse)
    def cover_letter(request: CoverLetterRequest):
        try:
            return CoverLetterResponse(cover_letter=app.state.letters.generate(request))
        except ContextTooLong as exc:
            raise HTTPException(422, str(exc)) from None
        except InvalidLetter as exc:
            raise HTTPException(502, str(exc)) from None
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(
                503, "Qwen unavailable; check model assets and configuration"
            ) from None

    return app


app = create_app()

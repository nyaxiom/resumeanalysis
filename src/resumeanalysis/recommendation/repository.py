"""SQLite event storage: commit feedback and model revision together, never pickle."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Protocol

from resumeanalysis.recommendation.schema import FeatureSnapshot, FeedbackEvent


class FeedbackSession(Protocol):
    def initialize(self, configuration: dict) -> dict: ...
    def events(self) -> list[FeedbackEvent]: ...
    def snapshot(self, recommendation_id: str) -> dict | None: ...
    def latest_snapshot(self, vacancy_id: str) -> dict | None: ...
    def save_snapshot(
        self, recommendation_id: str, vacancy_id: str, snapshot: FeatureSnapshot, prediction: float
    ) -> None: ...
    def append(self, event: FeedbackEvent) -> FeedbackEvent: ...
    def update_model(self, revision: int, reset_after_sequence: int) -> None: ...


class FeedbackRepository(Protocol):
    def transaction(self, user_id: str) -> ContextManager[FeedbackSession]: ...


class SQLiteSession:
    def __init__(self, connection, user_id):
        self.connection, self.user_id = connection, user_id

    def initialize(self, configuration):
        row = self.connection.execute(
            "SELECT configuration, revision, reset_after FROM preference_users WHERE user_id=?",
            (self.user_id,),
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO preference_users VALUES (?, ?, 0, 0)",
                (self.user_id, json.dumps(configuration, sort_keys=True)),
            )
            return {"revision": 0, "reset_after": 0}
        if json.loads(row[0]) != configuration:
            raise ValueError(
                "Preference configuration changed; use an explicit migration or a new database"
            )
        return {"revision": row[1], "reset_after": row[2]}

    def events(self):
        rows = self.connection.execute(
            "SELECT sequence, payload FROM feedback_events WHERE user_id=? ORDER BY sequence",
            (self.user_id,),
        )
        return [
            FeedbackEvent.model_validate_json(payload).model_copy(update={"sequence": seq})
            for seq, payload in rows
        ]

    def snapshot(self, recommendation_id):
        row = self.connection.execute(
            "SELECT vacancy_id, payload, prediction FROM recommendation_snapshots "
            "WHERE user_id=? AND recommendation_id=?",
            (self.user_id, recommendation_id),
        ).fetchone()
        return self._snapshot(row, recommendation_id)

    @staticmethod
    def _snapshot(row, recommendation_id):
        if row is None:
            return None
        return {
            "vacancy_id": row[0],
            "features": FeatureSnapshot.model_validate_json(row[1]),
            "prediction": row[2],
            "recommendation_id": recommendation_id,
        }

    def latest_snapshot(self, vacancy_id):
        row = self.connection.execute(
            "SELECT vacancy_id, payload, prediction, recommendation_id FROM recommendation_snapshots "
            "WHERE user_id=? AND vacancy_id=? ORDER BY sequence DESC LIMIT 1",
            (self.user_id, vacancy_id),
        ).fetchone()
        return self._snapshot(row, row[3]) if row else None

    def save_snapshot(self, recommendation_id, vacancy_id, snapshot, prediction):
        previous = self.connection.execute(
            "SELECT fingerprint FROM profile_versions WHERE user_id=? AND profile_id=? AND version=?",
            (self.user_id, snapshot.profile_id, snapshot.profile_version),
        ).fetchone()
        if previous and previous[0] != snapshot.profile_sha256:
            raise ValueError("Profile content changed without increasing its version")
        self.connection.execute(
            "INSERT OR IGNORE INTO profile_versions VALUES (?, ?, ?, ?)",
            (self.user_id, snapshot.profile_id, snapshot.profile_version, snapshot.profile_sha256),
        )
        self.connection.execute(
            "INSERT INTO recommendation_snapshots(user_id,recommendation_id,vacancy_id,payload,prediction) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.user_id, recommendation_id, vacancy_id, snapshot.model_dump_json(), prediction),
        )

    def append(self, event):
        if event.user_id != self.user_id:
            raise ValueError("Feedback user does not match transaction owner")
        cursor = self.connection.execute(
            "INSERT INTO feedback_events(user_id,event_id,payload) VALUES (?, ?, ?)",
            (self.user_id, event.event_id, event.model_dump_json()),
        )
        return event.model_copy(update={"sequence": cursor.lastrowid})

    def update_model(self, revision, reset_after_sequence):
        self.connection.execute(
            "UPDATE preference_users SET revision=?, reset_after=? WHERE user_id=?",
            (revision, reset_after_sequence, self.user_id),
        )


class SQLiteFeedbackRepository:
    """Local single-writer MVP; transactions also coordinate separate app processes.

    The history is the durable model state. Replaying it restores River through
    public APIs, without depending on private attribute or pickle formats.
    """

    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.path, isolation_level=None, timeout=30, check_same_thread=False
        )
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in {0, 1}:
            self.connection.close()
            raise ValueError("Unsupported feedback database schema")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS preference_users (
                user_id TEXT PRIMARY KEY, configuration TEXT NOT NULL,
                revision INTEGER NOT NULL, reset_after INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS recommendation_snapshots (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                recommendation_id TEXT NOT NULL, vacancy_id TEXT NOT NULL,
                payload TEXT NOT NULL, prediction REAL NOT NULL,
                UNIQUE(user_id, recommendation_id),
                FOREIGN KEY(user_id) REFERENCES preference_users(user_id));
            CREATE INDEX IF NOT EXISTS latest_recommendation
                ON recommendation_snapshots(user_id, vacancy_id, sequence);
            CREATE TABLE IF NOT EXISTS feedback_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                event_id TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(user_id, event_id),
                FOREIGN KEY(user_id) REFERENCES preference_users(user_id));
            CREATE TABLE IF NOT EXISTS profile_versions (
                user_id TEXT NOT NULL, profile_id TEXT NOT NULL, version INTEGER NOT NULL,
                fingerprint TEXT NOT NULL, PRIMARY KEY(user_id, profile_id, version));
            PRAGMA user_version=1;
        """)

    @contextmanager
    def transaction(self, user_id):
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield SQLiteSession(self.connection, user_id)
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def close(self):
        with self._lock:
            self.connection.close()

    def stored_configuration(self, user_id):
        with self._lock:
            row = self.connection.execute(
                "SELECT configuration FROM preference_users WHERE user_id=?", (user_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

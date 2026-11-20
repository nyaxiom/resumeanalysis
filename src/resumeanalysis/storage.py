"""SQLite transactions shared by feedback and numeric model snapshots; no pickle."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class Storage:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(path), isolation_level=None, timeout=30, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        try:
            # Never repurpose the old application's database or an unrelated database.
            tables = {
                r[0]
                for r in self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables and "ml_service_meta" not in tables:
                raise ValueError("Use a new database path for the minimal ML service")
            if tables:
                version = self.connection.execute("SELECT version FROM ml_service_meta").fetchone()
                if version is None or version[0] != 1:
                    raise ValueError("Unsupported ML service database version")
            self.connection.execute("PRAGMA journal_mode=WAL")
            with self.transaction() as db:
                db.execute("CREATE TABLE IF NOT EXISTS ml_service_meta (version INTEGER NOT NULL)")
                if not db.execute("SELECT 1 FROM ml_service_meta").fetchone():
                    db.execute("INSERT INTO ml_service_meta VALUES (1)")
                db.execute("""CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0,
                    model_json TEXT, model_checksum TEXT)""")
                db.execute("""CREATE TABLE IF NOT EXISTS vacancies (
                    user_id TEXT NOT NULL, vacancy_id TEXT NOT NULL,
                    title TEXT NOT NULL, description TEXT NOT NULL,
                    PRIMARY KEY (user_id, vacancy_id))""")
                db.execute("""CREATE TABLE IF NOT EXISTS feedback (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, vacancy_id TEXT NOT NULL,
                    title TEXT NOT NULL, description TEXT NOT NULL,
                    liked INTEGER NOT NULL CHECK (liked IN (0, 1)),
                    UNIQUE (user_id, vacancy_id))""")
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def transaction(self):
        # BEGIN IMMEDIATE also serializes writers in separate processes.
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def close(self):
        with self.lock:
            self.connection.close()

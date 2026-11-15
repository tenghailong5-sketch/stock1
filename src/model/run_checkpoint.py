"""Durable SQLite checkpoints for resumable, deterministic model batches.

A completed batch is reusable only when the run and batch input fingerprints match.
Running/failed batches remain pending after interruption and can be retried.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class CheckpointConflictError(ValueError):
    """Raised when a run or batch is resumed with different inputs."""


class SQLiteCheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_runs (
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, stage)
                );
                CREATE TABLE IF NOT EXISTS checkpoint_batches (
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
                    result_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, stage, batch_id),
                    FOREIGN KEY (run_id, stage)
                        REFERENCES checkpoint_runs(run_id, stage) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit on success and always close the SQLite connection."""
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def start_run(
        self, run_id: str, stage: str, input_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not all(str(value).strip() for value in (run_id, stage, input_fingerprint)):
            raise ValueError("run_id, stage and input_fingerprint are required")
        metadata_json = self._json(dict(metadata or {}))
        with self._connection() as db:
            row = db.execute(
                "SELECT input_fingerprint FROM checkpoint_runs WHERE run_id=? AND stage=?",
                (run_id, stage),
            ).fetchone()
            if row and row["input_fingerprint"] != input_fingerprint:
                raise CheckpointConflictError(
                    "run input fingerprint changed; use a new run_id or stage"
                )
            if not row:
                db.execute(
                    """INSERT INTO checkpoint_runs
                       (run_id, stage, input_fingerprint, metadata_json)
                       VALUES (?, ?, ?, ?)""",
                    (run_id, stage, input_fingerprint, metadata_json),
                )

    def register_batches(
        self, run_id: str, stage: str, batches: Mapping[str, str],
    ) -> None:
        """Register batch_id -> input fingerprint without resetting completed work."""
        with self._connection() as db:
            for batch_id, fingerprint in batches.items():
                if not str(batch_id).strip() or not str(fingerprint).strip():
                    raise ValueError("batch_id and batch fingerprint are required")
                row = db.execute(
                    """SELECT input_fingerprint FROM checkpoint_batches
                       WHERE run_id=? AND stage=? AND batch_id=?""",
                    (run_id, stage, batch_id),
                ).fetchone()
                if row and row["input_fingerprint"] != fingerprint:
                    raise CheckpointConflictError(
                        f"batch input fingerprint changed: {batch_id}"
                    )
                if not row:
                    db.execute(
                        """INSERT INTO checkpoint_batches
                           (run_id, stage, batch_id, input_fingerprint, status)
                           VALUES (?, ?, ?, ?, 'pending')""",
                        (run_id, stage, batch_id, fingerprint),
                    )

    def pending_batches(self, run_id: str, stage: str) -> list[str]:
        """Return pending, failed, or interrupted-running batches in stable order."""
        with self._connection() as db:
            rows = db.execute(
                """SELECT batch_id FROM checkpoint_batches
                   WHERE run_id=? AND stage=? AND status!='completed'
                   ORDER BY batch_id""",
                (run_id, stage),
            ).fetchall()
        return [row["batch_id"] for row in rows]

    def mark_started(self, run_id: str, stage: str, batch_id: str) -> None:
        with self._connection() as db:
            cur = db.execute(
                """UPDATE checkpoint_batches
                   SET status='running', error=NULL, updated_at=CURRENT_TIMESTAMP
                   WHERE run_id=? AND stage=? AND batch_id=? AND status!='completed'""",
                (run_id, stage, batch_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown or completed batch: {batch_id}")

    def mark_completed(
        self, run_id: str, stage: str, batch_id: str, result: Any,
    ) -> None:
        result_json = self._json(result)
        with self._connection() as db:
            cur = db.execute(
                """UPDATE checkpoint_batches
                   SET status='completed', result_json=?, error=NULL,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE run_id=? AND stage=? AND batch_id=?""",
                (result_json, run_id, stage, batch_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown batch: {batch_id}")

    def mark_failed(
        self, run_id: str, stage: str, batch_id: str, error: str,
    ) -> None:
        with self._connection() as db:
            cur = db.execute(
                """UPDATE checkpoint_batches
                   SET status='failed', error=?, updated_at=CURRENT_TIMESTAMP
                   WHERE run_id=? AND stage=? AND batch_id=?""",
                (str(error)[:2000], run_id, stage, batch_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown batch: {batch_id}")

    def load_result(self, run_id: str, stage: str, batch_id: str) -> Any:
        with self._connection() as db:
            row = db.execute(
                """SELECT status, result_json FROM checkpoint_batches
                   WHERE run_id=? AND stage=? AND batch_id=?""",
                (run_id, stage, batch_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown batch: {batch_id}")
        if row["status"] != "completed" or row["result_json"] is None:
            raise ValueError(f"batch is not completed: {batch_id}")
        return json.loads(row["result_json"])

    def summary(self, run_id: str, stage: str) -> dict[str, int]:
        with self._connection() as db:
            rows = db.execute(
                """SELECT status, COUNT(*) AS count FROM checkpoint_batches
                   WHERE run_id=? AND stage=? GROUP BY status""",
                (run_id, stage),
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        return {
            "total": sum(counts.values()),
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }

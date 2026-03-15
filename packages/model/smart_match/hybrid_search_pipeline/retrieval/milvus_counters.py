"""
Local Counter Store

Provides a simple per-key counter backed by SQLite. This is intended for
single-writer deployments (one API instance) and avoids extra Milvus state for
model_id allocation.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CounterValue:
    key: str
    value: int
    updated_at: int


class MilvusCounterStore:
    """Compatibility wrapper preserving the old public API.

    `collection_name` now acts as a logical namespace in the local SQLite DB.
    """

    def __init__(self, *, collection_name: str = "sv_counters") -> None:
        self.collection_name = (collection_name or "sv_counters").strip()
        if not self.collection_name:
            raise ValueError("collection_name must be non-empty.")
        db_path = Path(os.getenv("COUNTERS_DB_PATH", "data/model_id_counters.sqlite")).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )

    def get(self, key: str) -> Optional[CounterValue]:
        key = (key or "").strip()
        if not key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT key, value, updated_at FROM counters WHERE namespace = ? AND key = ?",
                (self.collection_name, key),
            ).fetchone()
        if row is None:
            return None
        return CounterValue(key=str(row[0]), value=int(row[1]), updated_at=int(row[2]))

    def set(self, key: str, value: int) -> CounterValue:
        key = (key or "").strip()
        if not key:
            raise ValueError("key must be non-empty.")
        updated_at = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO counters(namespace, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (self.collection_name, key, int(value), updated_at),
            )
        return CounterValue(key=key, value=int(value), updated_at=updated_at)

    def next(self, key: str) -> CounterValue:
        key = (key or "").strip()
        if not key:
            raise ValueError("key must be non-empty.")
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT value FROM counters WHERE namespace = ? AND key = ?",
                    (self.collection_name, key),
                ).fetchone()
                next_value = 1 if row is None else int(row[0]) + 1
                updated_at = int(time.time())
                conn.execute(
                    """
                    INSERT INTO counters(namespace, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(namespace, key)
                    DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (self.collection_name, key, next_value, updated_at),
                )
                conn.commit()
        return CounterValue(key=key, value=next_value, updated_at=updated_at)


__all__ = ["CounterValue", "MilvusCounterStore"]

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class SQLiteTTLCache:
    """Small local cache for external tool results.

    The cache deliberately stores only tool arguments and public search results;
    credentials are never included in keys or values.
    """

    def __init__(self, path: str = ".cache/knowledge_agent.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=5)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, namespace: str, payload: Any) -> Optional[Any]:
        key = self.make_key(namespace, payload)
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value, expires_at FROM tool_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row[1] <= now:
                connection.execute("DELETE FROM tool_cache WHERE cache_key = ?", (key,))
                return None
            return json.loads(row[0])

    def set(self, namespace: str, payload: Any, value: Any, ttl_seconds: int) -> None:
        key = self.make_key(namespace, payload)
        now = time.time()
        serialized = json.dumps(value, ensure_ascii=False)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_cache(cache_key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    created_at = excluded.created_at
                """,
                (key, serialized, now + ttl_seconds, now),
            )


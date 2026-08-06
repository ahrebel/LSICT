"""SQLite-backed cache for hashes, pHashes, CLIP embeddings, detection results.

Cache validity: each row stores (size, mtime). On lookup, if the current file
fingerprint differs from the cached one, the row is invalidated and re-computed.

This makes re-runs nearly instant for unchanged files.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from lsict.core import cache_key, file_fingerprint, IS_WINDOWS

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS file_meta (
    key          TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    mtime        REAL NOT NULL,
    sha256       TEXT,
    phash_hex    TEXT,
    has_person   INTEGER,
    has_face     INTEGER,
    nsfw_score   REAL,
    clip_dim     INTEGER,
    clip_emb     BLOB,
    updated_at   REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_sha     ON file_meta(sha256);
CREATE INDEX IF NOT EXISTS idx_phash   ON file_meta(phash_hex);
CREATE INDEX IF NOT EXISTS idx_person  ON file_meta(has_person);
CREATE INDEX IF NOT EXISTS idx_face    ON file_meta(has_face);
"""


class Cache:
    """Thin wrapper around sqlite3. One row per file."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        # WAL gives better concurrent-read performance; synchronous=NORMAL is
        # a reasonable durability/perf trade-off for a cache.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- internal ----------

    def _row(self, path: Path) -> Optional[sqlite3.Row]:
        """Return the cache row if it exists AND is still valid for this file."""
        key = cache_key(path)
        cur = self.conn.execute(
            "SELECT size, mtime, sha256, phash_hex, has_person, has_face, "
            "nsfw_score, clip_dim, clip_emb FROM file_meta WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        size, mtime = row[0], row[1]
        try:
            cur_size, cur_mtime = file_fingerprint(path)
        except OSError:
            return None
        # Allow tiny mtime drift (FS resolution); strict on size.
        if cur_size != size or abs(cur_mtime - mtime) > 1.0:
            # Stale — drop the row and force recomputation.
            self.conn.execute("DELETE FROM file_meta WHERE key = ?", (key,))
            self.conn.commit()
            return None
        return row

    def _ensure_base_row(self, path: Path) -> None:
        """Ensure a row exists with the current (size, mtime). No-op if valid."""
        key = cache_key(path)
        try:
            size, mtime = file_fingerprint(path)
        except OSError:
            return
        self.conn.execute(
            "INSERT INTO file_meta(key, path, size, mtime) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  path = excluded.path, "
            "  size = CASE WHEN file_meta.size = excluded.size "
            "              AND abs(file_meta.mtime - excluded.mtime) <= 1.0 "
            "         THEN file_meta.size ELSE excluded.size END, "
            "  mtime = CASE WHEN file_meta.size = excluded.size "
            "               AND abs(file_meta.mtime - excluded.mtime) <= 1.0 "
            "          THEN file_meta.mtime ELSE excluded.mtime END",
            (key, str(path), size, mtime),
        )

    # ---------- public getters ----------

    def get_sha256(self, path: Path) -> Optional[str]:
        row = self._row(path)
        return row[2] if row else None

    def get_phash(self, path: Path) -> Optional[int]:
        row = self._row(path)
        if row is None or row[3] is None:
            return None
        return int(row[3], 16)

    def get_detection(self, path: Path) -> Optional[tuple[bool, bool]]:
        row = self._row(path)
        if row is None or row[4] is None or row[5] is None:
            return None
        return (bool(row[4]), bool(row[5]))

    def get_nsfw_score(self, path: Path) -> Optional[float]:
        row = self._row(path)
        return row[6] if row and row[6] is not None else None

    def get_clip(self, path: Path) -> Optional[np.ndarray]:
        row = self._row(path)
        if row is None or row[7] is None or row[8] is None:
            return None
        dim, blob = row[7], row[8]
        return np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)

    # ---------- public setters (batched-friendly) ----------

    def set_sha256(self, path: Path, sha: str) -> None:
        self._ensure_base_row(path)
        self.conn.execute(
            "UPDATE file_meta SET sha256 = ?, updated_at = strftime('%s','now') "
            "WHERE key = ?",
            (sha, cache_key(path)),
        )

    def set_phash(self, path: Path, phash: int) -> None:
        self._ensure_base_row(path)
        self.conn.execute(
            "UPDATE file_meta SET phash_hex = ?, updated_at = strftime('%s','now') "
            "WHERE key = ?",
            (format(int(phash), "064x"), cache_key(path)),
        )

    def set_detection(self, path: Path, has_person: bool, has_face: bool) -> None:
        self._ensure_base_row(path)
        self.conn.execute(
            "UPDATE file_meta SET has_person = ?, has_face = ?, "
            "updated_at = strftime('%s','now') WHERE key = ?",
            (int(has_person), int(has_face), cache_key(path)),
        )

    def set_nsfw_score(self, path: Path, score: float) -> None:
        self._ensure_base_row(path)
        self.conn.execute(
            "UPDATE file_meta SET nsfw_score = ?, updated_at = strftime('%s','now') "
            "WHERE key = ?",
            (float(score), cache_key(path)),
        )

    def set_clip(self, path: Path, emb: np.ndarray) -> None:
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        self._ensure_base_row(path)
        self.conn.execute(
            "UPDATE file_meta SET clip_dim = ?, clip_emb = ?, "
            "updated_at = strftime('%s','now') WHERE key = ?",
            (int(emb.shape[-1]), emb.tobytes(), cache_key(path)),
        )

    # ---------- transaction helpers ----------

    def commit(self) -> None:
        self.conn.commit()

    def transaction(self):
        """Use as a context manager around a batch of writes."""
        return _Transaction(self.conn)

    # ---------- maintenance ----------

    def info(self) -> dict:
        cur = self.conn.execute("SELECT COUNT(*) FROM file_meta")
        n_total = cur.fetchone()[0]
        cur = self.conn.execute("SELECT COUNT(*) FROM file_meta WHERE sha256 IS NOT NULL")
        n_sha = cur.fetchone()[0]
        cur = self.conn.execute("SELECT COUNT(*) FROM file_meta WHERE phash_hex IS NOT NULL")
        n_phash = cur.fetchone()[0]
        cur = self.conn.execute("SELECT COUNT(*) FROM file_meta WHERE clip_emb IS NOT NULL")
        n_clip = cur.fetchone()[0]
        cur = self.conn.execute("SELECT COUNT(*) FROM file_meta WHERE has_person IS NOT NULL")
        n_det = cur.fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "total_rows": n_total,
            "with_sha256": n_sha,
            "with_phash": n_phash,
            "with_clip_emb": n_clip,
            "with_detection": n_det,
            "db_size_mb": round(size_bytes / (1024 * 1024), 1),
        }

    def prune_missing(self) -> int:
        """Remove rows whose underlying file no longer exists. Returns count removed."""
        cur = self.conn.execute("SELECT key, path FROM file_meta")
        to_delete = []
        for key, p in cur.fetchall():
            if not Path(p).exists():
                to_delete.append(key)
        if to_delete:
            self.conn.executemany(
                "DELETE FROM file_meta WHERE key = ?",
                [(k,) for k in to_delete],
            )
            self.conn.commit()
            self.conn.execute("VACUUM")
        return len(to_delete)

    def clear(self) -> None:
        self.conn.execute("DELETE FROM file_meta")
        self.conn.commit()
        self.conn.execute("VACUUM")


class _Transaction:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False


def default_cache_path(kept_dir: Path) -> Path:
    """Default cache lives next to KEPT (so it travels with the project).

    Falls back to the pre-0.3.0 `.imgcurate_cache.sqlite` filename if that
    file exists and the new one doesn't, so old caches keep working.
    """
    new = kept_dir / ".lsict_cache.sqlite"
    legacy = kept_dir / ".imgcurate_cache.sqlite"
    if not new.exists() and legacy.exists():
        return legacy
    return new

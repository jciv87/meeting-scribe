"""Speaker profile storage backed by SQLite."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DEFAULT_DB_PATH = Path.home() / "meeting-scribe" / "profiles" / "voices.db"
_EMBEDDING_DTYPE = np.float32


@dataclass
class SpeakerProfile:
    """A single identified speaker and their stored voice embeddings."""

    id: int
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)

    def centroid(self) -> np.ndarray:
        """Return the mean embedding across all stored samples."""
        if not self.embeddings:
            raise ValueError(
                f"Speaker '{self.name}' (id={self.id}) has no embeddings."
            )
        return np.mean(self.embeddings, axis=0)


class ProfileStore:
    """Manage speaker profiles and their embeddings in a SQLite database."""

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS speakers (
                    id         INTEGER PRIMARY KEY,
                    name       TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id           INTEGER PRIMARY KEY,
                    speaker_id   INTEGER REFERENCES speakers(id),
                    embedding    BLOB NOT NULL,
                    meeting_date TEXT,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_speaker(self, name: str) -> int:
        """Insert a new speaker and return their assigned id."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO speakers (name) VALUES (?)", (name,)
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def add_embedding(
        self,
        speaker_id: int,
        embedding: np.ndarray,
        meeting_date: str | None = None,
    ) -> None:
        """Store an embedding blob for a speaker."""
        blob = embedding.astype(_EMBEDDING_DTYPE).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT INTO embeddings (speaker_id, embedding, meeting_date) VALUES (?, ?, ?)",
                (speaker_id, blob, meeting_date),
            )
            self._conn.commit()

    def rename_speaker(self, speaker_id: int, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE speakers SET name = ? WHERE id = ?", (name, speaker_id)
            )
            self._conn.commit()

    def delete_speaker(self, speaker_id: int) -> None:
        """Delete a speaker and all their associated embeddings."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM embeddings WHERE speaker_id = ?", (speaker_id,)
            )
            self._conn.execute(
                "DELETE FROM speakers WHERE id = ?", (speaker_id,)
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all_profiles(self) -> list[SpeakerProfile]:
        """Load all speakers with their embeddings."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name FROM speakers ORDER BY id"
            ).fetchall()

        profiles: list[SpeakerProfile] = []
        for speaker_id, name in rows:
            embeddings = self._load_embeddings(speaker_id)
            profiles.append(SpeakerProfile(id=speaker_id, name=name, embeddings=embeddings))
        return profiles

    def get_profile(self, speaker_id: int) -> SpeakerProfile | None:
        """Return a single profile or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            return None
        speaker_id_, name = row
        return SpeakerProfile(
            id=speaker_id_,
            name=name,
            embeddings=self._load_embeddings(speaker_id_),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_embeddings(self, speaker_id: int) -> list[np.ndarray]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding FROM embeddings WHERE speaker_id = ? ORDER BY id",
                (speaker_id,),
            ).fetchall()
        return [
            np.frombuffer(blob, dtype=_EMBEDDING_DTYPE).copy()
            for (blob,) in rows
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

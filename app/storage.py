"""SQLite metadata persistence, isolated for future PostgreSQL migration."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import DocumentRecord, PaperMetadata


class MetadataStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                    page_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
                    created_at TEXT NOT NULL, result_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return self._record(row) if row else None

    def save_document(self, record: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO documents
                (id, filename, sha256, page_count, chunk_count, status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.id, record.filename, record.sha256, record.page_count, record.chunk_count,
                 record.status, record.metadata.model_dump_json(), record.created_at.isoformat()),
            )

    def update_status(self, document_id: str, status: str, chunk_count: int | None = None) -> None:
        with self._connect() as connection:
            if chunk_count is None:
                connection.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))
            else:
                connection.execute("UPDATE documents SET status = ?, chunk_count = ? WHERE id = ?", (status, chunk_count, document_id))

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._record(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def log_query(self, question: str, result_count: int) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO queries (question, created_at, result_count) VALUES (?, ?, ?)",
                               (question, datetime.now(timezone.utc).isoformat(), result_count))

    def query_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0])

    @staticmethod
    def _record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(id=row["id"], filename=row["filename"], sha256=row["sha256"],
                              page_count=row["page_count"], chunk_count=row["chunk_count"],
                              status=row["status"], metadata=PaperMetadata(**json.loads(row["metadata_json"])),
                              created_at=datetime.fromisoformat(row["created_at"]))

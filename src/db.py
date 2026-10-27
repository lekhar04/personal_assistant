"""SQLite storage for conversations, messages, long-term memories and document records.

Three separate concerns live in three separate tables:
  * messages   -> conversation history (always tied to exactly one conversation)
  * memories   -> long-term user memory (global to the user, survives chat deletion)
  * documents  -> uploaded document records (chunks themselves live in the vector DB)
"""
import json
import sqlite3
from contextlib import contextmanager
from typing import Optional

from . import config


class DatabaseError(Exception):
    """Raised for any SQLite failure, with a user-presentable message."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    meta TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'conversation' CHECK (scope IN ('conversation', 'global')),
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _conn():
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH, timeout=10)
    except (sqlite3.Error, OSError) as e:
        raise DatabaseError(f"Could not open the database: {e}") from e
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Database error: {e}") from e
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


# ---------------- conversations ----------------
def create_conversation(title: str = config.DEFAULT_CHAT_TITLE) -> int:
    with _conn() as c:
        return c.execute("INSERT INTO conversations (title) VALUES (?)", (title,)).lastrowid


def list_conversations() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM conversations ORDER BY updated_at DESC, id DESC").fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: int, title: str) -> None:
    with _conn() as c:
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def delete_conversation(conversation_id: int) -> None:
    """Deletes the conversation, its messages and conversation-scoped documents.

    Global documents survive (detached from the conversation); long-term memories
    survive too (their source_conversation_id becomes NULL).
    """
    with _conn() as c:
        c.execute(
            "UPDATE documents SET conversation_id = NULL WHERE conversation_id = ? AND scope = 'global'",
            (conversation_id,),
        )
        c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ---------------- messages ----------------
def add_message(conversation_id: int, role: str, content: str, meta: Optional[dict] = None) -> int:
    with _conn() as c:
        mid = c.execute(
            "INSERT INTO messages (conversation_id, role, content, meta) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(meta) if meta else None),
        ).lastrowid
        c.execute("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?", (conversation_id,))
    return mid


def _row_to_message(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["meta"] = json.loads(d["meta"]) if d.get("meta") else None
    return d


def get_messages(conversation_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def get_recent_messages(conversation_id: int, limit: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    return [_row_to_message(r) for r in reversed(rows)]


# ---------------- long-term memories ----------------
def add_memory(content: str, category: str, source_conversation_id: Optional[int]) -> int:
    with _conn() as c:
        return c.execute(
            "INSERT INTO memories (content, category, source_conversation_id) VALUES (?, ?, ?)",
            (content, category, source_conversation_id),
        ).lastrowid


def list_memories() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM memories ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_memory(memory_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


# ---------------- documents ----------------
def add_document(conversation_id: Optional[int], filename: str, scope: str, chunk_count: int) -> int:
    with _conn() as c:
        return c.execute(
            "INSERT INTO documents (conversation_id, filename, scope, chunk_count) VALUES (?, ?, ?, ?)",
            (conversation_id, filename, scope, chunk_count),
        ).lastrowid


def update_document_chunks(document_id: int, chunk_count: int) -> None:
    with _conn() as c:
        c.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, document_id))


def list_documents(conversation_id: Optional[int]) -> list[dict]:
    """Documents visible in a conversation: its own plus all global ones."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE scope = 'global' OR conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_conversation_only_documents(conversation_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM documents WHERE conversation_id = ? AND scope = 'conversation'",
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(document_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id = ?", (document_id,))

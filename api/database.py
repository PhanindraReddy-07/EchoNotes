"""
EchoNotes Database - SQLite persistent storage for job history and dashboard.

Stores all processing jobs with transcription results, analysis data,
and document outputs for dashboard display and history browsing.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database file location (next to uploads/outputs)
DB_PATH = Path(os.environ.get("ECHONOTES_DB", "./data/echonotes.db"))


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db():
    """Context manager for database connections."""
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'Untitled',
                status      TEXT NOT NULL DEFAULT 'pending',
                language    TEXT DEFAULT 'en',
                format      TEXT DEFAULT 'html',
                use_ai      INTEGER DEFAULT 1,

                -- Audio metadata
                audio_filename  TEXT,
                audio_duration  REAL DEFAULT 0,
                audio_size      INTEGER DEFAULT 0,

                -- Transcription results
                transcript      TEXT,
                confidence      REAL DEFAULT 0,
                word_count      INTEGER DEFAULT 0,

                -- Analysis results (stored as JSON)
                analysis_json   TEXT,

                -- Document output
                document_path   TEXT,
                document_format TEXT,

                -- Error tracking
                error_message   TEXT,
                
                -- Processing times (seconds)
                time_preprocess   REAL DEFAULT 0,
                time_transcribe   REAL DEFAULT 0,
                time_analyze      REAL DEFAULT 0,
                time_generate     REAL DEFAULT 0,
                time_total        REAL DEFAULT 0,

                -- Timestamps
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_language ON sessions(language);
        """)


# ============== CRUD Operations ==============

def create_session(
    session_id: str,
    title: str = "Untitled",
    language: str = "en",
    fmt: str = "html",
    use_ai: bool = True,
    audio_filename: str = None,
    audio_size: int = 0,
) -> dict:
    """Create a new processing session."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, title, status, language, format, use_ai,
                audio_filename, audio_size, created_at, updated_at)
               VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, title, language, fmt, int(use_ai),
             audio_filename, audio_size, now, now),
        )
    return get_session(session_id)


def update_session(session_id: str, **kwargs) -> dict:
    """Update session fields. Pass any column name as keyword argument."""
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    with get_db() as conn:
        conn.execute(f"UPDATE sessions SET {sets} WHERE id = ?", vals)
    return get_session(session_id)


def complete_session(
    session_id: str,
    transcript: str,
    confidence: float,
    word_count: int,
    audio_duration: float,
    analysis: dict,
    document_path: str,
    document_format: str,
    time_preprocess: float = 0,
    time_transcribe: float = 0,
    time_analyze: float = 0,
    time_generate: float = 0,
    time_total: float = 0,
):
    """Mark a session as completed with all results."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            """UPDATE sessions SET
                status = 'completed',
                transcript = ?, confidence = ?, word_count = ?,
                audio_duration = ?,
                analysis_json = ?,
                document_path = ?, document_format = ?,
                time_preprocess = ?, time_transcribe = ?,
                time_analyze = ?, time_generate = ?, time_total = ?,
                updated_at = ?, completed_at = ?
               WHERE id = ?""",
            (transcript, confidence, word_count, audio_duration,
             json.dumps(analysis, default=str),
             document_path, document_format,
             time_preprocess, time_transcribe, time_analyze,
             time_generate, time_total, now, now, session_id),
        )


def fail_session(session_id: str, error_message: str):
    """Mark session as failed."""
    update_session(session_id, status="failed", error_message=error_message)


def get_session(session_id: str) -> Optional[dict]:
    """Get a single session by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_sessions(
    limit: int = 20,
    offset: int = 0,
    status: str = None,
    language: str = None,
    search: str = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """List sessions with filtering, search, and pagination."""
    conditions = []
    params = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if language:
        conditions.append("language = ?")
        params.append(language)
    if search:
        conditions.append("(title LIKE ? OR transcript LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    allowed_sorts = {"created_at", "updated_at", "title", "confidence",
                     "word_count", "audio_duration", "time_total"}
    if sort_by not in allowed_sorts:
        sort_by = "created_at"
    order = "DESC" if sort_order.lower() == "desc" else "ASC"

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM sessions {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""SELECT * FROM sessions {where}
                ORDER BY {sort_by} {order}
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

    return {
        "sessions": [_row_to_dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def delete_session(session_id: str) -> bool:
    """Delete a session and return True if it existed."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return cursor.rowcount > 0


def get_dashboard_stats() -> dict:
    """Aggregate statistics for the dashboard."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                  AS total_sessions,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) AS processing,
                COALESCE(SUM(audio_duration), 0)          AS total_audio_seconds,
                COALESCE(SUM(word_count), 0)              AS total_words,
                COALESCE(AVG(CASE WHEN status='completed' THEN confidence END), 0) AS avg_confidence,
                COALESCE(AVG(CASE WHEN status='completed' THEN time_total END), 0) AS avg_processing_time
            FROM sessions
        """).fetchone()

        # Per-language breakdown
        langs = conn.execute("""
            SELECT language,
                   COUNT(*) AS count,
                   COALESCE(AVG(confidence), 0) AS avg_conf
            FROM sessions
            WHERE status='completed'
            GROUP BY language
        """).fetchall()

        # Per-format breakdown
        fmts = conn.execute("""
            SELECT document_format,
                   COUNT(*) AS count
            FROM sessions
            WHERE status='completed' AND document_format IS NOT NULL
            GROUP BY document_format
        """).fetchall()

        # Recent 7-day activity
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        daily = conn.execute("""
            SELECT DATE(created_at) AS day, COUNT(*) AS count
            FROM sessions
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY day
        """, (week_ago,)).fetchall()

    return {
        "total_sessions": row["total_sessions"],
        "completed": row["completed"],
        "failed": row["failed"],
        "processing": row["processing"],
        "total_audio_minutes": round(row["total_audio_seconds"] / 60, 1),
        "total_words": row["total_words"],
        "avg_confidence": round(row["avg_confidence"], 3),
        "avg_processing_time": round(row["avg_processing_time"], 1),
        "by_language": {r["language"]: {"count": r["count"], "avg_confidence": round(r["avg_conf"], 3)} for r in langs},
        "by_format": {r["document_format"]: r["count"] for r in fmts if r["document_format"]},
        "daily_activity": [{"date": r["day"], "count": r["count"]} for r in daily],
    }


# ============== Helpers ==============

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a Row to a plain dict, deserializing JSON fields."""
    d = dict(row)
    if d.get("analysis_json"):
        try:
            d["analysis"] = json.loads(d["analysis_json"])
        except (json.JSONDecodeError, TypeError):
            d["analysis"] = None
    else:
        d["analysis"] = None
    d.pop("analysis_json", None)
    d["use_ai"] = bool(d.get("use_ai", 0))
    return d

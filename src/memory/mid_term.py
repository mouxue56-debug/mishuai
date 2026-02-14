"""Mid-term memory - daily summaries and session data.

Stores session summaries and daily work logs in SQLite.
Used to provide context about "what happened earlier today".
"""

import json
from datetime import datetime, date
from typing import Optional

import aiosqlite

from src.utils.logger import get_logger

logger = get_logger("memory.mid")


class MidTermMemory:
    """SQLite-backed session and daily summary storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                message_count INTEGER DEFAULT 0,
                messages_json TEXT,
                summary TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                summary TEXT NOT NULL,
                task_count INTEGER DEFAULT 0,
                inquiry_count INTEGER DEFAULT 0,
                highlights TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await self._db.commit()
        logger.info("Mid-term memory tables ready")

    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()

    async def save_session(self, messages: list[dict], summary: str):
        """Save a conversation session.

        Args:
            messages: List of message dicts.
            summary: Session summary text.
        """
        now = datetime.now()
        await self._db.execute(
            """INSERT INTO sessions (date, start_time, end_time, message_count, messages_json, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now.strftime("%Y-%m-%d"),
                now.isoformat(),
                now.isoformat(),
                len(messages),
                json.dumps(messages, ensure_ascii=False),
                summary,
            ),
        )
        await self._db.commit()
        logger.debug(f"Session saved: {len(messages)} messages")

    async def get_summary(self, target_date: Optional[str] = None) -> Optional[str]:
        """Get the daily summary for a date.

        Args:
            target_date: Date string (YYYY-MM-DD). None for today.

        Returns:
            Summary text, or None if not found.
        """
        if target_date is None:
            target_date = date.today().isoformat()

        async with self._db.execute(
            "SELECT summary FROM daily_summaries WHERE date = ?", (target_date,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def save_daily_summary(
        self,
        summary: str,
        task_count: int = 0,
        inquiry_count: int = 0,
        highlights: str = "",
    ):
        """Save or update today's daily summary.

        Args:
            summary: Summary text.
            task_count: Number of tasks handled.
            inquiry_count: Number of customer inquiries.
            highlights: Key highlights of the day.
        """
        today = date.today().isoformat()
        await self._db.execute(
            """INSERT OR REPLACE INTO daily_summaries
               (date, summary, task_count, inquiry_count, highlights)
               VALUES (?, ?, ?, ?, ?)""",
            (today, summary, task_count, inquiry_count, highlights),
        )
        await self._db.commit()
        logger.info(f"Daily summary saved for {today}")

    async def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        """Get the most recent sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session dicts.
        """
        async with self._db.execute(
            """SELECT date, start_time, message_count, summary
               FROM sessions ORDER BY id DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "date": row[0],
                    "start_time": row[1],
                    "message_count": row[2],
                    "summary": row[3],
                }
                for row in rows
            ]

"""Long-term memory - persistent knowledge and data.

Stores memos, reminders, customer records, and learned preferences in SQLite.
This data persists across sessions and days.
"""

from datetime import datetime
from typing import Optional

import aiosqlite

from src.utils.logger import get_logger

logger = get_logger("memory.long")


class LongTermMemory:
    """SQLite-backed persistent memory for memos, reminders, and knowledge."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Create tables if they don't exist."""
        self._db = await aiosqlite.connect(self.db_path)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                speaker_id TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                is_archived INTEGER DEFAULT 0
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                remind_at TEXT NOT NULL,
                repeat_pattern TEXT DEFAULT 'none',
                is_completed INTEGER DEFAULT 0,
                speaker_id TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS customer_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                note TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS learned_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                context TEXT,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        await self._db.commit()
        logger.info("Long-term memory tables ready")

    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()

    # --- Memos ---

    async def save_memo(self, content: str, category: str = "general",
                        speaker_id: str = None) -> int:
        """Save a new memo.

        Args:
            content: Memo text.
            category: Category tag.
            speaker_id: Who created this memo.

        Returns:
            The memo ID.
        """
        cursor = await self._db.execute(
            "INSERT INTO memos (content, category, speaker_id) VALUES (?, ?, ?)",
            (content, category, speaker_id),
        )
        await self._db.commit()
        memo_id = cursor.lastrowid
        logger.info(f"Memo saved: #{memo_id} [{category}] {content[:50]}...")
        return memo_id

    async def search_memos(self, query: str) -> list[dict]:
        """Search memos by keyword.

        Args:
            query: Search keyword.

        Returns:
            List of matching memo dicts.
        """
        async with self._db.execute(
            """SELECT id, content, category, created_at
               FROM memos
               WHERE content LIKE ? AND is_archived = 0
               ORDER BY created_at DESC LIMIT 10""",
            (f"%{query}%",),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "content": r[1], "category": r[2], "created_at": r[3]}
                for r in rows
            ]

    async def get_recent_memos(self, limit: int = 5) -> list[dict]:
        """Get the most recent memos.

        Args:
            limit: Maximum number of memos.

        Returns:
            List of memo dicts.
        """
        async with self._db.execute(
            """SELECT id, content, category, created_at
               FROM memos WHERE is_archived = 0
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "content": r[1], "category": r[2], "created_at": r[3]}
                for r in rows
            ]

    async def archive_memo(self, memo_id: int):
        """Archive (soft-delete) a memo."""
        await self._db.execute(
            "UPDATE memos SET is_archived = 1 WHERE id = ?", (memo_id,)
        )
        await self._db.commit()

    # --- Reminders ---

    async def save_reminder(
        self, content: str, remind_at: datetime,
        repeat: str = "none", speaker_id: str = None
    ) -> int:
        """Save a new reminder.

        Args:
            content: Reminder text.
            remind_at: When to trigger.
            repeat: Repeat pattern ("none", "daily", "weekly").
            speaker_id: Who set this reminder.

        Returns:
            Reminder ID.
        """
        cursor = await self._db.execute(
            """INSERT INTO reminders (content, remind_at, repeat_pattern, speaker_id)
               VALUES (?, ?, ?, ?)""",
            (content, remind_at.isoformat(), repeat, speaker_id),
        )
        await self._db.commit()
        reminder_id = cursor.lastrowid
        logger.info(f"Reminder saved: #{reminder_id} at {remind_at}")
        return reminder_id

    async def get_pending_reminders(self) -> list[dict]:
        """Get all reminders that are due (remind_at <= now and not completed)."""
        now = datetime.now().isoformat()
        async with self._db.execute(
            """SELECT id, content, remind_at, repeat_pattern
               FROM reminders
               WHERE remind_at <= ? AND is_completed = 0
               ORDER BY remind_at""",
            (now,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "remind_at": r[2],
                    "repeat": r[3],
                }
                for r in rows
            ]

    async def get_upcoming_reminders(self, limit: int = 5) -> list[dict]:
        """Get upcoming reminders (not yet due)."""
        now = datetime.now().isoformat()
        async with self._db.execute(
            """SELECT id, content, remind_at, repeat_pattern
               FROM reminders
               WHERE remind_at > ? AND is_completed = 0
               ORDER BY remind_at LIMIT ?""",
            (now, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "remind_at": r[2],
                    "repeat": r[3],
                }
                for r in rows
            ]

    async def complete_reminder(self, reminder_id: int):
        """Mark a reminder as completed."""
        await self._db.execute(
            "UPDATE reminders SET is_completed = 1 WHERE id = ?", (reminder_id,)
        )
        await self._db.commit()

    # --- Customer Notes ---

    async def add_customer_note(self, customer_name: str, note: str,
                                 category: str = "general") -> int:
        """Add a note about a customer."""
        cursor = await self._db.execute(
            "INSERT INTO customer_notes (customer_name, note, category) VALUES (?, ?, ?)",
            (customer_name, note, category),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_customer_notes(self, customer_name: str) -> list[dict]:
        """Get all notes for a customer."""
        async with self._db.execute(
            """SELECT id, note, category, created_at
               FROM customer_notes
               WHERE customer_name LIKE ?
               ORDER BY created_at DESC""",
            (f"%{customer_name}%",),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "note": r[1], "category": r[2], "created_at": r[3]}
                for r in rows
            ]

    # --- Learned Preferences ---

    async def set_preference(self, key: str, value: str, context: str = ""):
        """Save or update a learned preference."""
        await self._db.execute(
            """INSERT OR REPLACE INTO learned_preferences (key, value, context, updated_at)
               VALUES (?, ?, ?, datetime('now', 'localtime'))""",
            (key, value, context),
        )
        await self._db.commit()

    async def get_preference(self, key: str) -> Optional[str]:
        """Get a learned preference value."""
        async with self._db.execute(
            "SELECT value FROM learned_preferences WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

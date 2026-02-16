"""Long-term memory - persistent knowledge and data.

Stores memos, reminders, customer records, and learned preferences in SQLite.
This data persists across sessions and days.

Schema versioning: Every table change bumps SCHEMA_VERSION and adds a
migration function in _MIGRATIONS. On startup, the database is automatically
migrated from its current version to the latest.
"""

from datetime import datetime
from typing import Optional

import aiosqlite

from src.utils.logger import get_logger

logger = get_logger("memory.long")

# --- Schema version history ---
# v1: Initial tables (memos, reminders, customer_notes, learned_preferences)
# v2: Add turn_logs table for TurnContext analytics
# v3: Upgrade learned_preferences with confidence/source/expires_at
# v4: Add event_memories table for key decisions/events
# v5: Add speakers, conversation_sessions tables; session_id on turn_logs
SCHEMA_VERSION = 5


class LongTermMemory:
    """SQLite-backed persistent memory for memos, reminders, and knowledge.

    Includes automatic schema migration — safe to add new columns/tables
    by incrementing SCHEMA_VERSION and adding a migration function.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Create tables if they don't exist, then run migrations."""
        self._db = await aiosqlite.connect(self.db_path)

        # --- Schema version tracking ---
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await self._db.commit()

        # Get current version (0 if table is empty = fresh database)
        async with self._db.execute(
            "SELECT version FROM _schema_version WHERE id = 1"
        ) as cursor:
            row = await cursor.fetchone()
            current_version = row[0] if row else 0

        if current_version == 0:
            # Fresh database — create all tables at latest version
            await self._create_all_tables()
            await self._set_schema_version(SCHEMA_VERSION)
            logger.info(f"Long-term memory initialized (schema v{SCHEMA_VERSION})")
        elif current_version < SCHEMA_VERSION:
            # Existing database — run migrations
            logger.info(
                f"Schema migration needed: v{current_version} → v{SCHEMA_VERSION}"
            )
            await self._run_migrations(current_version)
            logger.info(f"Schema migration complete (now v{SCHEMA_VERSION})")
        else:
            logger.info(f"Long-term memory ready (schema v{current_version})")

    async def _create_all_tables(self):
        """Create all tables at the latest schema version."""
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
                category TEXT DEFAULT 'general',
                confidence TEXT DEFAULT 'inferred',
                source TEXT DEFAULT '',
                context TEXT,
                expires_at TEXT,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # v4: Event memories for key decisions and events
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS event_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL DEFAULT 'decision',
                summary TEXT NOT NULL,
                detail TEXT,
                importance TEXT DEFAULT 'normal',
                source_request_id TEXT,
                expires_at TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # v2: Turn logs for TurnContext analytics
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS turn_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                intent TEXT DEFAULT 'chat',
                input_text TEXT,
                response_text TEXT,
                emotion TEXT DEFAULT 'neutral',
                speaker_id TEXT,
                session_id TEXT,
                llm_model TEXT,
                llm_duration_ms REAL DEFAULT 0,
                tts_engine TEXT,
                tts_duration_ms REAL DEFAULT 0,
                audio_bytes INTEGER DEFAULT 0,
                tool_count INTEGER DEFAULT 0,
                total_duration_ms REAL DEFAULT 0,
                aborted INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # v5: Registered speakers
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                role TEXT DEFAULT 'visitor',
                language TEXT DEFAULT 'japanese',
                permissions TEXT DEFAULT '["cattery_knowledge"]',
                style TEXT DEFAULT 'formal',
                notes TEXT DEFAULT '',
                invited_by TEXT,
                relationship TEXT DEFAULT '',
                is_active INTEGER DEFAULT 1,
                registered_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # v5: Conversation sessions for audit
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL,
                started_at TEXT DEFAULT (datetime('now', 'localtime')),
                ended_at TEXT,
                turn_count INTEGER DEFAULT 0,
                summary TEXT,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id)
            )
        """)

        await self._db.commit()

    async def _set_schema_version(self, version: int):
        """Set the current schema version."""
        await self._db.execute(
            """INSERT OR REPLACE INTO _schema_version (id, version, updated_at)
               VALUES (1, ?, datetime('now', 'localtime'))""",
            (version,),
        )
        await self._db.commit()

    async def _run_migrations(self, from_version: int):
        """Run all migrations from from_version to SCHEMA_VERSION.

        Each migration is a function that takes the db connection and
        performs the needed ALTER TABLE / CREATE TABLE operations.
        """
        migrations = {
            # v1 → v2: Add turn_logs table
            2: self._migrate_v1_to_v2,
            # v2 → v3: Upgrade learned_preferences with confidence/source/expires_at
            3: self._migrate_v2_to_v3,
            # v3 → v4: Add event_memories table
            4: self._migrate_v3_to_v4,
            # v4 → v5: Add speakers, conversation_sessions; session_id on turn_logs
            5: self._migrate_v4_to_v5,
        }

        for target_ver in range(from_version + 1, SCHEMA_VERSION + 1):
            migrate_fn = migrations.get(target_ver)
            if migrate_fn:
                logger.info(f"Running migration → v{target_ver}...")
                await migrate_fn()
                await self._set_schema_version(target_ver)
            else:
                logger.warning(f"No migration function for v{target_ver}, skipping")
                await self._set_schema_version(target_ver)

    async def _migrate_v1_to_v2(self):
        """Migration v1 → v2: Add turn_logs table for TurnContext analytics."""
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS turn_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                intent TEXT DEFAULT 'chat',
                input_text TEXT,
                response_text TEXT,
                emotion TEXT DEFAULT 'neutral',
                speaker_id TEXT,
                llm_model TEXT,
                llm_duration_ms REAL DEFAULT 0,
                tts_engine TEXT,
                tts_duration_ms REAL DEFAULT 0,
                audio_bytes INTEGER DEFAULT 0,
                tool_count INTEGER DEFAULT 0,
                total_duration_ms REAL DEFAULT 0,
                aborted INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await self._db.commit()
        logger.info("Created turn_logs table")

    async def _migrate_v2_to_v3(self):
        """Migration v2 → v3: Add confidence/source/expires_at to learned_preferences."""
        # Add new columns (SQLite ADD COLUMN is safe even if column exists check fails)
        try:
            await self._db.execute(
                "ALTER TABLE learned_preferences ADD COLUMN category TEXT DEFAULT 'general'"
            )
        except Exception:
            pass  # Column already exists
        try:
            await self._db.execute(
                "ALTER TABLE learned_preferences ADD COLUMN confidence TEXT DEFAULT 'inferred'"
            )
        except Exception:
            pass
        try:
            await self._db.execute(
                "ALTER TABLE learned_preferences ADD COLUMN source TEXT DEFAULT ''"
            )
        except Exception:
            pass
        try:
            await self._db.execute(
                "ALTER TABLE learned_preferences ADD COLUMN expires_at TEXT"
            )
        except Exception:
            pass
        await self._db.commit()
        logger.info("Upgraded learned_preferences table (added confidence/source/expires_at)")

    async def _migrate_v3_to_v4(self):
        """Migration v3 → v4: Add event_memories table."""
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS event_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL DEFAULT 'decision',
                summary TEXT NOT NULL,
                detail TEXT,
                importance TEXT DEFAULT 'normal',
                source_request_id TEXT,
                expires_at TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        await self._db.commit()
        logger.info("Created event_memories table")

    async def _migrate_v4_to_v5(self):
        """Migration v4 → v5: Add speakers, conversation_sessions; session_id on turn_logs."""
        # Add session_id column to turn_logs
        try:
            await self._db.execute(
                "ALTER TABLE turn_logs ADD COLUMN session_id TEXT DEFAULT NULL"
            )
        except Exception:
            pass  # Column already exists

        # Create speakers table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                role TEXT DEFAULT 'visitor',
                language TEXT DEFAULT 'japanese',
                permissions TEXT DEFAULT '["cattery_knowledge"]',
                style TEXT DEFAULT 'formal',
                notes TEXT DEFAULT '',
                invited_by TEXT,
                relationship TEXT DEFAULT '',
                is_active INTEGER DEFAULT 1,
                registered_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # Create conversation_sessions table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL,
                started_at TEXT DEFAULT (datetime('now', 'localtime')),
                ended_at TEXT,
                turn_count INTEGER DEFAULT 0,
                summary TEXT,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id)
            )
        """)

        await self._db.commit()
        logger.info("Created speakers + conversation_sessions tables; added session_id to turn_logs")

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
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "content": r[1], "category": r[2], "created_at": r[3]}
                for r in rows
            ]

    async def update_memo(self, memo_id: int, content: str,
                          category: str = None):
        """Update an existing memo's content and optionally category.

        Args:
            memo_id: Memo ID.
            content: New content text.
            category: New category (None = don't change).
        """
        if category is not None:
            await self._db.execute(
                "UPDATE memos SET content = ?, category = ? WHERE id = ?",
                (content, category, memo_id),
            )
        else:
            await self._db.execute(
                "UPDATE memos SET content = ? WHERE id = ?",
                (content, memo_id),
            )
        await self._db.commit()
        logger.info(f"Memo updated: #{memo_id}")

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

    async def update_reminder(self, reminder_id: int, content: str = None,
                              remind_at: str = None):
        """Update a reminder's content and/or time.

        Args:
            reminder_id: Reminder ID.
            content: New content (None = don't change).
            remind_at: New ISO datetime string (None = don't change).
        """
        if content and remind_at:
            await self._db.execute(
                "UPDATE reminders SET content = ?, remind_at = ? WHERE id = ?",
                (content, remind_at, reminder_id),
            )
        elif content:
            await self._db.execute(
                "UPDATE reminders SET content = ? WHERE id = ?",
                (content, reminder_id),
            )
        elif remind_at:
            await self._db.execute(
                "UPDATE reminders SET remind_at = ? WHERE id = ?",
                (remind_at, reminder_id),
            )
        await self._db.commit()
        logger.info(f"Reminder updated: #{reminder_id}")

    async def complete_reminder(self, reminder_id: int):
        """Mark a reminder as completed."""
        await self._db.execute(
            "UPDATE reminders SET is_completed = 1 WHERE id = ?", (reminder_id,)
        )
        await self._db.commit()

    async def delete_reminder(self, reminder_id: int):
        """Permanently delete a reminder.

        Args:
            reminder_id: Reminder ID.
        """
        await self._db.execute(
            "DELETE FROM reminders WHERE id = ?", (reminder_id,)
        )
        await self._db.commit()
        logger.info(f"Reminder deleted: #{reminder_id}")

    async def get_completed_reminders(self, limit: int = 20) -> list[dict]:
        """Get completed reminders.

        Args:
            limit: Maximum number of reminders.

        Returns:
            List of completed reminder dicts.
        """
        async with self._db.execute(
            """SELECT id, content, remind_at, repeat_pattern, created_at
               FROM reminders WHERE is_completed = 1
               ORDER BY remind_at DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "content": r[1], "remind_at": r[2],
                    "repeat": r[3], "created_at": r[4],
                }
                for r in rows
            ]

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

    async def set_preference(
        self,
        key: str,
        value: str,
        category: str = "general",
        confidence: str = "inferred",
        source: str = "",
        context: str = "",
        expires_at: Optional[str] = None,
    ):
        """Save or update a learned preference.

        Args:
            key: Preference key (e.g. "report_format", "favorite_drink").
            value: Preference value.
            category: Category ("communication", "work_style", "taste", "general").
            confidence: "explicit" (user stated directly) or "inferred" (AI deduced).
            source: Source context (e.g. conversation request_id).
            context: Additional context about when/why this was learned.
            expires_at: Optional ISO datetime for expiration (None = permanent).
        """
        # If updating an inferred preference with explicit confirmation, upgrade confidence
        existing = await self.get_preference_detail(key)
        if existing and confidence == "explicit" and existing.get("confidence") == "inferred":
            logger.info(f"Preference '{key}' upgraded: inferred → explicit")

        await self._db.execute(
            """INSERT OR REPLACE INTO learned_preferences
               (key, value, category, confidence, source, context, expires_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))""",
            (key, value, category, confidence, source, context, expires_at),
        )
        await self._db.commit()
        logger.info(f"Preference set: {key}={value} [{confidence}/{category}]")

    async def get_preference(self, key: str) -> Optional[str]:
        """Get a learned preference value (returns None if expired)."""
        async with self._db.execute(
            """SELECT value FROM learned_preferences
               WHERE key = ? AND (expires_at IS NULL OR expires_at > datetime('now', 'localtime'))""",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_preference_detail(self, key: str) -> Optional[dict]:
        """Get full preference details including confidence and expiration."""
        async with self._db.execute(
            """SELECT key, value, category, confidence, source, context, expires_at, updated_at
               FROM learned_preferences WHERE key = ?""",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "key": row[0], "value": row[1], "category": row[2],
                "confidence": row[3], "source": row[4], "context": row[5],
                "expires_at": row[6], "updated_at": row[7],
            }

    async def get_all_active_preferences(self) -> list[dict]:
        """Get all non-expired preferences, ordered by confidence then recency.

        Returns:
            List of preference dicts. Explicit preferences come first.
        """
        async with self._db.execute(
            """SELECT key, value, category, confidence, source, context, expires_at, updated_at
               FROM learned_preferences
               WHERE expires_at IS NULL OR expires_at > datetime('now', 'localtime')
               ORDER BY
                   CASE confidence WHEN 'explicit' THEN 0 ELSE 1 END,
                   updated_at DESC"""
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "key": r[0], "value": r[1], "category": r[2],
                    "confidence": r[3], "source": r[4], "context": r[5],
                    "expires_at": r[6], "updated_at": r[7],
                }
                for r in rows
            ]

    async def delete_preference(self, key: str):
        """Delete a learned preference."""
        await self._db.execute(
            "DELETE FROM learned_preferences WHERE key = ?", (key,)
        )
        await self._db.commit()
        logger.info(f"Preference deleted: {key}")

    async def cleanup_expired_preferences(self) -> int:
        """Remove expired preferences. Returns count of deleted rows."""
        cursor = await self._db.execute(
            """DELETE FROM learned_preferences
               WHERE expires_at IS NOT NULL AND expires_at <= datetime('now', 'localtime')"""
        )
        await self._db.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired preferences")
        return count

    async def compress_old_preferences(self, max_preferences: int = 30) -> int:
        """Compress preferences when there are too many.

        Strategy:
        1. Keep all explicit preferences (user stated them)
        2. Remove oldest inferred preferences that exceed the limit
        3. If same category has >5 inferred items, keep only newest 3

        Args:
            max_preferences: Max total preferences to keep.

        Returns:
            Number of deleted preferences.
        """
        # Count total preferences
        async with self._db.execute(
            "SELECT COUNT(*) FROM learned_preferences"
        ) as cursor:
            total = (await cursor.fetchone())[0]

        if total <= max_preferences:
            return 0

        deleted = 0

        # Step 1: For each category, if >5 inferred preferences, keep newest 3
        async with self._db.execute(
            """SELECT category, COUNT(*) as cnt FROM learned_preferences
               WHERE confidence = 'inferred'
               GROUP BY category HAVING cnt > 5"""
        ) as cursor:
            bloated_categories = await cursor.fetchall()

        for cat, cnt in bloated_categories:
            # Get IDs to keep (newest 3 inferred per category)
            async with self._db.execute(
                """SELECT id FROM learned_preferences
                   WHERE category = ? AND confidence = 'inferred'
                   ORDER BY updated_at DESC LIMIT 3""",
                (cat,),
            ) as cursor:
                keep_ids = [r[0] for r in await cursor.fetchall()]

            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                del_cursor = await self._db.execute(
                    f"""DELETE FROM learned_preferences
                       WHERE category = ? AND confidence = 'inferred'
                       AND id NOT IN ({placeholders})""",
                    (cat, *keep_ids),
                )
                deleted += del_cursor.rowcount

        # Step 2: If still over limit, remove oldest inferred globally
        async with self._db.execute(
            "SELECT COUNT(*) FROM learned_preferences"
        ) as cursor:
            remaining = (await cursor.fetchone())[0]

        if remaining > max_preferences:
            excess = remaining - max_preferences
            del_cursor = await self._db.execute(
                """DELETE FROM learned_preferences WHERE id IN (
                       SELECT id FROM learned_preferences
                       WHERE confidence = 'inferred'
                       ORDER BY updated_at ASC LIMIT ?
                   )""",
                (excess,),
            )
            deleted += del_cursor.rowcount

        if deleted > 0:
            await self._db.commit()
            logger.info(f"Compressed preferences: {deleted} old inferred entries removed")

        return deleted

    # --- Event Memories ---

    async def save_event(
        self,
        event_type: str,
        summary: str,
        detail: str = "",
        importance: str = "normal",
        source_request_id: str = "",
        expires_at: Optional[str] = None,
    ) -> int:
        """Save a key event or decision to long-term memory.

        Args:
            event_type: "decision", "milestone", "incident", "preference_change".
            summary: Short description (1 sentence).
            detail: Optional longer context.
            importance: "low", "normal", "high".
            source_request_id: Which conversation this came from.
            expires_at: Optional ISO datetime for auto-expiration.

        Returns:
            Event ID.
        """
        cursor = await self._db.execute(
            """INSERT INTO event_memories
               (event_type, summary, detail, importance, source_request_id, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_type, summary, detail, importance, source_request_id, expires_at),
        )
        await self._db.commit()
        event_id = cursor.lastrowid
        logger.info(f"Event saved: #{event_id} [{event_type}] {summary[:50]}")
        return event_id

    async def get_recent_events(self, limit: int = 10) -> list[dict]:
        """Get recent non-expired events ordered by recency.

        Args:
            limit: Max events to return.

        Returns:
            List of event dicts.
        """
        async with self._db.execute(
            """SELECT id, event_type, summary, detail, importance, created_at, expires_at
               FROM event_memories
               WHERE expires_at IS NULL OR expires_at > datetime('now', 'localtime')
               ORDER BY
                   CASE importance WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                   created_at DESC
               LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "event_type": r[1], "summary": r[2],
                    "detail": r[3], "importance": r[4],
                    "created_at": r[5], "expires_at": r[6],
                }
                for r in rows
            ]

    async def cleanup_expired_events(self) -> int:
        """Remove expired event memories. Returns count of deleted."""
        cursor = await self._db.execute(
            """DELETE FROM event_memories
               WHERE expires_at IS NOT NULL AND expires_at <= datetime('now', 'localtime')"""
        )
        await self._db.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired event memories")
        return count

    # --- Turn Logs (TurnContext analytics) ---

    async def save_turn_log(self, turn) -> int:
        """Save a TurnContext to the turn_logs table.

        Args:
            turn: TurnContext dataclass instance.

        Returns:
            Log entry ID.
        """
        session_id = getattr(turn, 'session_id', None)
        cursor = await self._db.execute(
            """INSERT INTO turn_logs
               (request_id, intent, input_text, response_text, emotion,
                speaker_id, session_id, llm_model, llm_duration_ms, tts_engine,
                tts_duration_ms, audio_bytes, tool_count, total_duration_ms,
                aborted, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                turn.request_id,
                turn.intent,
                turn.input_text[:500] if turn.input_text else None,
                turn.response_text[:500] if turn.response_text else None,
                turn.emotion,
                turn.speaker_id,
                session_id,
                turn.llm_model,
                turn.llm_duration_ms,
                turn.tts_engine,
                turn.tts_duration_ms,
                turn.audio_bytes,
                len(turn.tool_results),
                turn.total_duration_ms,
                1 if turn.aborted else 0,
                turn.error,
            ),
        )
        await self._db.commit()

        # Update session turn count
        if session_id:
            await self._db.execute(
                """UPDATE conversation_sessions
                   SET turn_count = turn_count + 1
                   WHERE id = ?""",
                (session_id,),
            )
            await self._db.commit()

        return cursor.lastrowid

    async def get_recent_turn_logs(self, limit: int = 20) -> list[dict]:
        """Get recent turn logs for analytics.

        Returns:
            List of turn log dicts ordered by most recent first.
        """
        async with self._db.execute(
            """SELECT request_id, intent, input_text, response_text, emotion,
                      llm_model, llm_duration_ms, tts_engine, tts_duration_ms,
                      audio_bytes, tool_count, total_duration_ms, aborted,
                      error, created_at
               FROM turn_logs ORDER BY id DESC LIMIT ?""",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "request_id": r[0], "intent": r[1],
                    "input_text": r[2], "response_text": r[3],
                    "emotion": r[4], "llm_model": r[5],
                    "llm_duration_ms": r[6], "tts_engine": r[7],
                    "tts_duration_ms": r[8], "audio_bytes": r[9],
                    "tool_count": r[10], "total_duration_ms": r[11],
                    "aborted": bool(r[12]), "error": r[13],
                    "created_at": r[14],
                }
                for r in rows
            ]

    # --- Speakers ---

    async def register_speaker(
        self,
        speaker_id: str,
        display_name: str,
        role: str = "visitor",
        language: str = "japanese",
        permissions: str = '["cattery_knowledge"]',
        style: str = "formal",
        notes: str = "",
        invited_by: str = None,
        relationship: str = "",
    ):
        """Register a new speaker in the database.

        Args:
            speaker_id: Unique ID (matches .npy filename).
            display_name: Display name.
            role: 'boss', 'staff', or 'visitor'.
            language: Primary language.
            permissions: JSON array of allowed tool names.
            style: Speech style ('casual', 'polite', 'formal').
            notes: Freeform notes.
            invited_by: speaker_id of who invited this person.
            relationship: Relationship description.
        """
        await self._db.execute(
            """INSERT OR REPLACE INTO speakers
               (id, display_name, role, language, permissions, style,
                notes, invited_by, relationship, is_active, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now', 'localtime'))""",
            (speaker_id, display_name, role, language, permissions, style,
             notes, invited_by, relationship),
        )
        await self._db.commit()
        logger.info(f"Speaker registered: {speaker_id} ({display_name}, role={role})")

    async def get_speaker(self, speaker_id: str) -> Optional[dict]:
        """Get a speaker by ID.

        Returns:
            Speaker dict or None if not found.
        """
        async with self._db.execute(
            """SELECT id, display_name, role, language, permissions, style,
                      notes, invited_by, relationship, is_active,
                      registered_at, updated_at
               FROM speakers WHERE id = ?""",
            (speaker_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0], "display_name": row[1], "role": row[2],
                "language": row[3], "permissions": row[4], "style": row[5],
                "notes": row[6], "invited_by": row[7], "relationship": row[8],
                "is_active": bool(row[9]), "registered_at": row[10],
                "updated_at": row[11],
            }

    async def get_all_speakers(self, include_inactive: bool = False) -> list[dict]:
        """Get all registered speakers.

        Args:
            include_inactive: If True, include deactivated speakers.

        Returns:
            List of speaker dicts.
        """
        query = """SELECT id, display_name, role, language, permissions, style,
                          notes, invited_by, relationship, is_active,
                          registered_at, updated_at
                   FROM speakers"""
        if not include_inactive:
            query += " WHERE is_active = 1"
        query += " ORDER BY registered_at"

        async with self._db.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "display_name": r[1], "role": r[2],
                    "language": r[3], "permissions": r[4], "style": r[5],
                    "notes": r[6], "invited_by": r[7], "relationship": r[8],
                    "is_active": bool(r[9]), "registered_at": r[10],
                    "updated_at": r[11],
                }
                for r in rows
            ]

    async def update_speaker(self, speaker_id: str, **kwargs):
        """Update speaker fields.

        Args:
            speaker_id: Speaker to update.
            **kwargs: Fields to update (display_name, role, language,
                      permissions, style, notes, relationship).
        """
        allowed_fields = {
            "display_name", "role", "language", "permissions",
            "style", "notes", "relationship",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return

        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(speaker_id)

        await self._db.execute(
            f"UPDATE speakers SET {set_clauses}, updated_at = datetime('now', 'localtime') WHERE id = ?",
            values,
        )
        await self._db.commit()
        logger.info(f"Speaker updated: {speaker_id} ({list(updates.keys())})")

    async def deactivate_speaker(self, speaker_id: str):
        """Deactivate a speaker (soft delete).

        Args:
            speaker_id: Speaker to deactivate.
        """
        await self._db.execute(
            "UPDATE speakers SET is_active = 0, updated_at = datetime('now', 'localtime') WHERE id = ?",
            (speaker_id,),
        )
        await self._db.commit()
        logger.info(f"Speaker deactivated: {speaker_id}")

    # --- Conversation Sessions ---

    async def create_session(self, speaker_id: str) -> str:
        """Create a new conversation session.

        Args:
            speaker_id: Who this session belongs to.

        Returns:
            Session UUID.
        """
        import uuid
        session_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO conversation_sessions (id, speaker_id) VALUES (?, ?)",
            (session_id, speaker_id),
        )
        await self._db.commit()
        logger.info(f"Session created: {session_id} for speaker {speaker_id}")
        return session_id

    async def end_session(self, session_id: str, summary: str = None):
        """End a conversation session.

        Args:
            session_id: Session to end.
            summary: Optional LLM-generated summary.
        """
        await self._db.execute(
            """UPDATE conversation_sessions
               SET ended_at = datetime('now', 'localtime'), summary = ?
               WHERE id = ?""",
            (summary, session_id),
        )
        await self._db.commit()
        logger.info(f"Session ended: {session_id}")

    async def get_sessions(
        self, speaker_id: str = None, limit: int = 20
    ) -> list[dict]:
        """Get conversation sessions, optionally filtered by speaker.

        Args:
            speaker_id: Filter by speaker (None = all speakers).
            limit: Max sessions to return.

        Returns:
            List of session dicts with speaker display_name.
        """
        if speaker_id:
            query = """
                SELECT cs.id, cs.speaker_id, s.display_name, cs.started_at,
                       cs.ended_at, cs.turn_count, cs.summary
                FROM conversation_sessions cs
                LEFT JOIN speakers s ON cs.speaker_id = s.id
                WHERE cs.speaker_id = ?
                ORDER BY cs.started_at DESC LIMIT ?
            """
            params = (speaker_id, limit)
        else:
            query = """
                SELECT cs.id, cs.speaker_id, s.display_name, cs.started_at,
                       cs.ended_at, cs.turn_count, cs.summary
                FROM conversation_sessions cs
                LEFT JOIN speakers s ON cs.speaker_id = s.id
                ORDER BY cs.started_at DESC LIMIT ?
            """
            params = (limit,)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "speaker_id": r[1],
                    "speaker_name": r[2] or "Unknown",
                    "started_at": r[3], "ended_at": r[4],
                    "turn_count": r[5], "summary": r[6],
                }
                for r in rows
            ]

    async def get_session_turns(self, session_id: str) -> list[dict]:
        """Get all turn logs for a specific session.

        Args:
            session_id: Session to query.

        Returns:
            List of turn log dicts ordered chronologically.
        """
        async with self._db.execute(
            """SELECT request_id, intent, input_text, response_text, emotion,
                      speaker_id, llm_model, total_duration_ms, created_at
               FROM turn_logs WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "request_id": r[0], "intent": r[1],
                    "input_text": r[2], "response_text": r[3],
                    "emotion": r[4], "speaker_id": r[5],
                    "llm_model": r[6], "total_duration_ms": r[7],
                    "created_at": r[8],
                }
                for r in rows
            ]

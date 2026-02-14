"""Memory management system - orchestrates short/mid/long-term memory.

Inspired by N.E.K.O's multi-tier memory architecture:
- Short-term: Current conversation context (in-memory)
- Mid-term: Daily summaries and session data (SQLite)
- Long-term: Customer files, knowledge, decisions (SQLite)
"""

import asyncio
from datetime import datetime
from typing import Optional

from src.memory.short_term import ShortTermMemory
from src.memory.mid_term import MidTermMemory
from src.memory.long_term import LongTermMemory
from src.utils.config_loader import get_main_config, ensure_data_dir
from src.utils.logger import get_logger

logger = get_logger("memory")


class MemoryManager:
    """Orchestrates the multi-tier memory system."""

    def __init__(self):
        config = get_main_config()
        memory_config = config.get("memory", {})

        self.db_path = memory_config.get("database_path", "data/memory.db")
        self.short_term = ShortTermMemory(
            max_turns=memory_config.get("short_term", {}).get("max_turns", 10)
        )
        self.mid_term = MidTermMemory(self.db_path)
        self.long_term = LongTermMemory(self.db_path)

        self._summary_interval = memory_config.get("mid_term", {}).get(
            "summary_interval_hours", 4
        )

    async def initialize(self):
        """Initialize database and load any persistent state."""
        ensure_data_dir()
        await self.mid_term.initialize()
        await self.long_term.initialize()
        logger.info("Memory system initialized")

    async def shutdown(self):
        """Save state and close connections."""
        # Generate final session summary before shutdown
        if self.short_term.messages:
            await self._generate_session_summary()
        await self.mid_term.close()
        await self.long_term.close()
        logger.info("Memory system shut down")

    def add_message(self, role: str, content: str):
        """Add a message to short-term memory.

        Args:
            role: "user" or "assistant"
            content: Message content.
        """
        self.short_term.add(role, content)

    def get_conversation_context(self) -> list[dict]:
        """Get the current conversation context for LLM.

        Returns:
            List of message dicts with role and content.
        """
        return self.short_term.get_context()

    async def save_memo(self, content: str, category: str = "general") -> int:
        """Save a memo to long-term memory.

        Args:
            content: Memo content.
            category: Category tag.

        Returns:
            Memo ID.
        """
        return await self.long_term.save_memo(content, category)

    async def search_memos(self, query: str) -> list[dict]:
        """Search memos in long-term memory.

        Args:
            query: Search keyword.

        Returns:
            List of matching memos.
        """
        return await self.long_term.search_memos(query)

    async def save_reminder(
        self, content: str, remind_at: datetime, repeat: str = "none"
    ) -> int:
        """Save a reminder.

        Args:
            content: Reminder content.
            remind_at: When to trigger the reminder.
            repeat: Repeat pattern ("none", "daily", "weekly").

        Returns:
            Reminder ID.
        """
        return await self.long_term.save_reminder(content, remind_at, repeat)

    async def get_pending_reminders(self) -> list[dict]:
        """Get reminders that are due."""
        return await self.long_term.get_pending_reminders()

    async def get_daily_summary(self, date: Optional[str] = None) -> Optional[str]:
        """Get the daily summary for a given date.

        Args:
            date: Date string (YYYY-MM-DD). None for today.
        """
        return await self.mid_term.get_summary(date)

    async def _generate_session_summary(self):
        """Generate and save a summary of the current session."""
        messages = self.short_term.get_context()
        if not messages:
            return

        # Format conversation for summary
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        # Save raw session data (LLM summarization can be done separately)
        await self.mid_term.save_session(
            messages=messages,
            summary=f"Session with {len(messages)} messages",
        )
        logger.info("Session summary saved")

    async def get_context_for_prompt(self) -> str:
        """Get a combined context string for the LLM system prompt.

        Combines relevant mid-term and long-term context.
        """
        parts = []

        # Today's summary
        today_summary = await self.get_daily_summary()
        if today_summary:
            parts.append(f"[今日のまとめ]\n{today_summary}")

        # Recent memos
        recent_memos = await self.long_term.get_recent_memos(5)
        if recent_memos:
            memo_text = "\n".join(
                f"- [{m['category']}] {m['content']} ({m['created_at']})"
                for m in recent_memos
            )
            parts.append(f"[最近のメモ]\n{memo_text}")

        # Pending reminders
        reminders = await self.get_pending_reminders()
        if reminders:
            reminder_text = "\n".join(
                f"- {r['content']} (期限: {r['remind_at']})" for r in reminders
            )
            parts.append(f"[リマインダー]\n{reminder_text}")

        return "\n\n".join(parts) if parts else ""

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
from src.memory.preference_extractor import PreferenceExtractor
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

        # Preference extractor (initialized lazily after LLM router is available)
        self._preference_extractor: Optional[PreferenceExtractor] = None

    async def initialize(self):
        """Initialize database and load any persistent state."""
        ensure_data_dir()
        await self.mid_term.initialize()
        await self.long_term.initialize()
        # Cleanup expired data and compress on startup
        await self.long_term.cleanup_expired_preferences()
        await self.long_term.cleanup_expired_events()
        await self.long_term.compress_old_preferences()
        logger.info("Memory system initialized")

    def init_preference_extractor(self, llm_router):
        """Initialize the preference extractor with an LLM router.

        Called after both memory and LLM are ready (from pipeline init).

        Args:
            llm_router: LLMRouter instance.
        """
        self._preference_extractor = PreferenceExtractor(
            self.long_term, llm_router
        )
        logger.info("Preference extractor initialized")

    async def shutdown(self):
        """Save state and close connections."""
        # Extract preferences from final conversation before shutdown
        if self._preference_extractor and self.short_term.messages:
            messages = self.short_term.get_context()
            try:
                await self._preference_extractor.force_extract(messages)
            except Exception as e:
                logger.error(f"Final preference extraction failed: {e}")

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

    async def maybe_extract_preferences(self, request_id: str = ""):
        """Trigger preference extraction if enough conversation has happened.

        Called after each assistant response. The extractor decides internally
        whether to actually run (based on turn count thresholds).

        Args:
            request_id: Current request ID for source tracking.
        """
        if not self._preference_extractor:
            return
        messages = self.short_term.get_context()
        # Run in background to not block the response pipeline
        import asyncio
        asyncio.create_task(
            self._preference_extractor.maybe_extract(messages, request_id)
        )

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

        Combines relevant mid-term and long-term context, including
        learned user preferences.
        """
        parts = []

        # Learned preferences (most important — affects response style)
        preferences = await self.long_term.get_all_active_preferences()
        if preferences:
            pref_lines = []
            for p in preferences:
                confidence_mark = "★" if p["confidence"] == "explicit" else "☆"
                expires_hint = ""
                if p.get("expires_at"):
                    expires_hint = f" (〜{p['expires_at'][:10]}まで)"
                pref_lines.append(
                    f"- {confidence_mark} {p['key']}: {p['value']}{expires_hint}"
                )
            parts.append(
                f"[ユーザーの好み・習慣]\n"
                f"★=本人が言った ☆=推測\n"
                + "\n".join(pref_lines)
            )

        # Event memories (key decisions, important events)
        events = await self.long_term.get_recent_events(5)
        if events:
            event_lines = []
            for e in events:
                importance_mark = "❗" if e["importance"] == "high" else "📌"
                expires_hint = ""
                if e.get("expires_at"):
                    expires_hint = f" (〜{e['expires_at'][:10]}まで)"
                event_lines.append(
                    f"- {importance_mark} [{e['event_type']}] {e['summary']}{expires_hint} ({e['created_at'][:10]})"
                )
            parts.append(f"[重要なイベント・決定]\n" + "\n".join(event_lines))

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

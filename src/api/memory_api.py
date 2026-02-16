"""Memory CRUD API for frontend settings UI.

Provides WebSocket command handlers for reading, creating, updating,
and deleting memos, reminders, and conversation history.
"""

from datetime import datetime
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("memory_api")


class MemoryAPI:
    """Frontend-facing memory CRUD operations via WebSocket."""

    def __init__(self, memory_manager):
        """Initialize with a reference to the live MemoryManager.

        Args:
            memory_manager: The MemoryManager instance from the pipeline.
        """
        self.memory = memory_manager

    # ------------------------------------------------------------------
    # Memos
    # ------------------------------------------------------------------

    async def handle_get_memos(self, query: str = "", limit: int = 20) -> dict:
        """Get memos, optionally filtered by search query.

        Args:
            query: Search keyword (empty = get recent).
            limit: Max results.

        Returns:
            Dict with "memos" list.
        """
        try:
            if query:
                memos = await self.memory.search_memos(query)
            else:
                memos = await self.memory.long_term.get_recent_memos(limit)
            return {"success": True, "memos": memos}
        except Exception as e:
            logger.error(f"Failed to get memos: {e}")
            return {"success": False, "memos": [], "error": str(e)}

    async def handle_save_memo(self, content: str,
                                category: str = "general") -> dict:
        """Create a new memo.

        Args:
            content: Memo text.
            category: Category tag.

        Returns:
            Dict with success and memo_id.
        """
        if not content or not content.strip():
            return {"success": False, "error": "Memo content cannot be empty"}

        try:
            memo_id = await self.memory.save_memo(content.strip(), category)
            logger.info(f"Memo created: #{memo_id}")
            return {"success": True, "memo_id": memo_id}
        except Exception as e:
            logger.error(f"Failed to save memo: {e}")
            return {"success": False, "error": str(e)}

    async def handle_update_memo(self, memo_id: int, content: str,
                                  category: Optional[str] = None) -> dict:
        """Update an existing memo's content.

        Args:
            memo_id: ID of the memo to update.
            content: New content text.
            category: New category (optional).

        Returns:
            Dict with success status.
        """
        if not memo_id:
            return {"success": False, "error": "memo_id is required"}
        if not content or not content.strip():
            return {"success": False, "error": "Memo content cannot be empty"}

        try:
            await self.memory.long_term.update_memo(
                memo_id, content.strip(), category
            )
            logger.info(f"Memo updated: #{memo_id}")
            return {"success": True, "memo_id": memo_id}
        except Exception as e:
            logger.error(f"Failed to update memo #{memo_id}: {e}")
            return {"success": False, "error": str(e)}

    async def handle_archive_memo(self, memo_id: int) -> dict:
        """Archive (soft-delete) a memo.

        Args:
            memo_id: ID of the memo to archive.

        Returns:
            Dict with success status.
        """
        if not memo_id:
            return {"success": False, "error": "memo_id is required"}

        try:
            await self.memory.long_term.archive_memo(memo_id)
            logger.info(f"Memo archived: #{memo_id}")
            return {"success": True, "memo_id": memo_id}
        except Exception as e:
            logger.error(f"Failed to archive memo #{memo_id}: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    async def handle_get_reminders(self,
                                    include_completed: bool = False) -> dict:
        """Get reminders (pending + upcoming, optionally completed).

        Args:
            include_completed: Whether to include completed reminders.

        Returns:
            Dict with "pending", "upcoming", and optionally "completed" lists.
        """
        try:
            pending = await self.memory.long_term.get_pending_reminders()
            upcoming = await self.memory.long_term.get_upcoming_reminders(20)
            result = {
                "success": True,
                "pending": pending,
                "upcoming": upcoming,
            }
            if include_completed:
                completed = await self.memory.long_term.get_completed_reminders(20)
                result["completed"] = completed
            return result
        except Exception as e:
            logger.error(f"Failed to get reminders: {e}")
            return {
                "success": False,
                "pending": [], "upcoming": [],
                "error": str(e),
            }

    async def handle_save_reminder(self, content: str, remind_at: str,
                                    repeat: str = "none") -> dict:
        """Create a new reminder.

        Args:
            content: Reminder text.
            remind_at: ISO datetime string.
            repeat: Repeat pattern ("none", "daily", "weekly").

        Returns:
            Dict with success and reminder_id.
        """
        if not content or not content.strip():
            return {"success": False, "error": "Reminder content cannot be empty"}
        if not remind_at:
            return {"success": False, "error": "remind_at is required"}

        try:
            dt = datetime.fromisoformat(remind_at)
            reminder_id = await self.memory.save_reminder(
                content.strip(), dt, repeat
            )
            logger.info(f"Reminder created: #{reminder_id} at {remind_at}")
            return {"success": True, "reminder_id": reminder_id}
        except ValueError as e:
            return {"success": False, "error": f"Invalid datetime format: {e}"}
        except Exception as e:
            logger.error(f"Failed to save reminder: {e}")
            return {"success": False, "error": str(e)}

    async def handle_complete_reminder(self, reminder_id: int) -> dict:
        """Mark a reminder as completed.

        Args:
            reminder_id: ID of the reminder.

        Returns:
            Dict with success status.
        """
        if not reminder_id:
            return {"success": False, "error": "reminder_id is required"}

        try:
            await self.memory.long_term.complete_reminder(reminder_id)
            logger.info(f"Reminder completed: #{reminder_id}")
            return {"success": True, "reminder_id": reminder_id}
        except Exception as e:
            logger.error(f"Failed to complete reminder #{reminder_id}: {e}")
            return {"success": False, "error": str(e)}

    async def handle_delete_reminder(self, reminder_id: int) -> dict:
        """Permanently delete a reminder.

        Args:
            reminder_id: ID of the reminder.

        Returns:
            Dict with success status.
        """
        if not reminder_id:
            return {"success": False, "error": "reminder_id is required"}

        try:
            await self.memory.long_term.delete_reminder(reminder_id)
            logger.info(f"Reminder deleted: #{reminder_id}")
            return {"success": True, "reminder_id": reminder_id}
        except Exception as e:
            logger.error(f"Failed to delete reminder #{reminder_id}: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Conversation History
    # ------------------------------------------------------------------

    async def handle_get_history(self, limit: int = 10) -> dict:
        """Get conversation history.

        Args:
            limit: Max past sessions to return.

        Returns:
            Dict with "current_session" and "past_sessions".
        """
        try:
            current = self.memory.short_term.get_full_history()
            sessions = await self.memory.mid_term.get_recent_sessions(limit)
            return {
                "success": True,
                "current_session": current,
                "past_sessions": sessions,
            }
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {
                "success": False,
                "current_session": [],
                "past_sessions": [],
                "error": str(e),
            }

    async def handle_clear_short_term(self) -> dict:
        """Clear the current conversation context (short-term memory).

        Returns:
            Dict with success status.
        """
        try:
            self.memory.short_term.clear()
            logger.info("Short-term memory cleared")
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to clear short-term memory: {e}")
            return {"success": False, "error": str(e)}

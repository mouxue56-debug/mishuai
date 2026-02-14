"""Memo and Reminder tools.

- add_memo: Save voice memos to persistent storage
- set_reminder: Set time-based reminders with voice notification
"""

import re
from datetime import datetime, timedelta
from typing import Any

from src.mcp.tool_registry import MCPTool
from src.memory.memory_manager import MemoryManager
from src.utils.logger import get_logger

logger = get_logger("tool.memo")

# Shared memory manager instance (initialized in pipeline)
_memory: MemoryManager | None = None


def set_memory_manager(memory: MemoryManager):
    """Set the shared memory manager instance."""
    global _memory
    _memory = memory


def _get_memory() -> MemoryManager:
    """Get the memory manager, initializing if needed."""
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


class AddMemoTool(MCPTool):
    """Save a voice memo to persistent storage."""

    name = "add_memo"
    description = (
        "メモを保存します。「〇〇を覚えておいて」「メモ：〇〇」などの時に使います。"
        "保存したメモは後から検索できます。"
    )
    parameters = {
        "content": {
            "type": "string",
            "description": "メモの内容",
            "required": True,
        },
        "category": {
            "type": "string",
            "enum": ["general", "customer", "schedule", "idea", "important"],
            "description": "メモのカテゴリ（デフォルト: general）",
        },
    }

    async def execute(self, content: str, category: str = "general", **kwargs) -> Any:
        """Save a memo.

        Args:
            content: Memo text.
            category: Category for organization.

        Returns:
            Confirmation message.
        """
        memory = _get_memory()
        memo_id = await memory.save_memo(content, category)
        logger.info(f"Memo saved: #{memo_id} [{category}] {content[:50]}")
        return f"メモ#{memo_id}を保存しました：「{content}」（カテゴリ: {category}）"


class SearchMemoTool(MCPTool):
    """Search saved memos."""

    name = "search_memos"
    description = "保存されたメモを検索します。「さっきのメモ」「〇〇のメモを探して」等の時に使います。"
    parameters = {
        "query": {
            "type": "string",
            "description": "検索キーワード。空の場合は最近のメモを表示",
        },
    }

    async def execute(self, query: str = "", **kwargs) -> Any:
        """Search memos.

        Args:
            query: Search keyword. Empty returns recent memos.

        Returns:
            Search results as formatted string.
        """
        memory = _get_memory()

        if query:
            memos = await memory.search_memos(query)
        else:
            memos = await memory.long_term.get_recent_memos(5)

        if not memos:
            return "メモは見つかりませんでした。"

        lines = []
        for m in memos:
            lines.append(f"#{m['id']} [{m['category']}] {m['content']} ({m['created_at']})")

        return f"メモ検索結果 ({len(memos)}件):\n" + "\n".join(lines)


class SetReminderTool(MCPTool):
    """Set a time-based reminder."""

    name = "set_reminder"
    description = (
        "リマインダーを設定します。"
        "「〇時に〇〇を教えて」「30分後にリマインド」「明日の9時に〇〇」等の時に使います。"
    )
    parameters = {
        "content": {
            "type": "string",
            "description": "リマインダーの内容",
            "required": True,
        },
        "time_spec": {
            "type": "string",
            "description": (
                "時間指定。以下の形式に対応:\n"
                "- '15:30' → 今日の15:30\n"
                "- '30m' → 30分後\n"
                "- '2h' → 2時間後\n"
                "- '2025-03-01 09:00' → 指定日時\n"
                "- 'tomorrow 09:00' → 明日の9:00"
            ),
            "required": True,
        },
        "repeat": {
            "type": "string",
            "enum": ["none", "daily", "weekly"],
            "description": "繰り返しパターン（デフォルト: none）",
        },
    }

    async def execute(
        self, content: str, time_spec: str, repeat: str = "none", **kwargs
    ) -> Any:
        """Set a reminder.

        Args:
            content: Reminder text.
            time_spec: Time specification string.
            repeat: Repeat pattern.

        Returns:
            Confirmation message.
        """
        remind_at = self._parse_time(time_spec)
        if remind_at is None:
            return f"時間の解析ができませんでした: 「{time_spec}」。例: '15:30', '30m', '2h'"

        memory = _get_memory()
        reminder_id = await memory.save_reminder(content, remind_at, repeat)

        time_str = remind_at.strftime("%Y-%m-%d %H:%M")
        repeat_str = {"none": "", "daily": "（毎日繰り返し）", "weekly": "（毎週繰り返し）"}
        logger.info(f"Reminder set: #{reminder_id} at {time_str}")

        return (
            f"リマインダー#{reminder_id}を設定しました！\n"
            f"内容: {content}\n"
            f"時間: {time_str} {repeat_str.get(repeat, '')}"
        )

    @staticmethod
    def _parse_time(time_spec: str) -> datetime | None:
        """Parse a time specification string into a datetime.

        Supports:
        - "15:30" → today at 15:30
        - "30m" → 30 minutes from now
        - "2h" → 2 hours from now
        - "2025-03-01 09:00" → specific datetime
        - "tomorrow 09:00" → tomorrow at 09:00
        """
        now = datetime.now()

        # Relative: "30m", "2h"
        match = re.match(r"^(\d+)\s*m(?:in(?:utes?)?)?$", time_spec.strip(), re.I)
        if match:
            return now + timedelta(minutes=int(match.group(1)))

        match = re.match(r"^(\d+)\s*h(?:ours?)?$", time_spec.strip(), re.I)
        if match:
            return now + timedelta(hours=int(match.group(1)))

        # Tomorrow: "tomorrow 09:00"
        match = re.match(r"^tomorrow\s+(\d{1,2}):(\d{2})$", time_spec.strip(), re.I)
        if match:
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(
                hour=int(match.group(1)),
                minute=int(match.group(2)),
                second=0,
                microsecond=0,
            )

        # Time only: "15:30"
        match = re.match(r"^(\d{1,2}):(\d{2})$", time_spec.strip())
        if match:
            target = now.replace(
                hour=int(match.group(1)),
                minute=int(match.group(2)),
                second=0,
                microsecond=0,
            )
            # If time already passed today, set for tomorrow
            if target <= now:
                target += timedelta(days=1)
            return target

        # Full datetime: "2025-03-01 09:00"
        for fmt in ["%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%m/%d %H:%M"]:
            try:
                parsed = datetime.strptime(time_spec.strip(), fmt)
                if parsed.year == 1900:  # No year specified
                    parsed = parsed.replace(year=now.year)
                return parsed
            except ValueError:
                continue

        return None

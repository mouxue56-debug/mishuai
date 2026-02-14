"""Schedule management tool.

Checks and manages daily schedules, appointments, and events.
Currently uses SQLite storage. Can be extended to sync with
Google Calendar or Notion calendar.
"""

from datetime import datetime, date, timedelta
from typing import Any, Optional

import aiosqlite

from src.mcp.tool_registry import MCPTool
from src.utils.config_loader import ensure_data_dir
from src.utils.logger import get_logger

logger = get_logger("tool.schedule")


class CheckScheduleTool(MCPTool):
    """Check today's schedule and upcoming events."""

    name = "check_schedule"
    description = (
        "今日の予定を確認します。"
        "「今日の予定は？」「次の予約は？」「明日のスケジュール」等の時に使います。"
    )
    parameters = {
        "target_date": {
            "type": "string",
            "description": "確認する日付（YYYY-MM-DD形式）。省略時は今日",
        },
        "category": {
            "type": "string",
            "enum": ["all", "visit", "delivery", "meeting", "other"],
            "description": "イベントカテゴリで絞り込み（デフォルト: all）",
        },
    }

    def __init__(self):
        self._db_path = str(ensure_data_dir() / "memory.db")
        self._initialized = False

    async def _ensure_table(self):
        """Create the schedule table if it doesn't exist."""
        if self._initialized:
            return

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT DEFAULT 'other',
                    customer_name TEXT,
                    is_completed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            await db.commit()

        self._initialized = True

    async def execute(
        self, target_date: str = "", category: str = "all", **kwargs
    ) -> Any:
        """Check schedule for a given date.

        Args:
            target_date: Date to check (YYYY-MM-DD). Empty for today.
            category: Filter by category.

        Returns:
            Formatted schedule.
        """
        await self._ensure_table()

        if not target_date:
            target_date = date.today().isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            if category == "all":
                query = """SELECT time, title, description, category, customer_name
                           FROM schedule WHERE date = ? AND is_completed = 0
                           ORDER BY time"""
                params = (target_date,)
            else:
                query = """SELECT time, title, description, category, customer_name
                           FROM schedule WHERE date = ? AND category = ? AND is_completed = 0
                           ORDER BY time"""
                params = (target_date, category)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            # Check if it's today
            if target_date == date.today().isoformat():
                return "今日の予定はありません。ゆっくりできますね〜"
            else:
                return f"{target_date} の予定はありません。"

        lines = [f"📅 {target_date} のスケジュール ({len(rows)}件):"]
        for row in rows:
            time_str = row[0] or "終日"
            title = row[1]
            desc = row[2] or ""
            cat = row[3] or ""
            customer = row[4] or ""

            entry = f"  {time_str} - {title}"
            if customer:
                entry += f" ({customer}様)"
            if desc:
                entry += f"\n    → {desc}"
            lines.append(entry)

        return "\n".join(lines)


class AddScheduleTool(MCPTool):
    """Add an event to the schedule."""

    name = "add_schedule"
    description = (
        "予定を追加します。"
        "「明日15時に田中さんの見学」「3/1に子猫のお引き渡し」等の時に使います。"
    )
    parameters = {
        "title": {
            "type": "string",
            "description": "予定のタイトル",
            "required": True,
        },
        "date": {
            "type": "string",
            "description": "日付（YYYY-MM-DD形式）",
            "required": True,
        },
        "time": {
            "type": "string",
            "description": "時間（HH:MM形式）。終日イベントの場合は省略",
        },
        "category": {
            "type": "string",
            "enum": ["visit", "delivery", "meeting", "other"],
            "description": "カテゴリ（見学/お引き渡し/打ち合わせ/その他）",
        },
        "customer_name": {
            "type": "string",
            "description": "関連する顧客名",
        },
        "description": {
            "type": "string",
            "description": "詳細メモ",
        },
    }

    def __init__(self):
        self._db_path = str(ensure_data_dir() / "memory.db")

    async def execute(
        self,
        title: str,
        date: str,
        time: str = "",
        category: str = "other",
        customer_name: str = "",
        description: str = "",
        **kwargs,
    ) -> Any:
        """Add a schedule event.

        Args:
            title: Event title.
            date: Date (YYYY-MM-DD).
            time: Time (HH:MM), optional.
            category: Event category.
            customer_name: Related customer.
            description: Additional details.

        Returns:
            Confirmation message.
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT DEFAULT 'other',
                    customer_name TEXT,
                    is_completed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            cursor = await db.execute(
                """INSERT INTO schedule (date, time, title, description, category, customer_name)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (date, time or None, title, description, category, customer_name),
            )
            await db.commit()
            event_id = cursor.lastrowid

        time_str = f" {time}" if time else ""
        customer_str = f"（{customer_name}様）" if customer_name else ""
        logger.info(f"Schedule added: #{event_id} {date}{time_str} {title}")

        return f"予定を追加しました！\n📅 {date}{time_str} - {title}{customer_str}"

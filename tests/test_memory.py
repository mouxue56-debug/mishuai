"""Tests for the memory system."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.memory.short_term import ShortTermMemory
from src.memory.mid_term import MidTermMemory
from src.memory.long_term import LongTermMemory


class TestShortTermMemory:
    """Tests for in-memory conversation context."""

    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_turns=5)
        mem.add("user", "こんにちは")
        mem.add("assistant", "こんにちは！")

        context = mem.get_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "こんにちは"

    def test_max_turns(self):
        mem = ShortTermMemory(max_turns=3)
        for i in range(10):
            mem.add("user", f"message {i}")
            mem.add("assistant", f"reply {i}")

        context = mem.get_context()
        assert len(context) == 6  # max_turns * 2

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add("user", "test")
        mem.clear()
        assert len(mem.get_context()) == 0

    def test_turn_count(self):
        mem = ShortTermMemory()
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        mem.add("user", "how are you")
        assert mem.turn_count == 2

    def test_get_last_message(self):
        mem = ShortTermMemory()
        mem.add("user", "first")
        mem.add("assistant", "second")
        mem.add("user", "third")

        assert mem.get_last_message()["content"] == "third"
        assert mem.get_last_message("assistant")["content"] == "second"

    def test_full_history_has_timestamps(self):
        mem = ShortTermMemory()
        mem.add("user", "test")
        history = mem.get_full_history()
        assert "timestamp" in history[0]


class TestLongTermMemory:
    """Tests for SQLite persistent memory."""

    @pytest.fixture
    async def memory(self):
        """Create a temp database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mem = LongTermMemory(db_path)
        await mem.initialize()
        yield mem
        await mem.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_save_and_search_memo(self, memory):
        memo_id = await memory.save_memo("明日10時に田中さんに電話", "customer")
        assert memo_id > 0

        results = await memory.search_memos("田中")
        assert len(results) == 1
        assert "田中" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recent_memos(self, memory):
        await memory.save_memo("memo 1", "general")
        await memory.save_memo("memo 2", "customer")
        await memory.save_memo("memo 3", "important")

        recent = await memory.get_recent_memos(2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["content"] == "memo 3"

    @pytest.mark.asyncio
    async def test_archive_memo(self, memory):
        memo_id = await memory.save_memo("to archive")
        await memory.archive_memo(memo_id)

        results = await memory.search_memos("archive")
        assert len(results) == 0  # Archived memos not in search

    @pytest.mark.asyncio
    async def test_save_and_get_reminder(self, memory):
        remind_at = datetime.now() - timedelta(minutes=5)  # Already due
        rid = await memory.save_reminder("テスト通知", remind_at)
        assert rid > 0

        pending = await memory.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["content"] == "テスト通知"

    @pytest.mark.asyncio
    async def test_complete_reminder(self, memory):
        remind_at = datetime.now() - timedelta(minutes=5)
        rid = await memory.save_reminder("done", remind_at)
        await memory.complete_reminder(rid)

        pending = await memory.get_pending_reminders()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_upcoming_reminders(self, memory):
        future = datetime.now() + timedelta(hours=2)
        await memory.save_reminder("future task", future)

        upcoming = await memory.get_upcoming_reminders()
        assert len(upcoming) == 1

    @pytest.mark.asyncio
    async def test_customer_notes(self, memory):
        await memory.add_customer_note("田中太郎", "サイベリアン希望、ブラウンタビー")
        notes = await memory.get_customer_notes("田中")
        assert len(notes) == 1

    @pytest.mark.asyncio
    async def test_preferences(self, memory):
        await memory.set_preference("greeting_style", "casual", "Will設定")
        value = await memory.get_preference("greeting_style")
        assert value == "casual"

        # Update
        await memory.set_preference("greeting_style", "formal")
        value = await memory.get_preference("greeting_style")
        assert value == "formal"


class TestMidTermMemory:
    """Tests for session summaries."""

    @pytest.fixture
    async def memory(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        mem = MidTermMemory(db_path)
        await mem.initialize()
        yield mem
        await mem.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_save_session(self, memory):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        await memory.save_session(messages, "Test session")

        recent = await memory.get_recent_sessions(1)
        assert len(recent) == 1
        assert recent[0]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_daily_summary(self, memory):
        await memory.save_daily_summary(
            "Today was productive",
            task_count=5,
            inquiry_count=3,
        )

        summary = await memory.get_summary()
        assert summary == "Today was productive"

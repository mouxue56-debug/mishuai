"""Tests for MCP tools."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.mcp.tool_registry import MCPTool, ToolRegistry
from src.mcp.tools.memo_reminder import AddMemoTool, SetReminderTool, SearchMemoTool
from src.mcp.tools.cattery_knowledge import CatteryKnowledgeTool
from src.mcp.tools.sns_caption import SNSCaptionTool
from src.mcp.tools.line_draft import LineDraftTool
from src.memory.memory_manager import MemoryManager


class TestAddMemoTool:
    """Tests for the memo tool."""

    @pytest.fixture
    async def setup_memory(self):
        """Set up memory for memo tools."""
        from src.mcp.tools.memo_reminder import set_memory_manager
        from src.utils.config_loader import ensure_data_dir

        ensure_data_dir()
        mem = MemoryManager()
        await mem.initialize()
        set_memory_manager(mem)
        yield mem
        await mem.shutdown()

    @pytest.mark.asyncio
    async def test_add_memo(self, setup_memory):
        tool = AddMemoTool()
        result = await tool.execute(content="テスト用メモ", category="general")
        assert "保存しました" in result
        assert "テスト用メモ" in result

    @pytest.mark.asyncio
    async def test_search_memo(self, setup_memory):
        add_tool = AddMemoTool()
        await add_tool.execute(content="田中さんに連絡", category="customer")

        search_tool = SearchMemoTool()
        result = await search_tool.execute(query="田中")
        assert "田中" in result


class TestSetReminderTool:
    """Tests for time parsing in the reminder tool."""

    def test_parse_minutes(self):
        result = SetReminderTool._parse_time("30m")
        assert result is not None
        assert result > datetime.now()
        diff = (result - datetime.now()).total_seconds()
        assert 1700 < diff < 1900  # ~30 minutes

    def test_parse_hours(self):
        result = SetReminderTool._parse_time("2h")
        assert result is not None
        diff = (result - datetime.now()).total_seconds()
        assert 7000 < diff < 7400  # ~2 hours

    def test_parse_time_only(self):
        result = SetReminderTool._parse_time("15:30")
        assert result is not None
        assert result.hour == 15
        assert result.minute == 30

    def test_parse_tomorrow(self):
        result = SetReminderTool._parse_time("tomorrow 09:00")
        assert result is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert result.date() == tomorrow.date()
        assert result.hour == 9

    def test_parse_full_datetime(self):
        result = SetReminderTool._parse_time("2026-03-01 09:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 3

    def test_parse_invalid(self):
        result = SetReminderTool._parse_time("invalid")
        assert result is None


class TestCatteryKnowledgeTool:
    """Tests for knowledge base search."""

    @pytest.mark.asyncio
    async def test_search_allergy(self):
        tool = CatteryKnowledgeTool()
        result = await tool.execute(query="アレルギー")
        assert "Fel d1" in result or "アレルゲン" in result or "アレルギー" in result

    @pytest.mark.asyncio
    async def test_search_breed(self):
        tool = CatteryKnowledgeTool()
        result = await tool.execute(query="体重", category="breed")
        assert "kg" in result

    @pytest.mark.asyncio
    async def test_search_faq(self):
        tool = CatteryKnowledgeTool()
        result = await tool.execute(query="見学", category="faq")
        assert "予約" in result

    @pytest.mark.asyncio
    async def test_search_not_found(self):
        tool = CatteryKnowledgeTool()
        result = await tool.execute(query="量子コンピュータ")
        assert "見つかりません" in result


class TestSNSCaptionTool:
    """Tests for SNS caption generation."""

    @pytest.mark.asyncio
    async def test_generate_instagram_caption(self):
        tool = SNSCaptionTool()
        result = await tool.execute(
            description="子猫がボールで遊んでいる",
            platform="instagram",
        )
        assert "子猫" in result
        assert "#" in result  # Has hashtags

    @pytest.mark.asyncio
    async def test_generate_chinese_caption(self):
        tool = SNSCaptionTool()
        result = await tool.execute(
            description="Kitten playing",
            language="chinese",
        )
        assert "西伯利亚" in result or "猫" in result


class TestLineDraftTool:
    """Tests for LINE draft generation."""

    @pytest.mark.asyncio
    async def test_generate_polite_draft(self):
        tool = LineDraftTool()
        result = await tool.execute(
            recipient="田中",
            context="見学予約の確認",
            tone="polite",
        )
        assert "田中" in result
        assert "福楽キャッテリー" in result

    @pytest.mark.asyncio
    async def test_generate_chinese_draft(self):
        tool = LineDraftTool()
        result = await tool.execute(
            recipient="王",
            context="诊疗预约",
            tone="formal",
            language="chinese",
        )
        assert "王" in result


class TestToolRegistry:
    """Tests for the tool registry."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self):
        registry = ToolRegistry()
        await registry.initialize()
        tools = registry.list_tools()
        assert len(tools) > 0
        assert "add_memo" in tools
        assert "cattery_knowledge" in tools

    @pytest.mark.asyncio
    async def test_get_schemas(self):
        registry = ToolRegistry()
        await registry.initialize()
        schemas = registry.get_all_schemas()
        assert len(schemas) > 0
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

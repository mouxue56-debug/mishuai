"""Tests for the core dialogue pipeline."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.llm_router import LLMRouter, LLMMessage, LLMResponse, LLMTask, LLMMode
from src.core.emotion_analyzer import EmotionAnalyzer
from src.core.interrupt_handler import InterruptHandler, PipelineState


class TestEmotionAnalyzer:
    """Tests for emotion analysis and Live2D mapping."""

    def test_update_emotion(self):
        analyzer = EmotionAnalyzer()
        result = analyzer.update_emotion("happy")
        assert result["expression"] == "smile"
        assert analyzer.current_emotion == "happy"

    def test_unknown_emotion_fallback(self):
        analyzer = EmotionAnalyzer()
        result = analyzer.update_emotion("unknown_emotion")
        assert analyzer.current_emotion == "neutral"

    def test_emotion_history(self):
        analyzer = EmotionAnalyzer()
        analyzer.update_emotion("happy")
        analyzer.update_emotion("thinking")
        analyzer.update_emotion("surprised")
        assert len(analyzer.emotion_history) == 3

    def test_get_emotion_for_frontend(self):
        analyzer = EmotionAnalyzer()
        analyzer.update_emotion("excited")
        data = analyzer.get_emotion_for_frontend()
        assert data["type"] == "emotion"
        assert data["emotion"] == "excited"

    def test_time_based_emotion(self):
        analyzer = EmotionAnalyzer()
        # This is time-dependent, just verify it returns valid result
        result = analyzer.get_time_based_emotion()
        assert result is None or result == "sleepy"


class TestInterruptHandler:
    """Tests for interruption handling."""

    def test_initial_state(self):
        handler = InterruptHandler()
        assert handler.state == PipelineState.IDLE
        assert not handler.is_speaking
        assert not handler.is_interrupted

    def test_set_state(self):
        handler = InterruptHandler()
        handler.set_state(PipelineState.LISTENING)
        assert handler.state == PipelineState.LISTENING

    @pytest.mark.asyncio
    async def test_interrupt_while_speaking(self):
        handler = InterruptHandler()
        handler.set_state(PipelineState.SPEAKING)
        assert handler.is_speaking

        await handler.interrupt()
        assert handler.state == PipelineState.INTERRUPTED
        assert handler.is_interrupted

    @pytest.mark.asyncio
    async def test_interrupt_while_idle_does_nothing(self):
        handler = InterruptHandler()
        handler.set_state(PipelineState.IDLE)
        await handler.interrupt()
        # Should not change to interrupted when not speaking
        assert handler.state == PipelineState.IDLE

    def test_clear_interrupt(self):
        handler = InterruptHandler()
        handler._interrupt_event.set()
        handler.clear_interrupt()
        assert not handler.is_interrupted

    def test_should_stop(self):
        handler = InterruptHandler()
        assert not handler.should_stop()
        handler._interrupt_event.set()
        assert handler.should_stop()


class TestLLMRouter:
    """Tests for LLM routing logic."""

    def test_set_mode(self):
        router = LLMRouter()
        router.set_mode(LLMMode.BUDGET)
        assert router.active_mode == LLMMode.BUDGET

    def test_parse_json_response(self):
        router = LLMRouter()
        response = router._parse_response(
            '{"text": "こんにちは！", "emotion": "happy"}',
            "test-model"
        )
        assert response.text == "こんにちは！"
        assert response.emotion == "happy"

    def test_parse_json_with_code_block(self):
        router = LLMRouter()
        response = router._parse_response(
            '```json\n{"text": "テスト", "emotion": "neutral"}\n```',
            "test-model"
        )
        assert response.text == "テスト"
        assert response.emotion == "neutral"

    def test_parse_plain_text_fallback(self):
        router = LLMRouter()
        response = router._parse_response(
            "This is just plain text without JSON",
            "test-model"
        )
        assert response.text == "This is just plain text without JSON"
        assert response.emotion == "neutral"

    def test_parse_json_with_tool_calls(self):
        router = LLMRouter()
        response = router._parse_response(
            '{"text": "調べますね", "emotion": "thinking", "tool_calls": [{"name": "search_customers", "arguments": {"query": "田中"}}]}',
            "test-model"
        )
        assert response.text == "調べますね"
        assert response.emotion == "thinking"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search_customers"

    def test_parse_embedded_json(self):
        router = LLMRouter()
        response = router._parse_response(
            'Sure! Here is my response: {"text": "はい！", "emotion": "happy"} That was fun.',
            "test-model"
        )
        assert response.text == "はい！"
        assert response.emotion == "happy"

    def test_build_system_prompt(self):
        router = LLMRouter()
        prompt = router._build_system_prompt()
        assert "ミケ" in prompt
        assert "福楽キャッテリー" in prompt
        assert "emotion" in prompt

    def test_system_prompt_with_speaker_profile(self):
        router = LLMRouter()
        profile = {"name": "ウィルさん", "style": "casual"}
        prompt = router._build_system_prompt(profile)
        assert "ウィルさん" in prompt
        assert "カジュアル" in prompt

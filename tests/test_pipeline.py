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

    # --- v0.2: Enhanced JSON parsing tests ---

    def test_parse_tool_call_singular(self):
        """tool_call (singular) should be normalized to tool_calls array."""
        router = LLMRouter()
        response = router._parse_response(
            '{"text": "検索しますね", "emotion": "thinking", "tool_call": {"name": "search_customers", "arguments": {"query": "佐藤"}}}',
            "test-model"
        )
        assert response.text == "検索しますね"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search_customers"

    def test_parse_emotion_inline_tag(self):
        """[EMOTION:tag] inline format should be parsed."""
        router = LLMRouter()
        response = router._parse_response(
            "[EMOTION:happy] はい、わかりました！",
            "test-model"
        )
        assert response.text == "はい、わかりました！"
        assert response.emotion == "happy"

    def test_parse_emotion_inline_fullwidth_colon(self):
        """[EMOTION：tag] with fullwidth colon should also work."""
        router = LLMRouter()
        response = router._parse_response(
            "[EMOTION：excited] すごいですね！",
            "test-model"
        )
        assert response.text == "すごいですね！"
        assert response.emotion == "excited"

    def test_parse_empty_response(self):
        """Empty string should return empty response."""
        router = LLMRouter()
        response = router._parse_response("", "test-model")
        assert response.text == ""
        assert response.model == "test-model"

    def test_parse_json_missing_text_uses_raw(self):
        """JSON without 'text' key should use raw text in embedded extraction."""
        router = LLMRouter()
        # Pure JSON without text key falls through to _try_parse_json which
        # returns the dict, then _build_response_from_dict uses raw_text fallback
        response = router._parse_response(
            '{"emotion": "happy", "reply": "こんにちは"}',
            "test-model"
        )
        # Since "text" key is missing, it uses raw_text
        assert response.emotion == "happy"

    def test_parse_code_block_without_json_tag(self):
        """Code block without json language tag should still parse."""
        router = LLMRouter()
        response = router._parse_response(
            '```\n{"text": "テスト中", "emotion": "thinking"}\n```',
            "test-model"
        )
        assert response.text == "テスト中"
        assert response.emotion == "thinking"


class TestToolPermission:
    """Tests for speaker-based tool permission checks in pipeline."""

    def test_owner_has_all_permissions(self):
        """Owner profile with 'all' permission should allow everything."""
        from src.core.pipeline import DialoguePipeline
        pipeline = DialoguePipeline()
        profile = {"name": "Will", "permissions": ["all"]}
        assert pipeline._check_tool_permission("search_customers", profile) is True
        assert pipeline._check_tool_permission("add_memo", profile) is True
        assert pipeline._check_tool_permission("set_reminder", profile) is True

    def test_limited_permissions(self):
        """Profile with specific permissions should only allow listed tools."""
        from src.core.pipeline import DialoguePipeline
        pipeline = DialoguePipeline()
        profile = {"name": "Staff", "permissions": ["cattery_knowledge", "check_schedule"]}
        assert pipeline._check_tool_permission("cattery_knowledge", profile) is True
        assert pipeline._check_tool_permission("check_schedule", profile) is True
        assert pipeline._check_tool_permission("add_memo", profile) is False
        assert pipeline._check_tool_permission("search_customers", profile) is False

    def test_unknown_speaker_only_knowledge(self):
        """Unknown speaker (no profile) should only access cattery_knowledge."""
        from src.core.pipeline import DialoguePipeline
        pipeline = DialoguePipeline()
        assert pipeline._check_tool_permission("cattery_knowledge", None) is True
        assert pipeline._check_tool_permission("add_memo", None) is False
        assert pipeline._check_tool_permission("search_customers", None) is False

    def test_empty_permissions_list(self):
        """Profile with empty permissions list should deny everything."""
        from src.core.pipeline import DialoguePipeline
        pipeline = DialoguePipeline()
        profile = {"name": "Guest", "permissions": []}
        assert pipeline._check_tool_permission("cattery_knowledge", profile) is False
        assert pipeline._check_tool_permission("add_memo", profile) is False


class TestConfigValidator:
    """Tests for Pydantic-based config validation."""

    def test_valid_config(self):
        from src.utils.config_validator import validate_config
        config = {
            "app": {"name": "test"},
            "llm": {"active_mode": "balanced"},
            "websocket": {"host": "0.0.0.0", "port": 8765},
        }
        result = validate_config(config)
        assert result.llm.active_mode == "balanced"
        assert result.websocket.port == 8765

    def test_invalid_llm_mode(self):
        from src.utils.config_validator import validate_config
        from pydantic import ValidationError
        config = {"llm": {"active_mode": "ultra"}}
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_valid_persona(self):
        from src.utils.config_validator import validate_persona
        persona = {
            "character": {"name": "ミケ", "role": "AI秘書"},
            "emotions": {"happy": {}},
        }
        result = validate_persona(persona)
        assert result.character.name == "ミケ"

    def test_empty_config_uses_defaults(self):
        from src.utils.config_validator import validate_config
        result = validate_config({})
        assert result.llm is None
        assert result.websocket is None

    def test_validate_configs_integration(self):
        """Test that actual config files validate successfully."""
        from src.utils.config_loader import validate_configs
        valid, errors = validate_configs()
        assert valid, f"Config validation errors: {errors}"

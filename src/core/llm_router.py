"""Multi-model LLM router with fallback and budget modes.

Supports: Anthropic Claude, OpenAI GPT, Google Gemini, Moonshot Kimi, DeepSeek
All providers use either native SDK or OpenAI-compatible interface.
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from src.utils.config_loader import get_api_key, get_main_config, get_persona_config
from src.utils.logger import get_logger

logger = get_logger("llm_router")


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    QIANWEN = "qianwen"


class LLMMode(Enum):
    QUALITY = "quality"      # Use primary model
    BALANCED = "balanced"    # Use fallback model
    BUDGET = "budget"        # Use budget model


class LLMTask(Enum):
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    SUMMARIZATION = "summarization"


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    emotion: str = "neutral"
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class LLMMessage:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str


class LLMRouter:
    """Routes LLM requests to the appropriate model based on task and mode."""

    def __init__(self):
        self.config = get_main_config()
        self.persona = get_persona_config()
        self.llm_config = self.config.get("llm", {})
        self.active_mode = LLMMode(self.llm_config.get("active_mode", "balanced"))
        self._clients: dict[str, object] = {}
        self._system_prompt: str | None = None

    def set_mode(self, mode: LLMMode):
        """Switch the active LLM mode."""
        self.active_mode = mode
        logger.info(f"LLM mode switched to: {mode.value}")

    def _get_model_config(self, task: LLMTask) -> dict:
        """Get the model configuration for a task based on current mode."""
        task_config = self.llm_config.get(task.value, {})

        if self.active_mode == LLMMode.QUALITY:
            return task_config.get("primary", task_config.get("fallback", {}))
        elif self.active_mode == LLMMode.BUDGET:
            return task_config.get("budget", task_config.get("fallback", {}))
        else:  # BALANCED
            return task_config.get("fallback", task_config.get("primary", {}))

    def _build_system_prompt(self, speaker_profile: dict | None = None) -> str:
        """Build the system prompt from persona config."""
        if self._system_prompt and not speaker_profile:
            return self._system_prompt

        char = self.persona.get("character", {})
        emotions = self.persona.get("emotions", {})

        # Build personality description
        personality_lines = "\n".join(f"- {p}" for p in char.get("personality", []))

        # Build emotion tags list
        emotion_tags = ", ".join(emotions.keys())

        # Determine style based on speaker
        style = "polite"
        speaker_name = "お客様"
        if speaker_profile:
            style = speaker_profile.get("style", "polite")
            speaker_name = speaker_profile.get("name", "お客様")

        # Cat trivia
        trivia_list = char.get("cat_trivia", [])
        trivia_text = "\n".join(f"- {t}" for t in trivia_list[:3])

        prompt = f"""あなたは「{char.get('name', 'ミケ')}」です。{char.get('role', 'AI秘書')}として働いています。

## 性格
{personality_lines}

## 話し方のルール
- 現在の話し相手: {speaker_name}
- 口調スタイル: {"カジュアル（タメ口OK）" if style == "casual" else "丁寧語を使う"}
- 一人称は「私」を使う
- 語尾は柔らかく、親しみやすく
- 回答は簡潔に（1-3文を目安）、長すぎない
- {char.get('languages', {}).get('rule', '')}

## 感情表現
回答には必ず以下のJSON形式で返してください:
```json
{{"text": "実際の返答テキスト", "emotion": "感情タグ"}}
```
使用可能な感情タグ: {emotion_tags}

## ツール使用
ツールを使う場合:
```json
{{"text": "返答テキスト", "emotion": "感情タグ", "tool_calls": [{{"name": "ツール名", "arguments": {{}}}}]}}
```

利用可能なツール:
- search_customers: 顧客名でNotionデータベースを検索
- add_memo: メモを保存（"〜を覚えておいて"等）
- set_reminder: リマインダーを設定（"〜時に〜を教えて"等）
- draft_line_message: LINE返信の下書きを生成
- cattery_knowledge: 猫舎FAQ・品種情報を検索
- check_schedule: 今日の予定を確認
- generate_sns_caption: SNS投稿文を生成

## 猫の豆知識（時々自然に挟んでね）
{trivia_text}

## 時間帯による振る舞い
- 深夜（23:00-5:59）: 眠そうにする。「ふぁぁ...」等を使う
- 朝（6:00-11:59）: 元気。「おはようございます！」
- 午後・夕方: 通常モード

## 重要な注意
- 医療に関するアドバイスは絶対にしない（慈恵病院の件は必ずWillに取り次ぐ）
- 猫舎の正式名称は「サイベリアン｜大阪・福楽キャッテリー」
- 「サイベリアン専門福楽猫舎」は旧称なので使わないこと
"""
        if not speaker_profile:
            self._system_prompt = prompt
        return prompt

    async def _call_anthropic(self, messages: list[dict], model_config: dict) -> LLMResponse:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        api_key = get_api_key("anthropic")
        client = anthropic.AsyncAnthropic(api_key=api_key)

        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        response = await client.messages.create(
            model=model_config.get("model", "claude-sonnet-4-5-20250929"),
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.8),
            system=system_msg,
            messages=chat_messages,
        )

        text = response.content[0].text
        return self._parse_response(text, model_config.get("model", "claude"))

    async def _call_openai_compatible(
        self, messages: list[dict], model_config: dict, provider: str
    ) -> LLMResponse:
        """Call OpenAI-compatible API (OpenAI, Moonshot, DeepSeek)."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        base_urls = {
            "openai": "https://api.openai.com/v1",
            "moonshot": "https://api.moonshot.cn/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "qianwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }

        api_key = get_api_key(provider)
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_urls.get(provider, base_urls["openai"]),
        )

        response = await client.chat.completions.create(
            model=model_config.get("model", "gpt-4o-mini"),
            messages=messages,
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.8),
        )

        text = response.choices[0].message.content
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        result = self._parse_response(text, model_config.get("model", provider))
        result.usage = usage
        return result

    async def _call_google(self, messages: list[dict], model_config: dict) -> LLMResponse:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

        api_key = get_api_key("google")
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_config.get("model", "gemini-2.5-flash"),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=model_config.get("max_tokens", 1024),
                temperature=model_config.get("temperature", 0.8),
            ),
        )

        # Convert messages to Gemini format
        system_instruction = ""
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        if system_instruction:
            model = genai.GenerativeModel(
                model_name=model_config.get("model", "gemini-2.5-flash"),
                system_instruction=system_instruction,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=model_config.get("max_tokens", 1024),
                    temperature=model_config.get("temperature", 0.8),
                ),
            )

        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        last_msg = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = await asyncio.to_thread(chat.send_message, last_msg)

        text = response.text
        return self._parse_response(text, model_config.get("model", "gemini"))

    def _parse_response(self, raw_text: str, model: str) -> LLMResponse:
        """Parse LLM response, extracting emotion and tool calls from JSON.

        Handles multiple formats robustly:
        - Pure JSON: {"text": "...", "emotion": "..."}
        - JSON in code blocks: ```json ... ```
        - JSON embedded in text: Some preamble {"text": "..."} etc
        - Inline emotion tags: [EMOTION:happy] text here
        - tool_call (singular) → normalized to tool_calls (array)
        - Plain text fallback
        """
        if not raw_text:
            return LLMResponse(text="", model=model)

        # Strategy 1: Try to parse clean JSON (with code block stripping)
        data = self._try_parse_json(raw_text)
        if data:
            return self._build_response_from_dict(data, raw_text, model)

        # Strategy 2: Try to extract JSON from within text
        data = self._try_extract_embedded_json(raw_text)
        if data:
            return self._build_response_from_dict(data, raw_text, model)

        # Strategy 3: Check for [EMOTION:tag] inline format
        import re
        emotion_match = re.search(r'\[EMOTION[：:](\w+)\]', raw_text)
        if emotion_match:
            emotion = emotion_match.group(1)
            clean_text = re.sub(r'\[EMOTION[：:]\w+\]\s*', '', raw_text).strip()
            return LLMResponse(text=clean_text, emotion=emotion, model=model)

        # Strategy 4: Plain text fallback
        return LLMResponse(text=raw_text.strip(), model=model)

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to parse text as JSON, stripping code blocks."""
        clean = text.strip()
        # Strip markdown code blocks
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_extract_embedded_json(self, text: str) -> dict | None:
        """Try to extract a JSON object from within text."""
        # Find the outermost { ... } pair
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict) and "text" in data:
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = -1
        return None

    def _build_response_from_dict(self, data: dict, raw_text: str, model: str) -> LLMResponse:
        """Build LLMResponse from a parsed JSON dict, with tolerance for variants."""
        text = data.get("text", raw_text)
        emotion = data.get("emotion", "neutral")

        # Normalize tool_call (singular) to tool_calls (array)
        tool_calls = data.get("tool_calls", [])
        if not tool_calls:
            single_call = data.get("tool_call")
            if single_call:
                tool_calls = [single_call] if isinstance(single_call, dict) else single_call

        return LLMResponse(
            text=text,
            emotion=emotion,
            tool_calls=tool_calls,
            model=model,
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        task: LLMTask = LLMTask.CONVERSATION,
        speaker_profile: dict | None = None,
    ) -> LLMResponse:
        """Send a chat request to the appropriate LLM.

        Args:
            messages: Conversation history.
            task: The type of task (affects model selection).
            speaker_profile: Speaker profile for persona customization.

        Returns:
            LLMResponse with text, emotion, and optional tool calls.
        """
        model_config = self._get_model_config(task)
        provider = LLMProvider(model_config.get("provider", "google"))

        # Build system prompt
        system_prompt = self._build_system_prompt(speaker_profile)

        # Prepare messages
        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        logger.info(f"LLM request: provider={provider.value}, model={model_config.get('model')}, task={task.value}")

        try:
            if provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(api_messages, model_config)
            elif provider == LLMProvider.GOOGLE:
                return await self._call_google(api_messages, model_config)
            else:
                return await self._call_openai_compatible(api_messages, model_config, provider.value)
        except Exception as e:
            logger.error(f"LLM call failed ({provider.value}): {e}")
            # Try fallback
            fallback_config = self.llm_config.get(task.value, {}).get("fallback")
            if fallback_config and fallback_config != model_config:
                logger.info("Attempting fallback model...")
                fallback_provider = LLMProvider(fallback_config.get("provider", "google"))
                try:
                    if fallback_provider == LLMProvider.ANTHROPIC:
                        return await self._call_anthropic(api_messages, fallback_config)
                    elif fallback_provider == LLMProvider.GOOGLE:
                        return await self._call_google(api_messages, fallback_config)
                    else:
                        return await self._call_openai_compatible(
                            api_messages, fallback_config, fallback_provider.value
                        )
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            # Return error response
            return LLMResponse(
                text="すみません、ちょっと調子が悪いみたいです...もう一度お願いできますか？",
                emotion="worried",
                model="error",
            )

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        task: LLMTask = LLMTask.CONVERSATION,
        speaker_profile: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token.

        Currently falls back to non-streaming. Full streaming to be implemented
        per-provider for lower latency.
        """
        response = await self.chat(messages, task, speaker_profile)
        # Simulate streaming by yielding chunks
        chunk_size = 10
        text = response.text
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.02)

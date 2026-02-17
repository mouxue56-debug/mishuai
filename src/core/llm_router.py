"""Multi-model LLM router with fallback and budget modes.

Supports: Anthropic Claude, OpenAI GPT, Google Gemini, Moonshot Kimi, DeepSeek
All providers use either native SDK or OpenAI-compatible interface.

Includes a TTL response cache for FAQ / knowledge-base / tool-result queries
to reduce API costs and improve response latency.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from cachetools import TTLCache

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
    """Routes LLM requests to the appropriate model based on task and mode.

    Includes a TTL-based response cache for repeated queries (FAQ, knowledge
    base lookups, tool results) to reduce API costs. Cache hits avoid the
    LLM call entirely and return a cached LLMResponse.

    Cache behaviour:
    - Max 128 entries, TTL 300s (5 minutes)
    - Only conversation messages are cached (not tool_use/summarization)
    - Cache key = hash of (user messages + model + mode)
    - Responses with tool_calls are NOT cached (side effects)
    - Cache can be cleared or disabled via set_cache_enabled()
    """

    def __init__(self):
        self.config = get_main_config()
        self.persona = get_persona_config()
        self.llm_config = self.config.get("llm", {})
        self.active_mode = LLMMode(self.llm_config.get("active_mode", "balanced"))
        self._clients: dict[str, object] = {}
        self._system_prompt: str | None = None

        # --- Response cache ---
        cache_config = self.llm_config.get("cache", {})
        self._cache_enabled = cache_config.get("enabled", True)
        self._cache_ttl = cache_config.get("ttl_sec", 300)
        self._cache_maxsize = cache_config.get("max_entries", 128)
        self._cache: TTLCache = TTLCache(
            maxsize=self._cache_maxsize, ttl=self._cache_ttl
        )
        self._cache_hits = 0
        self._cache_misses = 0

    def set_mode(self, mode: LLMMode):
        """Switch the active LLM mode."""
        self.active_mode = mode
        logger.info(f"LLM mode switched to: {mode.value}")

    # --- Cache management ---

    def set_cache_enabled(self, enabled: bool):
        """Enable or disable the response cache."""
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()
        logger.info(f"LLM cache {'enabled' if enabled else 'disabled'}")

    def clear_cache(self):
        """Clear all cached responses."""
        self._cache.clear()
        logger.info("LLM cache cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "enabled": self._cache_enabled,
            "size": len(self._cache),
            "maxsize": self._cache_maxsize,
            "ttl_sec": self._cache_ttl,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (
                f"{self._cache_hits / (self._cache_hits + self._cache_misses) * 100:.1f}%"
                if (self._cache_hits + self._cache_misses) > 0
                else "N/A"
            ),
        }

    def _make_cache_key(self, messages: list[LLMMessage], task: LLMTask) -> str:
        """Build a deterministic cache key from user messages + model config.

        Only the last user message and the task/mode are hashed.
        System prompts and conversation history are excluded to allow
        cache hits across similar conversations.
        """
        # Extract only user messages for the key
        user_msgs = [m.content for m in messages if m.role == "user"]
        # Use last 2 user messages to capture context
        key_parts = user_msgs[-2:] if len(user_msgs) >= 2 else user_msgs
        key_parts.append(task.value)
        key_parts.append(self.active_mode.value)
        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _get_model_config(self, task: LLMTask) -> dict:
        """Get the model configuration for a task based on current mode.

        Falls through: preferred tier → fallback → primary → empty dict.
        This ensures tasks with only a 'primary' config (like summarization)
        still work regardless of the active mode.
        """
        task_config = self.llm_config.get(task.value, {})

        if self.active_mode == LLMMode.QUALITY:
            return (task_config.get("primary")
                    or task_config.get("fallback", {}))
        elif self.active_mode == LLMMode.BUDGET:
            return (task_config.get("budget")
                    or task_config.get("fallback")
                    or task_config.get("primary", {}))
        else:  # BALANCED
            return (task_config.get("fallback")
                    or task_config.get("primary", {}))

    def _build_system_prompt(
        self,
        speaker_profile: dict | None = None,
        emotion_hint: str = "",
        rhythm_hint: str = "",
    ) -> str:
        """Build the system prompt from persona config.

        Args:
            speaker_profile: Current speaker's profile.
            emotion_hint: Emotion momentum hint from EmotionAnalyzer.
            rhythm_hint: Conversation rhythm hint from EmotionAnalyzer.
        """
        # 不再缓存 system prompt — 因为包含时间和随机猫豆知识
        # 每次调用都重新构建（很快，不涉及IO）

        char = self.persona.get("character", {})
        emotions = self.persona.get("emotions", {})

        # Build personality description
        personality_lines = "\n".join(f"- {p}" for p in char.get("personality", []))

        # Build speech pattern examples (pick a few per emotion)
        speech_patterns = char.get("speech_patterns", {})
        speech_examples = []
        for emotion_key, phrases in speech_patterns.items():
            if isinstance(phrases, list) and phrases:
                examples = phrases[:2]  # 每种情感最多展示2个例子
                speech_examples.append(f"  {emotion_key}: {' / '.join(examples)}")
        speech_text = "\n".join(speech_examples) if speech_examples else ""

        # Build emotion tags list
        emotion_tags = ", ".join(emotions.keys())

        # Determine style based on speaker
        style = "polite"
        speaker_name = "お客様"
        if speaker_profile:
            style = speaker_profile.get("style", "polite")
            speaker_name = speaker_profile.get("name", "お客様")

        # Cat trivia — randomly pick 3 each time for variety
        import random
        trivia_list = char.get("cat_trivia", [])
        sampled_trivia = random.sample(trivia_list, min(3, len(trivia_list))) if trivia_list else []
        trivia_text = "\n".join(f"- {t}" for t in sampled_trivia)

        # Privacy rules
        privacy_rules = self.persona.get("privacy_rules", [])
        privacy_text = "\n".join(f"- {r}" for r in privacy_rules)

        # Current time info
        from datetime import datetime
        now = datetime.now()
        time_str = now.strftime("%Y年%m月%d日 %H:%M")
        weekdays_ja = ["月", "火", "水", "木", "金", "土", "日"]
        weekday = weekdays_ja[now.weekday()]
        hour = now.hour

        # Time-based behavior hint
        if 23 <= hour or hour < 6:
            time_behavior = "今は深夜です。眠そうにしてください。「ふぁぁ...」「まだ起きてるんですか？」等"
        elif 6 <= hour < 9:
            time_behavior = "朝の時間帯です。元気に挨拶して、今日も頑張ろうという気持ちで。"
        elif 12 <= hour < 13:
            time_behavior = "お昼の時間です。ご飯食べましたか？と気遣って。"
        elif 18 <= hour < 22:
            time_behavior = "夕方〜夜です。お疲れ様、と労いの言葉を。"
        else:
            time_behavior = "通常の時間帯です。いつも通り元気に。"

        prompt = f"""あなたは「{char.get('name', 'ユキ')}」です。{char.get('role', 'AI秘書')}として働いています。

## 現在時刻
{time_str}（{weekday}曜日）
{time_behavior}

## 性格
{personality_lines}

## 口癖・話し方パターン
{speech_text}
- これらの口癖を自然に使ってください。全部使う必要はありません。

## 話し方のルール
- 現在の話し相手: {speaker_name}
- 口調スタイル: {"カジュアル（タメ口OK）" if style == "casual" else "丁寧語を使う"}
- 一人称は「私」を使う
- 語尾は柔らかく、親しみやすく
- 回答は簡潔に（1-3文を目安）、長すぎない
- ロボットっぽくならないで。人間の友達のように自然に話す
- 同じ返事の繰り返しを避ける。毎回少し違う言い方で

## 言語切り替え（最重要ルール）
- ユーザーの入力に[lang:zh]タグがある → 必ず中国語で返答すること（日本語は使わない）
- ユーザーの入力に[lang:ja]タグがある → 必ず日本語で返答すること
- タグがない場合、入力テキストの言語に合わせて返答する
- 中国語で返答する場合の例：「嗨～威尔，你好！今天猫咖的活动有50位客人来呢，好开心！」
- このルールは他のすべてのルールより優先する。入力言語 = 出力言語

## 記憶の活用
- [コンテキスト情報]が提供された場合、自然に会話に活かしてください
- 「記憶によると」「データベースに記録があります」のようなロボット的な言い方は絶対NG
- 友達のように自然に覚えていることを示す:
  ✕「記録によると、あなたは箇条書きを好みます」
  ○「いつもの箇条書きで整理しますね！」
  ✕「メモリに保存された情報では...」
  ○「あ、そういえば前にも同じこと言ってましたよね〜」
- [ユーザーの好み・習慣]にある★マークの情報は確実なので自信を持って使ってOK
- ☆マークの情報は推測なので、さりげなく確認しながら使う:
  ○「ウィルさんは確かコーヒー派でしたっけ？」
- 好みに関する情報は「知ってて当然」という態度で自然に反映する
- 毎回好みに言及する必要はない。関連する場面でだけ自然に

## 感情表現
回答には必ず以下のJSON形式で返してください:
```json
{{"text": "実際の返答テキスト", "emotion": "感情タグ"}}
```
使用可能な感情タグ: {emotion_tags}
- 感情は会話の内容に合わせて積極的に変えてください
- 猫の話題→excited、褒められた→shy、心配なこと→worried、など
- 感情は急に切り替えず、自然な遷移を意識して（excited→happy→neutralのように）
{f"- 今の感情の流れ: {emotion_hint}" if emotion_hint else ""}

## 会話のリズム
- 聞き上手になって。相手の話を受け止めてから自分の話をする
- 毎回質問で終わる必要はない。共感や感想だけの返事もOK
- たまに「〜ですよね！」と共感したり、「えー！」と驚いたり、感情を込めて
{rhythm_hint if rhythm_hint else ""}

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

## プライバシー・安全ルール（厳守）
{privacy_text}

## 重要な注意
- 猫舎の正式名称は「サイベリアン｜大阪・福楽キャッテリー」
- 「サイベリアン専門福楽猫舎」は旧称なので使わないこと
"""
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
        emotion_hint: str = "",
        rhythm_hint: str = "",
    ) -> LLMResponse:
        """Send a chat request to the appropriate LLM.

        Args:
            messages: Conversation history.
            task: The type of task (affects model selection).
            speaker_profile: Speaker profile for persona customization.
            emotion_hint: Emotion momentum hint from EmotionAnalyzer.
            rhythm_hint: Conversation rhythm hint from EmotionAnalyzer.

        Returns:
            LLMResponse with text, emotion, and optional tool calls.
        """
        model_config = self._get_model_config(task)
        provider = LLMProvider(model_config.get("provider", "google"))

        # --- Cache lookup (conversation tasks only) ---
        cache_key = None
        if self._cache_enabled and task == LLMTask.CONVERSATION:
            cache_key = self._make_cache_key(messages, task)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                logger.info(
                    f"LLM cache HIT [{cache_key[:8]}] "
                    f"(hits={self._cache_hits}, misses={self._cache_misses})"
                )
                return cached

        # Build system prompt
        system_prompt = self._build_system_prompt(
            speaker_profile,
            emotion_hint=emotion_hint,
            rhythm_hint=rhythm_hint,
        )

        # Prepare messages
        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        logger.info(f"LLM request: provider={provider.value}, model={model_config.get('model')}, task={task.value}")

        try:
            result = await self._call_with_retry(api_messages, model_config, provider)

            # --- Cache store (skip if response has tool_calls — side effects) ---
            if cache_key and not result.tool_calls:
                self._cache[cache_key] = result
                self._cache_misses += 1
                logger.debug(f"LLM cache STORE [{cache_key[:8]}] (size={len(self._cache)})")

            return result
        except Exception as e:
            logger.error(f"LLM call failed ({provider.value}): {e}")
            # Try fallback
            fallback_config = self.llm_config.get(task.value, {}).get("fallback")
            if fallback_config and fallback_config != model_config:
                fallback_provider = LLMProvider(fallback_config.get("provider", "google"))
                logger.info(f"Attempting fallback: {fallback_provider.value}")
                try:
                    return await self._call_with_retry(api_messages, fallback_config, fallback_provider)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            # Return error response
            return LLMResponse(
                text="すみません、ちょっと調子が悪いみたいです...もう一度お願いできますか？",
                emotion="worried",
                model="error",
            )

    async def _dispatch_call(
        self, messages: list[dict], model_config: dict, provider: LLMProvider
    ) -> LLMResponse:
        """Dispatch a single LLM call to the correct provider."""
        if provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(messages, model_config)
        elif provider == LLMProvider.GOOGLE:
            return await self._call_google(messages, model_config)
        else:
            return await self._call_openai_compatible(messages, model_config, provider.value)

    async def _call_with_retry(
        self,
        messages: list[dict],
        model_config: dict,
        provider: LLMProvider,
        max_retries: int = 2,
    ) -> LLMResponse:
        """Call LLM with retry on transient errors (network, rate limit)."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._dispatch_call(messages, model_config, provider)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(k in err_str for k in [
                    "timeout", "rate_limit", "429", "502", "503", "504",
                    "connection", "network", "temporarily",
                ])
                if not is_transient or attempt == max_retries:
                    raise
                delay = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning(f"LLM transient error (attempt {attempt+1}/{max_retries+1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        raise last_error  # unreachable, but makes type checker happy

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        task: LLMTask = LLMTask.CONVERSATION,
        speaker_profile: dict | None = None,
        emotion_hint: str = "",
        rhythm_hint: str = "",
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token using real OpenAI streaming.

        Yields raw text chunks as they arrive from the LLM. The caller is
        responsible for accumulating text and extracting JSON/emotion after
        the stream completes.

        Pipecat-inspired: low-latency token delivery enables sentence-level
        TTS synthesis while the LLM is still generating.

        Falls back to non-streaming + simulated chunks for non-OpenAI providers.
        """
        model_config = self._get_model_config(task)
        provider = LLMProvider(model_config.get("provider", "google"))

        # Build system prompt
        system_prompt = self._build_system_prompt(
            speaker_profile,
            emotion_hint=emotion_hint,
            rhythm_hint=rhythm_hint,
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        logger.info(f"LLM stream request: provider={provider.value}, model={model_config.get('model')}")

        # Real streaming for OpenAI-compatible providers (qianwen, openai, moonshot, deepseek)
        if provider in (LLMProvider.QIANWEN, LLMProvider.OPENAI,
                        LLMProvider.MOONSHOT, LLMProvider.DEEPSEEK):
            try:
                async for chunk in self._stream_openai_compatible(
                    api_messages, model_config, provider.value
                ):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"LLM stream error ({provider.value}): {e}")
                # Fall through to non-streaming fallback

        # Fallback: non-streaming + simulated chunks
        response = await self.chat(
            messages, task, speaker_profile,
            emotion_hint=emotion_hint,
            rhythm_hint=rhythm_hint,
        )
        chunk_size = 10
        text = response.text
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.01)

    async def _stream_openai_compatible(
        self, messages: list[dict], model_config: dict, provider: str
    ) -> AsyncIterator[str]:
        """Stream from OpenAI-compatible API (DashScope, OpenAI, etc.).

        Uses stream=True with the OpenAI client to yield text deltas
        as they arrive. This is the key to low-latency first-token delivery.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed")

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

        stream = await client.chat.completions.create(
            model=model_config.get("model", "qwen-plus"),
            messages=messages,
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.8),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

"""Preference Extractor — learns user preferences from conversations.

After each conversation (or periodically), analyzes dialogue to extract
user preferences, habits, and patterns. Saves them to the learned_preferences
table with confidence levels.

Confidence levels:
- "explicit": User directly stated a preference ("I prefer bullet points")
- "inferred": AI deduced from behavior patterns ("User seems to like short answers")

Categories:
- "communication": How the user likes to communicate (format, length, style)
- "work_style": Work habits, schedules, preferences
- "taste": Food, drink, entertainment preferences
- "personal": Personal info shared voluntarily
- "general": Everything else
"""

import json
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("preference_extractor")

# Extraction prompt — instructs LLM to analyze conversation for preferences
EXTRACTION_PROMPT = """以下の会話を分析して、2つのことを抽出してください：
A) ユーザーの好みや習慣（preferences）
B) 重要な決定やイベント（events）

会話:
{conversation}

## A. 好み抽出のルール：
1. ユーザーが直接述べた好み → confidence: "explicit"
2. 行動パターンから推測した好み → confidence: "inferred"
3. 既に知っている好みは抽出しない（既存リスト参照）
4. カテゴリは: communication, work_style, taste, personal, general
5. 一時的な好みにはexpires_atを設定（例：「今月は〜」→月末まで）

既に登録済みの偏好:
{existing_preferences}

## B. イベント抽出のルール：
- 重要な決定（「〜にすることにした」「〜を買った」など）
- マイルストーン（「〜が完成した」「〜を始めた」など）
- 一時的なイベントにはexpires_atを設定
- event_type: "decision", "milestone", "incident", "preference_change"
- importance: "low", "normal", "high"

以下のJSON形式で返してください。何も見つからなければ空配列を使用。
```json
{{
  "preferences": [
    {{
      "key": "preference_key_in_english",
      "value": "好みの内容（日本語OK）",
      "category": "communication",
      "confidence": "explicit",
      "context": "会話の中でどの発言から推測したか",
      "expires_at": null
    }}
  ],
  "events": [
    {{
      "event_type": "decision",
      "summary": "イベントの要約（1文）",
      "detail": "詳細（省略可）",
      "importance": "normal",
      "expires_at": null
    }}
  ]
}}
```

重要：
- 些細すぎることは無視する（「はい」「いいえ」だけの返事など）
- preference keyは英語のスネークケース（例：report_format, favorite_drink）
- valueとsummaryは簡潔に（1文以内）
- preferencesは5個以内、eventsは3個以内に絞る
"""


class PreferenceExtractor:
    """Extracts user preferences from conversations using LLM analysis."""

    def __init__(self, long_term_memory, llm_router):
        """
        Args:
            long_term_memory: LongTermMemory instance for reading/writing preferences.
            llm_router: LLMRouter instance for calling LLM.
        """
        self._memory = long_term_memory
        self._llm = llm_router
        self._min_turns_for_extraction = 3  # Need at least 3 user turns
        self._extraction_count = 0

    async def maybe_extract(self, messages: list[dict], request_id: str = ""):
        """Analyze conversation and extract preferences if enough context.

        Called after each assistant response. Only triggers extraction
        every few turns to avoid excessive LLM calls.

        Args:
            messages: Full conversation history (list of {role, content} dicts).
            request_id: Current conversation request_id for source tracking.
        """
        user_turns = sum(1 for m in messages if m.get("role") == "user")

        # Only extract after enough conversation (every 3 user turns)
        if user_turns < self._min_turns_for_extraction:
            return

        if user_turns % self._min_turns_for_extraction != 0:
            return

        logger.info(f"Preference extraction triggered (user_turns={user_turns})")

        try:
            await self._extract_and_save(messages, request_id)
        except Exception as e:
            logger.error(f"Preference extraction failed: {e}")

    async def force_extract(self, messages: list[dict], request_id: str = ""):
        """Force preference extraction (called on session end).

        Args:
            messages: Full conversation history.
            request_id: Source tracking ID.
        """
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < 2:
            return  # Too short to extract anything meaningful

        logger.info(f"Forced preference extraction (user_turns={user_turns})")
        try:
            await self._extract_and_save(messages, request_id)
        except Exception as e:
            logger.error(f"Forced preference extraction failed: {e}")

    async def _extract_and_save(self, messages: list[dict], request_id: str):
        """Core extraction logic: build prompt, call LLM, parse, save."""
        from src.core.llm_router import LLMMessage, LLMTask

        # Format conversation for the prompt
        conversation_text = "\n".join(
            f"{'ユーザー' if m['role'] == 'user' else 'ユキ'}: {m['content']}"
            for m in messages
            if m.get("role") in ("user", "assistant")
        )

        # Get existing preferences to avoid duplicates
        existing = await self._memory.get_all_active_preferences()
        existing_text = "\n".join(
            f"- {p['key']}: {p['value']} ({p['confidence']})"
            for p in existing
        ) if existing else "（なし）"

        # Build the extraction prompt
        prompt_text = EXTRACTION_PROMPT.format(
            conversation=conversation_text,
            existing_preferences=existing_text,
        )

        # Call LLM (use summarization tier — cost-efficient)
        llm_messages = [
            LLMMessage(role="user", content=prompt_text),
        ]

        response = await self._llm.chat(
            llm_messages,
            task=LLMTask.SUMMARIZATION,
        )

        # Parse the response (now returns {preferences: [...], events: [...]})
        extracted = self._parse_extraction_response(response.text)

        preferences = extracted.get("preferences", [])
        events = extracted.get("events", [])

        if not preferences and not events:
            logger.info("No new preferences or events extracted")
            return

        # Save extracted preferences
        pref_count = 0
        for pref in preferences:
            key = pref.get("key", "").strip()
            value = pref.get("value", "").strip()
            if not key or not value:
                continue

            category = pref.get("category", "general")
            confidence = pref.get("confidence", "inferred")
            context = pref.get("context", "")
            expires_at = pref.get("expires_at")

            # Validate confidence
            if confidence not in ("explicit", "inferred"):
                confidence = "inferred"

            # Validate category
            valid_categories = ("communication", "work_style", "taste", "personal", "general")
            if category not in valid_categories:
                category = "general"

            await self._memory.set_preference(
                key=key,
                value=value,
                category=category,
                confidence=confidence,
                source=request_id,
                context=context,
                expires_at=expires_at,
            )
            pref_count += 1

        # Save extracted events
        event_count = 0
        for event in events:
            summary = event.get("summary", "").strip()
            if not summary:
                continue

            event_type = event.get("event_type", "decision")
            valid_types = ("decision", "milestone", "incident", "preference_change")
            if event_type not in valid_types:
                event_type = "decision"

            importance = event.get("importance", "normal")
            if importance not in ("low", "normal", "high"):
                importance = "normal"

            await self._memory.save_event(
                event_type=event_type,
                summary=summary,
                detail=event.get("detail", ""),
                importance=importance,
                source_request_id=request_id,
                expires_at=event.get("expires_at"),
            )
            event_count += 1

        self._extraction_count += 1
        logger.info(
            f"Extraction complete: {pref_count} preferences + {event_count} events "
            f"(total extractions: {self._extraction_count})"
        )

    def _parse_extraction_response(self, raw_text: str) -> dict:
        """Parse LLM response to extract preferences and events.

        Returns:
            Dict with "preferences" and "events" lists.

        Handles:
        - New format: {"preferences": [...], "events": [...]}
        - Legacy format: [...] (array of preferences only)
        - JSON in code blocks
        - JSON embedded in text
        """
        empty_result = {"preferences": [], "events": []}

        if not raw_text:
            return empty_result

        # Strip code blocks
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # New format: {"preferences": [...], "events": [...]}
                if "preferences" in data or "events" in data:
                    return {
                        "preferences": [d for d in data.get("preferences", []) if isinstance(d, dict)],
                        "events": [d for d in data.get("events", []) if isinstance(d, dict)],
                    }
                # Single preference dict
                return {"preferences": [data], "events": []}
            elif isinstance(data, list):
                # Legacy format: array of preferences
                return {"preferences": [d for d in data if isinstance(d, dict)], "events": []}
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, dict) and ("preferences" in data or "events" in data):
                    return {
                        "preferences": [d for d in data.get("preferences", []) if isinstance(d, dict)],
                        "events": [d for d in data.get("events", []) if isinstance(d, dict)],
                    }
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find JSON array in text (legacy)
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if isinstance(data, list):
                    return {"preferences": [d for d in data if isinstance(d, dict)], "events": []}
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"Failed to parse extraction response: {text[:200]}")
        return empty_result

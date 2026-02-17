"""Sentence splitter for streaming TTS.

Pipecat-inspired: accumulates LLM tokens and splits at natural sentence
boundaries so TTS can start synthesizing the first sentence while the
LLM is still generating.

Optimized for Japanese and Chinese text with mixed punctuation support.
"""

import re
from typing import Optional


class SentenceSplitter:
    """Accumulates streaming text and yields complete sentences.

    Design (inspired by Pipecat's LLMSentenceAggregator):
    - Accumulates text from LLM token stream
    - Splits at Japanese/Chinese/English sentence boundaries
    - Short sentence merging: combines very short fragments (< min_length)
      to avoid TTS overhead on tiny chunks
    - Handles JSON-wrapped responses: strips ```json wrapper if detected
    """

    # Sentence-ending punctuation for Japanese, Chinese, and English
    SENTENCE_ENDERS = re.compile(
        r'[。！？\!\?\.\.\.…\n]'
        r'|(?<=[^0-9])\.(?:\s|$)'  # English period (not decimal)
    )

    # Finer split points (commas, semicolons) for long sentences
    CLAUSE_ENDERS = re.compile(
        r'[、，；;：:～〜]'
    )

    def __init__(self, min_length: int = 8, max_length: int = 80):
        """
        Args:
            min_length: Minimum characters per sentence (merge shorter fragments).
            max_length: Force-split at this length even without punctuation.
        """
        self.min_length = min_length
        self.max_length = max_length
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        """Push new text and return any complete sentences.

        Args:
            text: New text chunk from LLM stream.

        Returns:
            List of complete sentences (may be empty).
        """
        self._buffer += text
        return self._extract_sentences()

    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer.

        Call this when the LLM stream ends.

        Returns:
            Remaining text, or None if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    def reset(self):
        """Clear the buffer."""
        self._buffer = ""

    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from the buffer."""
        sentences = []

        while True:
            # Look for sentence-ending punctuation
            match = self.SENTENCE_ENDERS.search(self._buffer)

            if match:
                # Split at the punctuation (include it in the sentence)
                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()
                self._buffer = self._buffer[end_pos:].lstrip()

                if sentence:
                    # Merge short fragments with previous sentence
                    if sentences and len(sentence) < self.min_length:
                        sentences[-1] += sentence
                    else:
                        sentences.append(sentence)
            elif len(self._buffer) > self.max_length:
                # Force split at clause boundary or max_length
                clause_match = self.CLAUSE_ENDERS.search(
                    self._buffer[:self.max_length]
                )
                if clause_match:
                    end_pos = clause_match.end()
                    sentence = self._buffer[:end_pos].strip()
                    self._buffer = self._buffer[end_pos:].lstrip()
                else:
                    # Hard split at max_length
                    sentence = self._buffer[:self.max_length].strip()
                    self._buffer = self._buffer[self.max_length:].lstrip()

                if sentence:
                    sentences.append(sentence)
            else:
                break

        return sentences


def extract_json_text_and_emotion(raw_text: str) -> tuple[str, str]:
    """Extract text and emotion from an LLM response that may be JSON-wrapped.

    Handles formats:
    - {"text": "...", "emotion": "..."}
    - ```json {"text": "...", "emotion": "..."} ```
    - Plain text (returns as-is with "neutral")

    Args:
        raw_text: Full accumulated LLM response.

    Returns:
        Tuple of (text, emotion).
    """
    import json

    clean = raw_text.strip()

    # Strip markdown code blocks
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    # Try parsing as JSON
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "text" in data:
            return data.get("text", ""), data.get("emotion", "neutral")
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting embedded JSON
    depth = 0
    start = -1
    for i, ch in enumerate(clean):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    data = json.loads(clean[start:i + 1])
                    if isinstance(data, dict) and "text" in data:
                        return data.get("text", ""), data.get("emotion", "neutral")
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    # Check for inline [EMOTION:tag] format
    emotion_match = re.search(r'\[EMOTION[：:](\w+)\]', clean)
    if emotion_match:
        emotion = emotion_match.group(1)
        text = re.sub(r'\[EMOTION[：:]\w+\]\s*', '', clean).strip()
        return text, emotion

    # Plain text fallback
    return clean, "neutral"

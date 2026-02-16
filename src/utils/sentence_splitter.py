"""Sentence splitter for streaming TTS playback.

Splits Japanese/Chinese/English text into natural speech segments
for per-sentence TTS synthesis and playback.
"""

import re

# Sentence-ending punctuation for Japanese, Chinese, English
_SENTENCE_END = re.compile(r'(?<=[。！？.!?\n])\s*')

# Minimum characters for a standalone segment (avoids tiny fragments)
_MIN_SEGMENT_LEN = 5


def split_sentences(text: str) -> list[str]:
    """Split text into natural speech segments.

    Splits on sentence boundaries (。！？.!?) and merges
    very short segments into the previous one.

    Args:
        text: Full text to split.

    Returns:
        List of text segments suitable for individual TTS synthesis.
    """
    if not text or not text.strip():
        return []

    # Split on sentence boundaries
    raw_segments = _SENTENCE_END.split(text.strip())
    segments = [s.strip() for s in raw_segments if s.strip()]

    if not segments:
        return [text.strip()]

    # Merge short segments into previous
    merged = []
    for seg in segments:
        if merged and len(seg) < _MIN_SEGMENT_LEN:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    return merged

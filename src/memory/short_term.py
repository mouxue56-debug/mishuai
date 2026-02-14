"""Short-term memory - current conversation context.

Holds the most recent conversation turns in memory.
No persistence - cleared when the session ends.
"""

from collections import deque
from datetime import datetime
from typing import Optional


class ShortTermMemory:
    """In-memory conversation context buffer.

    Keeps the most recent N turns of conversation for LLM context.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: deque[dict] = deque(maxlen=max_turns * 2)  # *2 for user+assistant
        self.current_emotion: str = "neutral"
        self.current_speaker: Optional[str] = None
        self.session_start: datetime = datetime.now()

    def add(self, role: str, content: str):
        """Add a message to the conversation buffer.

        Args:
            role: "user" or "assistant"
            content: Message text.
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def get_context(self) -> list[dict]:
        """Get conversation context for LLM input.

        Returns:
            List of message dicts (role, content only - no timestamp for LLM).
        """
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def get_full_history(self) -> list[dict]:
        """Get full message history including timestamps."""
        return list(self.messages)

    def clear(self):
        """Clear the conversation buffer."""
        self.messages.clear()
        self.current_emotion = "neutral"
        self.session_start = datetime.now()

    def get_last_message(self, role: Optional[str] = None) -> Optional[dict]:
        """Get the most recent message, optionally filtered by role."""
        for msg in reversed(self.messages):
            if role is None or msg["role"] == role:
                return msg
        return None

    @property
    def turn_count(self) -> int:
        """Number of user turns in the conversation."""
        return sum(1 for m in self.messages if m["role"] == "user")

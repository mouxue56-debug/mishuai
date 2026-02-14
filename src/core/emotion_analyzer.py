"""Emotion analysis from LLM output and context.

Extracts emotion tags from LLM responses and maps them to Live2D expressions.
"""

from datetime import datetime
from typing import Optional

from src.utils.config_loader import get_persona_config
from src.utils.logger import get_logger

logger = get_logger("emotion")


# Default emotion mapping (emotion tag → Live2D expression ID)
EMOTION_MAP = {
    "neutral": {"expression": "normal", "motion_group": "Idle"},
    "happy": {"expression": "smile", "motion_group": "Happy"},
    "excited": {"expression": "excited", "motion_group": "Happy"},
    "thinking": {"expression": "thinking", "motion_group": "Thinking"},
    "surprised": {"expression": "surprised", "motion_group": "Surprised"},
    "sleepy": {"expression": "sleepy", "motion_group": "Sleepy"},
    "worried": {"expression": "worried", "motion_group": "Worried"},
    "shy": {"expression": "shy", "motion_group": "Shy"},
}


class EmotionAnalyzer:
    """Analyzes and manages the character's emotional state."""

    def __init__(self):
        self.persona = get_persona_config()
        self.current_emotion: str = "neutral"
        self.emotion_history: list[tuple[str, datetime]] = []
        self._load_emotion_map()

    def _load_emotion_map(self):
        """Load emotion map from persona config."""
        emotions = self.persona.get("emotions", {})
        for tag, info in emotions.items():
            if tag not in EMOTION_MAP:
                EMOTION_MAP[tag] = {
                    "expression": info.get("live2d_expression", "normal"),
                    "motion_group": info.get("motion_group", "Idle"),
                }

    def update_emotion(self, emotion_tag: str) -> dict:
        """Update the current emotion and return Live2D expression data.

        Args:
            emotion_tag: The emotion tag from LLM response.

        Returns:
            Dict with Live2D expression and motion data.
        """
        if emotion_tag not in EMOTION_MAP:
            logger.warning(f"Unknown emotion tag: {emotion_tag}, falling back to neutral")
            emotion_tag = "neutral"

        self.current_emotion = emotion_tag
        self.emotion_history.append((emotion_tag, datetime.now()))

        # Keep only last 50 entries
        if len(self.emotion_history) > 50:
            self.emotion_history = self.emotion_history[-50:]

        expression_data = EMOTION_MAP[emotion_tag]
        logger.debug(f"Emotion updated: {emotion_tag} → {expression_data}")
        return expression_data

    def get_time_based_emotion(self) -> Optional[str]:
        """Get emotion modifier based on current time of day."""
        hour = datetime.now().hour

        if 23 <= hour or hour < 6:
            return "sleepy"
        return None

    def get_current_expression(self) -> dict:
        """Get the current Live2D expression data."""
        return EMOTION_MAP.get(self.current_emotion, EMOTION_MAP["neutral"])

    def get_emotion_for_frontend(self) -> dict:
        """Get emotion data formatted for the WebSocket frontend message."""
        expression = self.get_current_expression()
        return {
            "type": "emotion",
            "emotion": self.current_emotion,
            "expression": expression["expression"],
            "motion_group": expression["motion_group"],
        }

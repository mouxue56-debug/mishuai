"""Emotion analysis from LLM output and context.

Extracts emotion tags from LLM responses and maps them to Live2D expressions.
Tracks emotion momentum for natural emotional continuity across turns.
"""

from collections import Counter
from datetime import datetime, timedelta
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

# Emotion valence (positive/negative/neutral energy level)
EMOTION_VALENCE = {
    "neutral": 0,
    "happy": 2,
    "excited": 3,
    "thinking": 0,
    "surprised": 1,
    "sleepy": -1,
    "worried": -2,
    "shy": 1,
}


class EmotionAnalyzer:
    """Analyzes and manages the character's emotional state.

    Tracks emotion history to provide momentum hints to the LLM,
    enabling natural emotional continuity across conversation turns.
    """

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

    def get_emotion_momentum_hint(self) -> str:
        """Get a natural language hint about recent emotional state.

        Used by the system prompt to maintain emotional continuity.
        Returns empty string if no meaningful momentum exists.
        """
        # Only consider recent emotions (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        recent = [
            (e, t) for e, t in self.emotion_history
            if t >= cutoff
        ]

        if len(recent) < 2:
            return ""

        # Count recent emotions
        emotion_counts = Counter(e for e, _ in recent)
        dominant = emotion_counts.most_common(1)[0]
        dominant_emotion, dominant_count = dominant

        # If same emotion dominated recently, hint at it
        if dominant_count >= 2 and dominant_emotion != "neutral":
            hints = {
                "happy": "最近の会話で嬉しい気持ちが続いています。テンション高めでOK。",
                "excited": "猫の話などで盛り上がっています！興奮が続いている感じで。",
                "worried": "心配な話題が続いています。気遣いを忘れずに。",
                "sleepy": "深夜で眠そうな雰囲気が続いています。",
                "shy": "照れている流れが続いています。少し恥ずかしそうに。",
                "thinking": "考え事が多い会話です。真剣に向き合って。",
                "surprised": "驚きが連続しています。リアクション大きめで。",
            }
            return hints.get(dominant_emotion, "")

        # Check for mood shift (valence change)
        if len(recent) >= 3:
            recent_emotions = [e for e, _ in recent[-3:]]
            valences = [EMOTION_VALENCE.get(e, 0) for e in recent_emotions]
            avg_valence = sum(valences) / len(valences)

            if avg_valence >= 2:
                return "全体的にポジティブな会話の流れです。明るく！"
            elif avg_valence <= -1:
                return "少し落ち着いた/心配な会話の流れです。優しく。"

        return ""

    def get_conversation_rhythm_hint(self, user_turn_count: int) -> str:
        """Get conversation rhythm hints based on turn count and patterns.

        Suggests when to ask follow-up questions, empathize, or summarize.

        Args:
            user_turn_count: Number of user turns so far in the conversation.

        Returns:
            Rhythm hint string for the system prompt.
        """
        hints = []

        # After a few turns, start asking follow-up questions
        if user_turn_count == 2:
            hints.append("相手の話をもう少し深掘りしてもいいかも。「それでそれで？」のように。")
        elif user_turn_count == 5:
            hints.append("会話が続いています。軽く今までの話題を振り返ってもOK。")
        elif user_turn_count >= 8 and user_turn_count % 4 == 0:
            hints.append("長い会話です。相手が疲れてないか気遣ってもいいかも。")

        # Check if last few emotions were the same — suggest variety
        if len(self.emotion_history) >= 3:
            last_3 = [e for e, _ in self.emotion_history[-3:]]
            if len(set(last_3)) == 1 and last_3[0] != "neutral":
                hints.append(f"感情が{last_3[0]}のまま固定されています。自然な変化を入れて。")

        return "\n".join(hints) if hints else ""

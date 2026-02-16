"""Metrics collector for dashboard display.

Aggregates turn_logs data from LongTermMemory and provides
session-level metrics for the frontend dashboard.
"""

from collections import Counter
from statistics import mean

from src.utils.logger import get_logger

logger = get_logger("metrics")


class MetricsCollector:
    """Aggregates turn_logs data for dashboard display."""

    def __init__(self, long_term_memory):
        self._ltm = long_term_memory

    async def get_dashboard_data(self) -> dict:
        """Return aggregated metrics for the dashboard.

        Returns:
            Dict with session metrics suitable for frontend display.
        """
        try:
            logs = await self._ltm.get_recent_turn_logs(100)
        except Exception as e:
            logger.warning(f"Failed to fetch turn logs: {e}")
            logs = []

        if not logs:
            return {
                "total_turns": 0,
                "avg_llm_ms": 0,
                "avg_tts_ms": 0,
                "avg_total_ms": 0,
                "total_cost_jpy": 0,
                "model_usage": {},
                "emotion_distribution": {},
                "tools_usage": {},
                "speaker_distribution": {},
                "abort_rate": 0,
                "error_rate": 0,
                "interrupt_count": 0,
                "cache_hit_rate": 0,
            }

        total = len(logs)

        # Latency stats
        llm_times = [l.get("llm_duration_ms", 0) for l in logs if l.get("llm_duration_ms")]
        tts_times = [l.get("tts_duration_ms", 0) for l in logs if l.get("tts_duration_ms")]
        total_times = [l.get("total_duration_ms", 0) for l in logs if l.get("total_duration_ms")]

        # Model usage
        model_usage = dict(Counter(
            l.get("llm_model", "unknown") for l in logs if l.get("llm_model")
        ))

        # Emotion distribution
        emotion_dist = dict(Counter(
            l.get("emotion", "neutral") for l in logs if l.get("emotion")
        ))

        # Tools usage
        tools = []
        for l in logs:
            tool_names = l.get("tools_called", [])
            if isinstance(tool_names, str):
                tools.append(tool_names)
            elif isinstance(tool_names, list):
                tools.extend(tool_names)
        tools_usage = dict(Counter(tools))

        # Speaker distribution
        speaker_dist = dict(Counter(
            l.get("speaker_id", "unknown") for l in logs if l.get("speaker_id")
        ))

        # Rates
        abort_count = sum(1 for l in logs if l.get("aborted"))
        error_count = sum(1 for l in logs if l.get("error"))
        cache_hits = sum(1 for l in logs if l.get("cache_hit"))

        return {
            "total_turns": total,
            "avg_llm_ms": round(mean(llm_times), 1) if llm_times else 0,
            "avg_tts_ms": round(mean(tts_times), 1) if tts_times else 0,
            "avg_total_ms": round(mean(total_times), 1) if total_times else 0,
            "total_cost_jpy": round(sum(l.get("estimated_cost", 0) for l in logs), 2),
            "model_usage": model_usage,
            "emotion_distribution": emotion_dist,
            "tools_usage": tools_usage,
            "speaker_distribution": speaker_dist,
            "abort_rate": round(abort_count / max(total, 1) * 100, 1),
            "error_rate": round(error_count / max(total, 1) * 100, 1),
            "interrupt_count": abort_count,
            "cache_hit_rate": round(cache_hits / max(total, 1) * 100, 1),
        }

    async def get_session_summary(self) -> dict:
        """Get current session summary for WS broadcast."""
        return await self.get_dashboard_data()

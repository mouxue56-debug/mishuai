"""Proactive Speech Scheduler.

Determines when the AI should speak up without being prompted.
Inspired by N.E.K.O's cross-server proactive task analysis.

Trigger types:
1. Time-based: Morning greeting, goodnight, hourly idle chatter
2. Idle timeout: If no conversation for N minutes, say something
3. Vision-triggered: Person detected on camera → greet them
4. Reminder-based: (already in main.py, but scheduler can enhance)
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable

from src.utils.config_loader import get_main_config, get_persona_config
from src.utils.logger import get_logger

logger = get_logger("scheduler")


class ProactiveScheduler:
    """Decides when and what the AI should proactively say.

    Outputs proactive messages through a callback that feeds into
    the pipeline's output queue (same path as regular responses).
    """

    def __init__(self):
        config = get_main_config()
        sched_config = config.get("proactive", {})

        self.enabled = sched_config.get("enabled", True)
        self.idle_timeout_min = sched_config.get("idle_timeout_min", 15)
        self.greeting_hours = sched_config.get("greeting_hours", [9, 13])
        self.check_interval_sec = sched_config.get("check_interval_sec", 60)

        self._running = False
        self._last_interaction_time = time.time()
        self._last_greeting_hour: Optional[int] = None
        self._last_proactive_time = 0.0
        self._cooldown_sec = sched_config.get("cooldown_sec", 300)  # 5 min between proactive messages

        # Callback: async fn(text: str, emotion: str, trigger: str)
        self._on_proactive_speak: Optional[Callable] = None

        # Vision state (set externally)
        self._pending_vision_event: Optional[str] = None

    def on_proactive_speak(self, callback: Callable):
        """Register callback for proactive speech events."""
        self._on_proactive_speak = callback

    def notify_interaction(self):
        """Call this whenever a user interaction occurs (resets idle timer)."""
        self._last_interaction_time = time.time()

    def notify_vision_event(self, description: str):
        """Call this when the vision module detects something noteworthy."""
        self._pending_vision_event = description

    async def start(self):
        """Start the scheduler loop."""
        if not self.enabled:
            logger.info("Proactive scheduler disabled")
            return

        self._running = True
        logger.info(f"Proactive scheduler started (idle={self.idle_timeout_min}min, "
                     f"cooldown={self._cooldown_sec}s)")

        while self._running:
            try:
                await self._check_triggers()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            await asyncio.sleep(self.check_interval_sec)

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

    async def _check_triggers(self):
        """Evaluate all trigger conditions."""
        now = time.time()

        # Cooldown: don't spam proactive messages
        if now - self._last_proactive_time < self._cooldown_sec:
            return

        # Priority 1: Vision event (someone appeared)
        if self._pending_vision_event:
            event = self._pending_vision_event
            self._pending_vision_event = None
            await self._fire("vision", event)
            return

        # Priority 2: Time-based greetings
        hour = datetime.now().hour
        if hour in self.greeting_hours and self._last_greeting_hour != hour:
            self._last_greeting_hour = hour
            await self._fire("greeting", self._make_greeting(hour))
            return

        # Priority 3: Idle timeout
        idle_sec = now - self._last_interaction_time
        if idle_sec > self.idle_timeout_min * 60:
            # Reset timer so we don't fire repeatedly
            self._last_interaction_time = now
            await self._fire("idle", self._make_idle_message())
            return

    async def _fire(self, trigger: str, context: str):
        """Fire a proactive speech event."""
        if not self._on_proactive_speak:
            return

        logger.info(f"Proactive trigger: {trigger} → {context[:60]}")
        self._last_proactive_time = time.time()

        cb = self._on_proactive_speak
        if asyncio.iscoroutinefunction(cb):
            await cb(context, trigger)
        else:
            cb(context, trigger)

    def _make_greeting(self, hour: int) -> str:
        """Build a greeting prompt based on time of day."""
        if 5 <= hour < 11:
            return "朝の挨拶をしてください。今日の予定があれば軽く触れてください。"
        elif 11 <= hour < 14:
            return "お昼の挨拶をしてください。午前中お疲れ様でしたと声をかけてください。"
        elif 14 <= hour < 18:
            return "午後の挨拶をしてください。"
        elif 18 <= hour < 22:
            return "夕方の挨拶をしてください。今日一日お疲れ様でしたと伝えてください。"
        else:
            return "夜遅いです。まだ起きているなら早めに休むよう声をかけてください。"

    def _make_idle_message(self) -> str:
        """Build a prompt for idle chatter."""
        hour = datetime.now().hour
        persona = get_persona_config()
        cat_trivia = persona.get("cat_trivia", [])

        if cat_trivia:
            return (
                "しばらく会話がありません。猫の豆知識を一つ共有するか、"
                "何か手伝えることがないか軽く声をかけてください。"
                "短めに、自然な感じでお願いします。"
            )
        else:
            return (
                "しばらく会話がありません。"
                "何か手伝えることがないか軽く声をかけてください。短めに。"
            )

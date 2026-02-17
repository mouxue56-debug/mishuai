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
import random
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable

import aiohttp

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
        self.idle_min_sec = sched_config.get("idle_min_sec", 60)    # 最短1分钟
        self.idle_max_sec = sched_config.get("idle_max_sec", 300)   # 最长5分钟
        self.greeting_hours = sched_config.get("greeting_hours", [9, 13])
        self.check_interval_sec = sched_config.get("check_interval_sec", 30)  # 检查频率提高到30秒

        self._running = False
        self._last_interaction_time = time.time()
        self._last_greeting_hour: Optional[int] = None
        self._last_proactive_time = 0.0
        self._cooldown_sec = sched_config.get("cooldown_sec", 60)  # 冷却缩短到1分钟
        self._next_idle_threshold = self._random_idle_threshold()  # 下次空闲触发时间

        # Callback: async fn(text: str, emotion: str, trigger: str)
        self._on_proactive_speak: Optional[Callable] = None

        # Vision state (set externally)
        self._pending_vision_event: Optional[str] = None

        # Weather cache (refresh every 30 minutes)
        self._weather_cache: Optional[str] = None
        self._weather_cache_time: float = 0
        self._weather_cache_ttl: float = 1800  # 30分钟
        self._weather_location = sched_config.get("weather_location", "Osaka")  # 默认大阪

    def on_proactive_speak(self, callback: Callable):
        """Register callback for proactive speech events."""
        self._on_proactive_speak = callback

    def _random_idle_threshold(self) -> float:
        """生成1-5分钟之间的随机等待时间（秒）"""
        return random.uniform(self.idle_min_sec, self.idle_max_sec)

    def notify_interaction(self):
        """Call this whenever a user interaction occurs (resets idle timer)."""
        self._last_interaction_time = time.time()
        self._next_idle_threshold = self._random_idle_threshold()  # 每次交互后重新随机

    def notify_vision_event(self, description: str):
        """Call this when the vision module detects something noteworthy."""
        self._pending_vision_event = description

    async def start(self):
        """Start the scheduler loop."""
        self._running = True
        logger.info(f"Proactive scheduler started (idle={self.idle_timeout_min}min, "
                     f"cooldown={self._cooldown_sec}s)")

        # 首次获取天气
        await self._fetch_weather()

        while self._running:
            try:
                # 定期更新天气缓存
                await self._fetch_weather()
                await self._check_triggers()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            await asyncio.sleep(self.check_interval_sec)

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

    async def _check_triggers(self):
        """Evaluate all trigger conditions."""
        if not self.enabled:
            return
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

        # Priority 3: Idle timeout (1-5分钟随机间隔)
        idle_sec = now - self._last_interaction_time
        if idle_sec > self._next_idle_threshold:
            # 重置计时器并生成下一个随机阈值
            self._last_interaction_time = now
            self._next_idle_threshold = self._random_idle_threshold()
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
        weather_hint = f"\n現在の天気: {self._weather_cache}。天気にも触れてください。" if self._weather_cache else ""

        if 5 <= hour < 11:
            return f"朝の挨拶をユキらしくしてください。元気に、今日も一緒に頑張ろうという気持ちで。猫の話を少し混ぜてもOK。{weather_hint}"
        elif 11 <= hour < 14:
            return f"お昼の挨拶をしてください。午前お疲れ様、何か食べましたか？と気遣いつつ。{weather_hint}"
        elif 14 <= hour < 18:
            return f"午後の挨拶をしてください。おやつの時間だとか、午後も頑張ろうとか、軽い感じで。{weather_hint}"
        elif 18 <= hour < 22:
            return f"夕方の挨拶をしてください。今日一日お疲れ様でした、と労いつつ、ゆっくり休んでねという気持ちで。{weather_hint}"
        else:
            return f"夜遅いです。眠そうにしつつ、まだ起きてるの？早く寝た方がいいですよ〜と声をかけてください。{weather_hint}"

    def _make_idle_message(self) -> str:
        """Build a prompt for idle chatter — varied and natural."""
        hour = datetime.now().hour
        persona = get_persona_config()
        char = persona.get("character", {})
        cat_trivia = char.get("cat_trivia", [])

        # 根据时间段调整语气
        if 23 <= hour or hour < 6:
            time_hint = "今は深夜です。眠そうにしながら"
        elif 6 <= hour < 9:
            time_hint = "朝早い時間です。元気に"
        elif 12 <= hour < 13:
            time_hint = "お昼の時間です。"
        else:
            time_hint = ""

        # 多样化的搭话模式，随机选择
        idle_prompts = [
            # 猫の豆知識系
            "猫の豆知識を一つ自然に共有してください。「あ、そういえば知ってました？」のように自然な切り出しで。",
            # 关心系
            "ウィルさんの体調や気分を軽く気遣ってください。「ウィルさん、疲れてません？」のように。",
            # 独り言系
            "独り言のように何か呟いてください。「う〜ん、今日のおやつ何にしようかな...」のような軽い内容で。",
            # 好奇心系
            "何か最近気になったことや面白い発見を話してください。好奇心旺盛な感じで。",
            # 手伝い系
            "何か手伝えることがないか声をかけてください。「なんか手伝えることありますか〜？」のように。",
            # おすすめ系
            "今の時間帯に合った何かをおすすめしてください（飲み物、休憩、ストレッチなど）。",
        ]

        if cat_trivia:
            # 猫豆知識は確率を上げる
            idle_prompts.append("猫の豆知識を一つ教えてください。テンション高めで。")
            idle_prompts.append("サイベリアンの話をしてください。猫好きとして熱く。")

        # 天気情報があれば天気系の搭話も追加
        if self._weather_cache:
            idle_prompts.append(
                f"現在の天気情報: {self._weather_cache}。"
                f"天気に触れて自然に話しかけてください。「今日は寒いですね〜」のように。"
            )

        chosen = random.choice(idle_prompts)
        return (
            f"{time_hint}"
            f"ユキとして自然に話しかけてください。{chosen}"
            f"1-2文で短く、キャラに合った感じで。"
        )

    async def _fetch_weather(self):
        """wttr.in から天気情報を取得（無料、APIキー不要）"""
        now = time.time()
        if self._weather_cache and (now - self._weather_cache_time) < self._weather_cache_ttl:
            return  # キャッシュがまだ有効

        try:
            url = f"https://wttr.in/{self._weather_location}?format=%C+%t+%h+%w&lang=ja"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        text = text.strip()
                        if text and "Unknown" not in text:
                            self._weather_cache = f"{self._weather_location}: {text}"
                            self._weather_cache_time = now
                            logger.info(f"Weather updated: {self._weather_cache}")
                        else:
                            logger.warning(f"Weather response invalid: {text}")
                    else:
                        logger.warning(f"Weather fetch failed: HTTP {resp.status}")
        except asyncio.TimeoutError:
            logger.debug("Weather fetch timeout (non-critical)")
        except Exception as e:
            logger.debug(f"Weather fetch error (non-critical): {type(e).__name__}: {e}")

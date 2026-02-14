"""Wake word detection for hands-free activation.

Uses Sherpa-ONNX for offline wake word detection.
The AI secretary activates when the wake word is detected.
"""

import asyncio
from typing import Callable, Optional

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("wake_word")


class WakeWordDetector:
    """Offline wake word detector using Sherpa-ONNX.

    Listens continuously for the configured wake word and triggers
    the pipeline when detected.
    """

    def __init__(self):
        config = get_main_config()
        wake_config = config.get("audio", {}).get("wake_word", {})

        self.enabled = wake_config.get("enabled", True)
        self.keyword = wake_config.get("keyword", "ミケちゃん")
        self.sensitivity = wake_config.get("sensitivity", 0.5)
        self.engine = wake_config.get("engine", "sherpa_onnx")

        self._detector = None
        self._on_wake: Optional[Callable] = None
        self._running = False

    def on_wake(self, callback: Callable):
        """Register callback for when wake word is detected."""
        self._on_wake = callback

    async def initialize(self):
        """Initialize the wake word detection engine."""
        if not self.enabled:
            logger.info("Wake word detection disabled")
            return

        if self.engine == "sherpa_onnx":
            await self._init_sherpa()
        else:
            logger.warning(f"Unknown wake word engine: {self.engine}")

    async def _init_sherpa(self):
        """Initialize Sherpa-ONNX keyword spotter."""
        try:
            import sherpa_onnx

            # Note: Requires a keyword model file
            # For Japanese wake words, you may need a custom model
            # or use Sherpa-ONNX's keyword spotting with a phoneme list
            logger.info(f"Wake word detector initialized: '{self.keyword}'")
            logger.info("Note: Custom wake word model needed for Japanese keywords")
        except ImportError:
            logger.warning(
                "sherpa-onnx not installed. Wake word detection disabled. "
                "Run: pip install sherpa-onnx"
            )
            self.enabled = False

    async def process_audio(self, frame: bytes) -> bool:
        """Process an audio frame for wake word detection.

        Args:
            frame: Raw audio bytes (int16, 16kHz).

        Returns:
            True if wake word detected, False otherwise.
        """
        if not self.enabled:
            return False

        # Placeholder: Real implementation would use Sherpa-ONNX keyword spotter
        # The actual detection depends on the model file
        return False

    async def detect_loop(self, audio_stream):
        """Continuous wake word detection loop.

        Args:
            audio_stream: Async iterator yielding audio frames.
        """
        if not self.enabled:
            return

        self._running = True
        logger.info(f"Wake word detection active: listening for '{self.keyword}'")

        while self._running:
            try:
                async for frame in audio_stream:
                    if not self._running:
                        break

                    detected = await self.process_audio(frame)
                    if detected:
                        logger.info(f"Wake word detected: '{self.keyword}'")
                        if self._on_wake:
                            if asyncio.iscoroutinefunction(self._on_wake):
                                await self._on_wake()
                            else:
                                self._on_wake()
            except Exception as e:
                logger.error(f"Wake word detection error: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        """Stop wake word detection."""
        self._running = False
        logger.info("Wake word detection stopped")

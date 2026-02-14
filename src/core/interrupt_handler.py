"""Interrupt handler for managing speech interruptions.

When a new user utterance is detected while the AI is speaking,
the current response should be interrupted gracefully.
(Inspired by Pipecat's interruption handling pattern)
"""

import asyncio
from enum import Enum
from typing import Callable, Optional

from src.utils.logger import get_logger

logger = get_logger("interrupt")


class PipelineState(Enum):
    IDLE = "idle"                   # Waiting for input
    LISTENING = "listening"         # VAD detected speech
    PROCESSING = "processing"      # LLM is generating response
    SPEAKING = "speaking"          # TTS is playing audio
    INTERRUPTED = "interrupted"    # User interrupted during speech


class InterruptHandler:
    """Manages interruption of the AI's speech when user starts talking."""

    def __init__(self):
        self.state = PipelineState.IDLE
        self._interrupt_event = asyncio.Event()
        self._on_interrupt_callbacks: list[Callable] = []
        self._speaking_task: Optional[asyncio.Task] = None

    @property
    def is_speaking(self) -> bool:
        return self.state == PipelineState.SPEAKING

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_event.is_set()

    def set_state(self, state: PipelineState):
        """Update pipeline state."""
        old_state = self.state
        self.state = state
        if old_state != state:
            logger.debug(f"Pipeline state: {old_state.value} → {state.value}")

    def register_interrupt_callback(self, callback: Callable):
        """Register a callback to be called when interruption occurs."""
        self._on_interrupt_callbacks.append(callback)

    async def interrupt(self):
        """Signal an interruption (user started speaking while AI is talking)."""
        if self.state == PipelineState.SPEAKING:
            logger.info("Interruption detected! Stopping current speech.")
            self._interrupt_event.set()
            self.state = PipelineState.INTERRUPTED

            # Cancel the speaking task if it exists
            if self._speaking_task and not self._speaking_task.done():
                self._speaking_task.cancel()

            # Notify callbacks
            for callback in self._on_interrupt_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Interrupt callback error: {e}")

    def clear_interrupt(self):
        """Clear the interruption flag."""
        self._interrupt_event.clear()

    def set_speaking_task(self, task: asyncio.Task):
        """Set the current speaking task (so it can be cancelled on interrupt)."""
        self._speaking_task = task

    async def wait_for_interrupt(self, timeout: float = None) -> bool:
        """Wait for an interruption event.

        Returns:
            True if interrupted, False if timed out.
        """
        try:
            await asyncio.wait_for(self._interrupt_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def should_stop(self) -> bool:
        """Check if the current operation should stop due to interruption."""
        return self._interrupt_event.is_set()

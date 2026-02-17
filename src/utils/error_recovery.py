"""Error recovery patterns inspired by py-xiaozhi.

Provides reusable components for robust error handling:
1. ExponentialBackoff - Retry with exponential delays
2. CircuitBreaker - Stop calling failing services temporarily
3. SilencePeriod - Anti-echo suppression after TTS playback
4. TaskLifecycle - Tracked async tasks with cleanup
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger("recovery")


class ExponentialBackoff:
    """Retry operations with exponential backoff delays.

    Inspired by py-xiaozhi's reconnection strategy:
    - Start with base_delay
    - Double on each failure (up to max_delay)
    - Reset on success
    - Optional jitter to avoid thundering herd

    Usage:
        backoff = ExponentialBackoff()
        while True:
            try:
                result = await do_something()
                backoff.reset()
                break
            except Exception:
                delay = backoff.next_delay()
                await asyncio.sleep(delay)
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self._attempts = 0
        self._current_delay = base_delay

    def next_delay(self) -> float:
        """Get the next delay and advance the backoff counter."""
        import random
        delay = min(self._current_delay, self.max_delay)
        # Add jitter (±jitter%)
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        self._current_delay *= self.multiplier
        self._attempts += 1
        return max(delay, 0.1)

    def reset(self):
        """Reset after a successful operation."""
        self._attempts = 0
        self._current_delay = self.base_delay

    @property
    def attempts(self) -> int:
        return self._attempts


class CircuitBreaker:
    """Prevent repeated calls to a failing service.

    Inspired by py-xiaozhi's circuit breaker pattern:
    - CLOSED: normal operation, calls pass through
    - OPEN: service is failing, calls are rejected immediately
    - HALF_OPEN: after cooldown, allow one test call

    Usage:
        breaker = CircuitBreaker(failure_threshold=3, cooldown_sec=30)
        if breaker.can_call():
            try:
                result = await external_service()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # Use fallback or return cached result
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_sec: float = 30.0,
        name: str = "",
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.name = name or "circuit"

        self._failures = 0
        self._state = "closed"  # closed | open | half_open
        self._last_failure_time = 0.0
        self._last_success_time = 0.0

    def can_call(self) -> bool:
        """Check if a call should be allowed."""
        if self._state == "closed":
            return True
        elif self._state == "open":
            # Check if cooldown has passed
            if time.time() - self._last_failure_time > self.cooldown_sec:
                self._state = "half_open"
                logger.info(f"CircuitBreaker [{self.name}]: half_open (testing)")
                return True
            return False
        elif self._state == "half_open":
            return True  # Allow one test call
        return False

    def record_success(self):
        """Record a successful call."""
        if self._state == "half_open":
            logger.info(f"CircuitBreaker [{self.name}]: closed (recovered)")
        self._failures = 0
        self._state = "closed"
        self._last_success_time = time.time()

    def record_failure(self):
        """Record a failed call."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"CircuitBreaker [{self.name}]: OPEN "
                f"({self._failures} failures, cooldown={self.cooldown_sec}s)"
            )

    @property
    def state(self) -> str:
        return self._state

    @property
    def failures(self) -> int:
        return self._failures


class SilencePeriod:
    """Anti-echo suppression period after TTS playback.

    Inspired by py-xiaozhi's silence_period pattern:
    After TTS finishes playing, suppress microphone input for a short
    period to prevent echo from being picked up as new user input.

    Usage:
        silence = SilencePeriod(duration_ms=200)
        # After TTS playback ends:
        silence.start()
        # In audio processing:
        if silence.is_active():
            return  # Skip this audio frame
    """

    def __init__(self, duration_ms: int = 200):
        self.duration_ms = duration_ms
        self._start_time: float = 0.0
        self._active = False

    def start(self):
        """Start the silence period."""
        self._start_time = time.time()
        self._active = True

    def is_active(self) -> bool:
        """Check if the silence period is still active."""
        if not self._active:
            return False
        elapsed_ms = (time.time() - self._start_time) * 1000
        if elapsed_ms > self.duration_ms:
            self._active = False
            return False
        return True

    def cancel(self):
        """Cancel the silence period early."""
        self._active = False


class TaskLifecycle:
    """Tracked async task management with cleanup.

    Inspired by py-xiaozhi's spawn() pattern:
    - Tracks all spawned async tasks
    - Provides cleanup on shutdown
    - Logs task failures

    Usage:
        lifecycle = TaskLifecycle()
        lifecycle.spawn(my_coroutine(), name="worker")
        # ... later
        await lifecycle.shutdown()
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._counter = 0

    def spawn(
        self,
        coro,
        name: str = None,
    ) -> asyncio.Task:
        """Spawn a tracked async task.

        Args:
            coro: Coroutine to run.
            name: Optional name for logging.

        Returns:
            The created asyncio.Task.
        """
        self._counter += 1
        task_name = name or f"task_{self._counter}"

        task = asyncio.create_task(coro, name=task_name)
        self._tasks[task_name] = task

        def _on_done(t: asyncio.Task):
            self._tasks.pop(task_name, None)
            if t.cancelled():
                logger.debug(f"Task [{task_name}] cancelled")
            elif t.exception():
                logger.error(f"Task [{task_name}] failed: {t.exception()}")
            else:
                logger.debug(f"Task [{task_name}] completed")

        task.add_done_callback(_on_done)
        return task

    async def shutdown(self, timeout: float = 5.0):
        """Cancel all tracked tasks and wait for completion.

        Args:
            timeout: Maximum seconds to wait for tasks to finish.
        """
        if not self._tasks:
            return

        logger.info(f"TaskLifecycle: shutting down {len(self._tasks)} tasks")
        for name, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()

        if self._tasks:
            pending = [t for t in self._tasks.values() if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=timeout)

        self._tasks.clear()

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())

    @property
    def task_names(self) -> list[str]:
        return list(self._tasks.keys())


async def retry_with_backoff(
    fn: Callable[..., Awaitable],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable] = None,
    **kwargs,
):
    """Convenience function to retry an async operation with exponential backoff.

    Args:
        fn: Async function to call.
        max_attempts: Maximum number of attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        on_retry: Optional callback(attempt, error, delay) on each retry.
        **kwargs: Arguments passed to fn.

    Returns:
        Result from fn.

    Raises:
        The last exception if all attempts fail.
    """
    backoff = ExponentialBackoff(
        base_delay=base_delay, max_delay=max_delay
    )
    last_error = None

    for attempt in range(max_attempts):
        try:
            result = await fn(**kwargs)
            backoff.reset()
            return result
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = backoff.next_delay()
                if on_retry:
                    on_retry(attempt + 1, e, delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_attempts} in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

    raise last_error

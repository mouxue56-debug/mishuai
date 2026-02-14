"""Microphone input with VAD (Voice Activity Detection).

Captures audio from the microphone, uses WebRTC VAD to detect speech,
and sends complete utterances to the pipeline for processing.
"""

import asyncio
import struct
from collections import deque
from typing import Callable, Optional

import numpy as np

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("microphone")


class MicrophoneInput:
    """Handles microphone capture with Voice Activity Detection.

    Uses PyAudio for capture and WebRTC VAD for speech detection.
    Sends completed utterances to a callback for ASR processing.
    """

    def __init__(self):
        config = get_main_config()
        audio_config = config.get("audio", {})
        input_config = audio_config.get("input", {})
        vad_config = audio_config.get("vad", {})

        self.sample_rate = input_config.get("sample_rate", 16000)
        self.channels = input_config.get("channels", 1)
        self.chunk_size = input_config.get("chunk_size", 1024)
        self.device_index = input_config.get("device_index")

        self.vad_mode = vad_config.get("mode", 3)
        self.frame_duration_ms = vad_config.get("frame_duration_ms", 30)
        self.silence_threshold_ms = vad_config.get("silence_threshold_ms", 800)

        self._stream = None
        self._audio = None
        self._running = False
        self._on_utterance: Optional[Callable] = None
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None

    def on_utterance(self, callback: Callable):
        """Register callback for when a complete utterance is detected."""
        self._on_utterance = callback

    def on_speech_start(self, callback: Callable):
        """Register callback for when speech begins."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable):
        """Register callback for when speech ends."""
        self._on_speech_end = callback

    async def start(self):
        """Start capturing audio from the microphone."""
        try:
            import pyaudio
            import webrtcvad
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Run: pip install pyaudio webrtcvad")
            return

        self._audio = pyaudio.PyAudio()
        self._vad = webrtcvad.Vad(self.vad_mode)

        # Calculate frame size for VAD
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

        # Open mic stream
        stream_kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.channels,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": self.frame_size,
        }
        if self.device_index is not None:
            stream_kwargs["input_device_index"] = self.device_index

        self._stream = self._audio.open(**stream_kwargs)
        self._running = True

        logger.info(f"Microphone started (rate={self.sample_rate}, vad_mode={self.vad_mode})")
        await self._capture_loop()

    async def _capture_loop(self):
        """Main audio capture and VAD loop."""
        speech_frames: list[bytes] = []
        is_speaking = False
        silence_frames = 0
        silence_threshold_frames = int(
            self.silence_threshold_ms / self.frame_duration_ms
        )
        # Ring buffer for pre-speech audio (to catch the start of utterances)
        pre_speech_buffer: deque = deque(maxlen=10)

        while self._running:
            try:
                # Read audio frame (run in thread to avoid blocking)
                frame = await asyncio.to_thread(
                    self._stream.read, self.frame_size, exception_on_overflow=False
                )

                # Check VAD
                is_speech = self._vad.is_speech(frame, self.sample_rate)

                if is_speech:
                    if not is_speaking:
                        # Speech just started
                        is_speaking = True
                        silence_frames = 0
                        logger.debug("Speech started")
                        if self._on_speech_start:
                            await self._on_speech_start()
                        # Include pre-speech buffer
                        speech_frames = list(pre_speech_buffer)

                    speech_frames.append(frame)
                    silence_frames = 0
                else:
                    if is_speaking:
                        speech_frames.append(frame)
                        silence_frames += 1

                        if silence_frames >= silence_threshold_frames:
                            # Speech ended - send utterance
                            is_speaking = False
                            logger.debug(f"Speech ended ({len(speech_frames)} frames)")

                            if self._on_speech_end:
                                await self._on_speech_end()

                            # Combine frames and send
                            audio_data = b"".join(speech_frames)
                            speech_frames = []

                            if self._on_utterance:
                                await self._on_utterance(audio_data)
                    else:
                        pre_speech_buffer.append(frame)

            except Exception as e:
                logger.error(f"Mic capture error: {e}")
                await asyncio.sleep(0.1)

    async def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
        logger.info("Microphone stopped")

    def get_audio_level(self, frame: bytes) -> float:
        """Calculate the RMS audio level of a frame.

        Args:
            frame: Raw audio bytes (int16).

        Returns:
            RMS level as a float (0.0 to 1.0).
        """
        try:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples ** 2))
            return min(rms / 32768.0, 1.0)
        except Exception:
            return 0.0

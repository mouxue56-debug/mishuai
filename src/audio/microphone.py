"""Microphone input with VAD (Voice Activity Detection).

Captures audio from the microphone, uses WebRTC VAD or Silero VAD to detect
speech, and sends complete utterances to the pipeline for processing.

Supports two VAD engines:
- webrtc: Lightweight, low-latency, good for most use cases
- silero: ML-based, more accurate, better for noisy environments
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

    Uses PyAudio for capture and either WebRTC VAD or Silero VAD for
    speech detection. Sends completed utterances to a callback for ASR.
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

        self.vad_engine = vad_config.get("engine", "webrtc")  # "webrtc" | "silero"
        self.vad_mode = vad_config.get("mode", 3)
        self.frame_duration_ms = vad_config.get("frame_duration_ms", 30)
        self.silence_threshold_ms = vad_config.get("silence_threshold_ms", 800)
        self.silero_threshold = vad_config.get("silero_threshold", 0.5)

        self._stream = None
        self._audio = None
        self._vad = None
        self._silero_model = None
        self._running = False
        self._suppressed = False  # Echo suppression during TTS playback
        self._on_utterance: Optional[Callable] = None
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None

    def suppress_echo(self):
        """Start echo suppression (during TTS playback).

        While suppressed:
        - on_speech_start still fires → allows registered user to interrupt
        - on_utterance is BLOCKED → prevents TTS echo from being processed as input
        """
        self._suppressed = True
        logger.debug("Echo suppression ON")

    def stop_suppress(self):
        """Stop echo suppression (after TTS playback ends)."""
        self._suppressed = False
        logger.debug("Echo suppression OFF")

    def on_utterance(self, callback: Callable):
        """Register callback for when a complete utterance is detected."""
        self._on_utterance = callback

    def on_speech_start(self, callback: Callable):
        """Register callback for when speech begins."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable):
        """Register callback for when speech ends."""
        self._on_speech_end = callback

    def _init_vad(self):
        """Initialize the selected VAD engine."""
        if self.vad_engine == "silero":
            return self._init_silero_vad()
        return self._init_webrtc_vad()

    def _init_webrtc_vad(self):
        """Initialize WebRTC VAD."""
        try:
            import webrtcvad
        except ImportError:
            logger.error("webrtcvad not installed. Run: pip install webrtcvad")
            return False
        self._vad = webrtcvad.Vad(self.vad_mode)
        logger.info(f"Using WebRTC VAD (mode={self.vad_mode})")
        return True

    def _init_silero_vad(self):
        """Initialize Silero VAD (torch-based, more accurate)."""
        try:
            import torch
        except ImportError:
            logger.warning("torch not installed for Silero VAD, falling back to WebRTC")
            return self._init_webrtc_vad()

        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._silero_model = model
            logger.info(f"Using Silero VAD (threshold={self.silero_threshold})")
            return True
        except Exception as e:
            logger.warning(f"Silero VAD init failed: {e}, falling back to WebRTC")
            return self._init_webrtc_vad()

    def _is_speech(self, frame: bytes) -> bool:
        """Check if an audio frame contains speech using the active VAD engine."""
        if self._silero_model is not None:
            return self._silero_is_speech(frame)
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self.sample_rate)
            except Exception:
                return False
        return False

    def _silero_is_speech(self, frame: bytes) -> bool:
        """Check speech using Silero VAD."""
        try:
            import torch
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio)
            confidence = self._silero_model(tensor, self.sample_rate).item()
            return confidence >= self.silero_threshold
        except Exception:
            return False

    async def start(self):
        """Start capturing audio from the microphone."""
        try:
            import pyaudio
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Run: pip install pyaudio")
            return

        self._audio = pyaudio.PyAudio()

        if not self._init_vad():
            logger.error("No VAD engine available, cannot start microphone")
            return

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
                is_speech = self._is_speech(frame)

                if is_speech:
                    if not is_speaking:
                        # Speech just started
                        is_speaking = True
                        silence_frames = 0
                        logger.debug("Speech started")
                        # on_speech_start always fires (even during echo suppression)
                        # — this allows barge-in interrupts
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
                            # Speech ended
                            is_speaking = False
                            logger.debug(f"Speech ended ({len(speech_frames)} frames)")

                            if self._on_speech_end:
                                await self._on_speech_end()

                            # Combine frames
                            audio_data = b"".join(speech_frames)
                            speech_frames = []

                            # Echo suppression: drop utterance during TTS playback
                            if self._suppressed:
                                logger.debug("Utterance dropped (echo suppression)")
                            elif self._on_utterance:
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

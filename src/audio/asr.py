"""ASR (Automatic Speech Recognition) module.

Converts speech audio to text using configurable engines:
- Sherpa-ONNX (offline, low latency)
- Faster-Whisper (offline, higher quality)
- Google Speech API (online)
"""

import asyncio
from typing import Optional

import numpy as np

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("asr")


class ASREngine:
    """Speech-to-text engine with multiple backend support."""

    def __init__(self):
        config = get_main_config()
        asr_config = config.get("asr", {})

        self.engine = asr_config.get("engine", "sherpa_onnx")
        self.language = asr_config.get("language", "ja")
        self.sample_rate = asr_config.get("sherpa_onnx", {}).get("sample_rate", 16000)

        self._recognizer = None

    async def initialize(self):
        """Initialize the ASR engine."""
        if self.engine == "sherpa_onnx":
            await self._init_sherpa()
        elif self.engine == "faster_whisper":
            await self._init_whisper()
        else:
            logger.warning(f"Unknown ASR engine: {self.engine}")

    async def _init_sherpa(self):
        """Initialize Sherpa-ONNX for streaming ASR."""
        try:
            import sherpa_onnx

            # Sherpa-ONNX streaming recognizer config
            # You need to download model files first
            # See: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/
            logger.info("Sherpa-ONNX ASR: ready (model files required)")
            logger.info(
                "Download Japanese model: "
                "https://github.com/k2-fsa/sherpa-onnx/releases "
                "(look for Japanese streaming ASR models)"
            )
        except ImportError:
            logger.warning("sherpa-onnx not installed. Run: pip install sherpa-onnx")

    async def _init_whisper(self):
        """Initialize Faster-Whisper for offline ASR."""
        try:
            from faster_whisper import WhisperModel

            self._recognizer = await asyncio.to_thread(
                WhisperModel, "small", device="cpu", compute_type="int8"
            )
            logger.info("Faster-Whisper ASR initialized (model=small)")
        except ImportError:
            logger.warning("faster-whisper not installed. Run: pip install faster-whisper")

    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes (int16, 16kHz mono).

        Returns:
            Transcribed text, or None if transcription failed.
        """
        if not audio_data:
            return None

        try:
            if self.engine == "sherpa_onnx":
                return await self._transcribe_sherpa(audio_data)
            elif self.engine == "faster_whisper":
                return await self._transcribe_whisper(audio_data)
            else:
                logger.error(f"No ASR engine available: {self.engine}")
                return None
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return None

    async def _transcribe_sherpa(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using Sherpa-ONNX."""
        if self._recognizer is None:
            logger.warning("Sherpa-ONNX recognizer not initialized")
            return None

        # Convert bytes to float32 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Run recognition
        # stream = self._recognizer.create_stream()
        # stream.accept_waveform(self.sample_rate, audio_array)
        # self._recognizer.decode_stream(stream)
        # text = stream.result.text
        # return text.strip() if text else None
        return None

    async def _transcribe_whisper(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using Faster-Whisper."""
        if self._recognizer is None:
            logger.warning("Faster-Whisper not initialized")
            return None

        # Convert bytes to float32 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Run transcription in thread
        segments, info = await asyncio.to_thread(
            self._recognizer.transcribe,
            audio_array,
            language=self.language,
            beam_size=5,
            best_of=3,
            vad_filter=True,
        )

        text = " ".join(segment.text for segment in segments)
        if text.strip():
            logger.info(f"ASR result: '{text.strip()}'")
            return text.strip()
        return None

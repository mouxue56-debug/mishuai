"""ASR (Automatic Speech Recognition) module.

Converts speech audio to text using configurable engines:
- DashScope Paraformer (online, multi-language, recommended)
- Sherpa-ONNX (offline, low latency)
- Faster-Whisper (offline, higher quality)
"""

import asyncio
import os
import tempfile
import wave
from http import HTTPStatus
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

        self.engine = asr_config.get("engine", "dashscope")
        self.language = asr_config.get("language", ["zh", "en", "ja"])
        self.sample_rate = asr_config.get("dashscope", {}).get(
            "sample_rate",
            asr_config.get("sherpa_onnx", {}).get("sample_rate", 16000),
        )

        # DashScope config
        ds_config = asr_config.get("dashscope", {})
        self._ds_model = ds_config.get("model", "paraformer-realtime-v2")

        self._recognizer = None
        self._ready = False

    async def initialize(self):
        """Initialize the ASR engine."""
        if self.engine == "dashscope":
            await self._init_dashscope()
        elif self.engine == "sherpa_onnx":
            await self._init_sherpa()
        elif self.engine == "faster_whisper":
            await self._init_whisper()
        else:
            logger.warning(f"Unknown ASR engine: {self.engine}")

    async def _init_dashscope(self):
        """Initialize DashScope Paraformer ASR."""
        try:
            import dashscope
            from dashscope.audio.asr import Recognition

            api_key = os.environ.get("QIANWEN_API_KEY") or os.environ.get(
                "DASHSCOPE_API_KEY"
            )
            if not api_key:
                logger.error(
                    "DashScope ASR: No API key (QIANWEN_API_KEY or DASHSCOPE_API_KEY)"
                )
                return

            dashscope.api_key = api_key
            self._ready = True

            lang_str = (
                ", ".join(self.language)
                if isinstance(self.language, list)
                else self.language
            )
            logger.info(
                f"DashScope ASR initialized (model={self._ds_model}, "
                f"languages={lang_str})"
            )
        except ImportError:
            logger.error("dashscope not installed. Run: pip install dashscope")

    async def _init_sherpa(self):
        """Initialize Sherpa-ONNX for streaming ASR."""
        try:
            import sherpa_onnx

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
            self._ready = True
            logger.info("Faster-Whisper ASR initialized (model=small)")
        except ImportError:
            logger.warning(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

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
            if self.engine == "dashscope":
                return await self._transcribe_dashscope(audio_data)
            elif self.engine == "sherpa_onnx":
                return await self._transcribe_sherpa(audio_data)
            elif self.engine == "faster_whisper":
                return await self._transcribe_whisper(audio_data)
            else:
                logger.error(f"No ASR engine available: {self.engine}")
                return None
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return None

    async def _transcribe_dashscope(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using DashScope Paraformer.

        Uses the Recognition.call() non-streaming API which accepts a file path.
        We write PCM data to a temporary WAV file, call the API, then clean up.
        """
        if not self._ready:
            logger.warning("DashScope ASR not initialized")
            return None

        # Prepare language hints
        lang_hints = (
            self.language
            if isinstance(self.language, list)
            else [self.language]
        )

        # Write PCM data to a temporary WAV file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # int16 = 2 bytes
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)

            # Call DashScope API in a thread (it's blocking)
            result = await asyncio.to_thread(
                self._call_dashscope, tmp_path, lang_hints
            )
            return result

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _call_dashscope(
        self, wav_path: str, lang_hints: list
    ) -> Optional[str]:
        """Synchronous DashScope Recognition.call() wrapper."""
        from dashscope.audio.asr import Recognition, RecognitionCallback

        # Recognition requires a callback even for synchronous call() mode
        callback = RecognitionCallback()

        recognition = Recognition(
            model=self._ds_model,
            callback=callback,
            format="wav",
            sample_rate=self.sample_rate,
            language_hints=lang_hints,
        )

        result = recognition.call(wav_path)

        if result.status_code == HTTPStatus.OK:
            sentences = result.get_sentence()
            if sentences:
                # get_sentence() returns a list of sentence dicts
                # Each dict has "text", "begin_time", "end_time", etc.
                if isinstance(sentences, list):
                    texts = [
                        s.get("text", "").strip()
                        for s in sentences
                        if s.get("text", "").strip()
                    ]
                    text = "".join(texts)
                elif isinstance(sentences, dict) and "text" in sentences:
                    text = sentences["text"].strip()
                else:
                    text = ""

                if text:
                    logger.info(f"ASR result: '{text}'")
                    return text
        else:
            logger.warning(
                f"DashScope ASR error: {result.status_code} "
                f"{getattr(result, 'message', 'unknown')}"
            )

        return None

    async def _transcribe_sherpa(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using Sherpa-ONNX."""
        if self._recognizer is None:
            logger.warning("Sherpa-ONNX recognizer not initialized")
            return None

        # Convert bytes to float32 array
        audio_array = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Run recognition (stub — model files required)
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
        audio_array = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Determine language parameter
        lang = self.language
        if isinstance(lang, list):
            lang = lang[0] if lang else None

        # Run transcription in thread
        segments, info = await asyncio.to_thread(
            self._recognizer.transcribe,
            audio_array,
            language=lang,
            beam_size=5,
            best_of=3,
            vad_filter=True,
        )

        text = " ".join(segment.text for segment in segments)
        if text.strip():
            logger.info(f"ASR result: '{text.strip()}'")
            return text.strip()
        return None

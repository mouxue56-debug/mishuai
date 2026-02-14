"""TTS (Text-to-Speech) module.

Converts text to speech audio using configurable engines:
- VOICEVOX (Japanese, local, free, high quality)
- CosyVoice (Chinese, local, free)
- Edge TTS (multilingual, free, cloud)
"""

import asyncio
import io
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config_loader import get_main_config
from src.utils.logger import get_logger

logger = get_logger("tts")


class TTSEngine:
    """Text-to-Speech engine with multiple backend support."""

    def __init__(self):
        config = get_main_config()
        tts_config = config.get("tts", {})

        self.active_language = tts_config.get("active_language", "japanese")
        self.ja_config = tts_config.get("japanese", {})
        self.zh_config = tts_config.get("chinese", {})

        self._voicevox_url: Optional[str] = None
        self._initialized = False

    async def initialize(self):
        """Initialize TTS engines based on config."""
        if self.active_language == "japanese":
            engine = self.ja_config.get("engine", "voicevox")
            if engine == "voicevox":
                await self._init_voicevox()
            elif engine == "edge_tts":
                logger.info("Edge TTS initialized (Japanese)")
        elif self.active_language == "chinese":
            engine = self.zh_config.get("engine", "edge_tts")
            if engine == "edge_tts":
                logger.info("Edge TTS initialized (Chinese)")

        self._initialized = True

    async def _init_voicevox(self):
        """Initialize VOICEVOX connection."""
        import aiohttp

        vv_config = self.ja_config.get("voicevox", {})
        self._voicevox_url = vv_config.get("host", "http://127.0.0.1:50021")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._voicevox_url}/speakers") as resp:
                    if resp.status == 200:
                        speakers = await resp.json()
                        logger.info(
                            f"VOICEVOX connected. {len(speakers)} speakers available."
                        )
                    else:
                        logger.warning(
                            f"VOICEVOX not responding at {self._voicevox_url}. "
                            "Make sure VOICEVOX is running."
                        )
        except Exception as e:
            logger.warning(
                f"VOICEVOX not available: {e}. "
                f"Start VOICEVOX or use edge_tts as fallback."
            )

    async def synthesize(self, text: str, language: Optional[str] = None) -> Optional[bytes]:
        """Synthesize text to speech audio.

        Args:
            text: Text to convert to speech.
            language: Override language ("japanese" or "chinese").

        Returns:
            Audio data as WAV bytes, or None if synthesis failed.
        """
        if not text.strip():
            return None

        lang = language or self.active_language

        try:
            if lang == "japanese":
                engine = self.ja_config.get("engine", "voicevox")
                if engine == "voicevox":
                    return await self._synthesize_voicevox(text)
                else:
                    return await self._synthesize_edge_tts(text, "ja-JP")
            elif lang == "chinese":
                engine = self.zh_config.get("engine", "edge_tts")
                if engine == "edge_tts":
                    return await self._synthesize_edge_tts(text, "zh-CN")
            else:
                return await self._synthesize_edge_tts(text, "ja-JP")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            # Fallback to Edge TTS
            try:
                logger.info("Falling back to Edge TTS...")
                return await self._synthesize_edge_tts(text, "ja-JP" if lang == "japanese" else "zh-CN")
            except Exception as e2:
                logger.error(f"Edge TTS fallback also failed: {e2}")
                return None

    async def _synthesize_voicevox(self, text: str) -> Optional[bytes]:
        """Synthesize using VOICEVOX.

        VOICEVOX two-step process:
        1. POST /audio_query - Generate audio query from text
        2. POST /synthesis - Generate WAV from audio query
        """
        import aiohttp

        vv_config = self.ja_config.get("voicevox", {})
        speaker_id = vv_config.get("speaker_id", 3)
        speed_scale = vv_config.get("speed_scale", 1.1)
        pitch_scale = vv_config.get("pitch_scale", 0.0)
        intonation_scale = vv_config.get("intonation_scale", 1.2)

        base_url = self._voicevox_url or "http://127.0.0.1:50021"

        async with aiohttp.ClientSession() as session:
            # Step 1: Create audio query
            async with session.post(
                f"{base_url}/audio_query",
                params={"text": text, "speaker": speaker_id},
            ) as resp:
                if resp.status != 200:
                    logger.error(f"VOICEVOX audio_query failed: {resp.status}")
                    return None
                query = await resp.json()

            # Adjust parameters
            query["speedScale"] = speed_scale
            query["pitchScale"] = pitch_scale
            query["intonationScale"] = intonation_scale

            # Step 2: Synthesize audio
            async with session.post(
                f"{base_url}/synthesis",
                params={"speaker": speaker_id},
                json=query,
            ) as resp:
                if resp.status != 200:
                    logger.error(f"VOICEVOX synthesis failed: {resp.status}")
                    return None
                audio_data = await resp.read()
                logger.debug(f"VOICEVOX synthesis OK: {len(audio_data)} bytes")
                return audio_data

    async def _synthesize_edge_tts(self, text: str, lang_code: str) -> Optional[bytes]:
        """Synthesize using Microsoft Edge TTS (free, cloud-based).

        Args:
            text: Text to synthesize.
            lang_code: Language code prefix (e.g., "ja-JP", "zh-CN").
        """
        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return None

        # Select voice based on language
        if lang_code.startswith("ja"):
            config = self.ja_config.get("edge_tts", {})
            voice = config.get("voice", "ja-JP-NanamiNeural")
            rate = config.get("rate", "+10%")
        elif lang_code.startswith("zh"):
            config = self.zh_config.get("edge_tts", {})
            voice = config.get("voice", "zh-CN-XiaoxiaoNeural")
            rate = config.get("rate", "+5%")
        else:
            voice = "ja-JP-NanamiNeural"
            rate = "+10%"

        communicate = edge_tts.Communicate(text, voice, rate=rate)

        # Collect audio data
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if audio_chunks:
            audio_data = b"".join(audio_chunks)
            logger.debug(f"Edge TTS synthesis OK: {len(audio_data)} bytes, voice={voice}")
            return audio_data

        return None

    async def get_audio_for_lip_sync(self, audio_data: bytes) -> list[float]:
        """Extract volume levels from audio for lip sync animation.

        Args:
            audio_data: WAV audio bytes.

        Returns:
            List of volume levels (0.0-1.0) per frame.
        """
        try:
            # Skip WAV header (44 bytes for standard WAV)
            if audio_data[:4] == b"RIFF":
                pcm_data = audio_data[44:]
            else:
                pcm_data = audio_data

            # Convert to numpy array
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

            # Calculate RMS per frame (for lip sync)
            frame_size = 1024  # ~23ms at 44100Hz
            volumes = []
            for i in range(0, len(samples) - frame_size, frame_size):
                frame = samples[i:i + frame_size]
                rms = np.sqrt(np.mean(frame ** 2))
                volume = min(rms / 16384.0, 1.0)  # Normalize
                volumes.append(float(volume))

            return volumes
        except Exception as e:
            logger.error(f"Lip sync extraction error: {e}")
            return []

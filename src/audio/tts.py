"""TTS (Text-to-Speech) module.

Converts text to speech audio using configurable engines:
- VOICEVOX (Japanese, local, free, high quality, emotion via style switching)
- CosyVoice v3 (DashScope API, auto-detects emotion from text context)
- Edge TTS (multilingual, free, cloud)
"""

import asyncio
import io
import os
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

    # VOICEVOX emotion → style ID mapping (ずんだもん / zundamon)
    # Maps LLM emotion tags to VOICEVOX voice styles for expressive speech
    VOICEVOX_EMOTION_STYLES = {
        # ずんだもん (default speaker_id=3)
        "zundamon": {
            "neutral":   3,   # ノーマル
            "happy":     1,   # あまあま (sweet)
            "excited":   1,   # あまあま
            "shy":       22,  # ささやき (whisper)
            "sad":       76,  # なみだめ (tearful)
            "angry":     7,   # ツンツン (tsun)
            "thinking":  38,  # ヒソヒソ (hushed)
            "surprised": 3,   # ノーマル (with higher intonation)
            "confused":  75,  # ヘロヘロ (sluggish)
        },
        # 四国めたん
        "metan": {
            "neutral":   2,
            "happy":     0,   # あまあま
            "excited":   0,
            "shy":       36,  # ささやき
            "sad":       37,  # ヒソヒソ
            "angry":     6,   # ツンツン
            "thinking":  37,
            "surprised": 2,
            "confused":  2,
        },
    }

    def __init__(self):
        config = get_main_config()
        tts_config = config.get("tts", {})

        self.active_language = tts_config.get("active_language", "japanese")
        self.ja_config = tts_config.get("japanese", {})
        self.zh_config = tts_config.get("chinese", {})

        self._voicevox_url: Optional[str] = None
        self._cosyvoice_ready = False
        self._initialized = False
        self._current_emotion: str = "neutral"

    async def initialize(self):
        """Initialize TTS engines based on config."""
        if self.active_language == "japanese":
            engine = self.ja_config.get("engine", "voicevox")
            if engine == "voicevox":
                await self._init_voicevox()
            elif engine == "cosyvoice_dashscope":
                self._init_cosyvoice_dashscope()
            elif engine == "edge_tts":
                logger.info("Edge TTS initialized (Japanese)")
        elif self.active_language == "chinese":
            engine = self.zh_config.get("engine", "edge_tts")
            if engine == "cosyvoice_dashscope":
                self._init_cosyvoice_dashscope()
            elif engine == "edge_tts":
                logger.info("Edge TTS initialized (Chinese)")

        self._initialized = True

    def _init_cosyvoice_dashscope(self):
        """Initialize CosyVoice DashScope API connection."""
        try:
            import dashscope
            api_key = os.environ.get("QIANWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                logger.warning("CosyVoice DashScope: No API key found (QIANWEN_API_KEY or DASHSCOPE_API_KEY)")
                return
            dashscope.api_key = api_key
            self._cosyvoice_ready = True
            cv_config = self.ja_config.get("cosyvoice_dashscope", self.zh_config.get("cosyvoice_dashscope", {}))
            voice = cv_config.get("voice", "loongriko_v3")
            model = cv_config.get("model", "cosyvoice-v3-flash")
            logger.info(f"CosyVoice DashScope initialized (model={model}, voice={voice})")
        except ImportError:
            logger.error("dashscope not installed. Run: pip install dashscope")
        except Exception as e:
            logger.warning(f"CosyVoice DashScope init error: {e}")

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

    async def synthesize(
        self, text: str, language: Optional[str] = None, emotion: Optional[str] = None
    ) -> Optional[bytes]:
        """Synthesize text to speech audio.

        Args:
            text: Text to convert to speech.
            language: Override language ("japanese" or "chinese").
            emotion: Emotion tag from LLM (e.g., "happy", "sad") for VOICEVOX style switching.

        Returns:
            Audio data as WAV/MP3 bytes, or None if synthesis failed.
        """
        if not text.strip():
            return None

        if emotion:
            self._current_emotion = emotion

        lang = language or self.active_language
        lang_code = "ja-JP" if lang == "japanese" else "zh-CN"

        try:
            audio_data = None

            if lang == "japanese":
                engine = self.ja_config.get("engine", "voicevox")
                if engine == "voicevox":
                    audio_data = await self._synthesize_voicevox(text, emotion=self._current_emotion)
                elif engine == "cosyvoice_dashscope":
                    audio_data = await self._synthesize_cosyvoice_dashscope(text, lang)
                else:
                    audio_data = await self._synthesize_edge_tts(text, "ja-JP")
            elif lang == "chinese":
                engine = self.zh_config.get("engine", "edge_tts")
                if engine == "cosyvoice_dashscope":
                    if not self._cosyvoice_ready:
                        self._init_cosyvoice_dashscope()
                    audio_data = await self._synthesize_cosyvoice_dashscope(text, lang)
                elif engine == "edge_tts":
                    audio_data = await self._synthesize_edge_tts(text, "zh-CN")
                else:
                    audio_data = await self._synthesize_edge_tts(text, "zh-CN")
            else:
                audio_data = await self._synthesize_edge_tts(text, "ja-JP")

            # If primary engine returned empty, fallback to Edge TTS
            if not audio_data:
                logger.warning(f"Primary TTS returned empty, falling back to Edge TTS ({lang_code})")
                audio_data = await self._synthesize_edge_tts(text, lang_code)

            return audio_data

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            # Fallback to Edge TTS
            try:
                logger.info("Falling back to Edge TTS...")
                return await self._synthesize_edge_tts(text, lang_code)
            except Exception as e2:
                logger.error(f"Edge TTS fallback also failed: {e2}")
                return None

    def _get_voicevox_style_id(self, emotion: str) -> int:
        """Get VOICEVOX style ID based on current emotion.

        Uses the emotion → style mapping to select the right voice style.
        Falls back to the default speaker_id from config if no mapping found.
        """
        vv_config = self.ja_config.get("voicevox", {})
        default_id = vv_config.get("speaker_id", 3)
        character = vv_config.get("character", "zundamon")

        styles = self.VOICEVOX_EMOTION_STYLES.get(character, {})
        style_id = styles.get(emotion, default_id)
        return style_id

    async def _synthesize_voicevox(
        self, text: str, emotion: str = "neutral"
    ) -> Optional[bytes]:
        """Synthesize using VOICEVOX with emotion-based style switching.

        VOICEVOX two-step process:
        1. POST /audio_query - Generate audio query from text
        2. POST /synthesis - Generate WAV from audio query

        The speaker style ID changes based on the current emotion,
        giving the character different vocal qualities for each mood.
        """
        import aiohttp

        vv_config = self.ja_config.get("voicevox", {})
        speed_scale = vv_config.get("speed_scale", 1.1)
        pitch_scale = vv_config.get("pitch_scale", 0.0)
        intonation_scale = vv_config.get("intonation_scale", 1.2)

        # Select style based on emotion
        speaker_id = self._get_voicevox_style_id(emotion)

        # Adjust prosody parameters based on emotion for extra expressiveness
        emotion_adjustments = {
            "happy":     {"pitchScale": 0.05,  "speedScale": speed_scale * 1.05, "intonationScale": 1.4},
            "excited":   {"pitchScale": 0.08,  "speedScale": speed_scale * 1.1,  "intonationScale": 1.5},
            "sad":       {"pitchScale": -0.05, "speedScale": speed_scale * 0.9,  "intonationScale": 0.8},
            "angry":     {"pitchScale": 0.03,  "speedScale": speed_scale * 1.05, "intonationScale": 1.6},
            "shy":       {"pitchScale": -0.02, "speedScale": speed_scale * 0.95, "intonationScale": 1.0},
            "thinking":  {"pitchScale": 0.0,   "speedScale": speed_scale * 0.9,  "intonationScale": 0.9},
            "surprised": {"pitchScale": 0.1,   "speedScale": speed_scale * 1.1,  "intonationScale": 1.5},
        }
        adj = emotion_adjustments.get(emotion, {})

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

            # Adjust parameters (use emotion-specific or defaults)
            query["speedScale"] = adj.get("speedScale", speed_scale)
            query["pitchScale"] = adj.get("pitchScale", pitch_scale)
            query["intonationScale"] = adj.get("intonationScale", intonation_scale)

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
                logger.info(
                    f"VOICEVOX synthesis OK: {len(audio_data)} bytes, "
                    f"style={speaker_id}, emotion={emotion}"
                )
                return audio_data

    # CosyVoice DashScope AudioFormat mapping
    # Maps config string "format_samplerate" to AudioFormat enum constants
    COSYVOICE_FORMAT_MAP = {
        "wav_22050":  "WAV_22050HZ_MONO_16BIT",
        "wav_24000":  "WAV_24000HZ_MONO_16BIT",
        "wav_16000":  "WAV_16000HZ_MONO_16BIT",
        "wav_44100":  "WAV_44100HZ_MONO_16BIT",
        "wav_48000":  "WAV_48000HZ_MONO_16BIT",
        "mp3_22050":  "MP3_22050HZ_MONO_256KBPS",
        "mp3_24000":  "MP3_24000HZ_MONO_256KBPS",
        "mp3_16000":  "MP3_16000HZ_MONO_128KBPS",
    }

    async def _synthesize_cosyvoice_dashscope(
        self, text: str, language: str = "japanese"
    ) -> Optional[bytes]:
        """Synthesize using CosyVoice v3 via DashScope API.

        CosyVoice v3 auto-detects emotion from text context,
        so no explicit emotion parameter is needed — the model
        naturally adjusts prosody based on content.

        Uses asyncio.run_in_executor() to wrap the blocking SDK call.

        Returns:
            WAV/MP3 audio bytes, or None if failed.
        """
        if not self._cosyvoice_ready:
            logger.warning("CosyVoice DashScope not initialized, falling back to Edge TTS")
            lang_code = "ja-JP" if language == "japanese" else "zh-CN"
            return await self._synthesize_edge_tts(text, lang_code)

        # Get config from the active language section
        if language == "japanese":
            cv_config = self.ja_config.get("cosyvoice_dashscope", {})
        else:
            cv_config = self.zh_config.get("cosyvoice_dashscope", {})

        model = cv_config.get("model", "cosyvoice-v3-flash")
        voice = cv_config.get("voice", "loongriko_v3")
        fmt = cv_config.get("format", "wav")
        sample_rate = cv_config.get("sample_rate", 22050)

        # Resolve AudioFormat enum
        format_key = f"{fmt}_{sample_rate}"

        def _sync_synthesize():
            """Blocking DashScope TTS call."""
            import dashscope
            from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat

            # Look up format constant from AudioFormat enum
            format_name = self.COSYVOICE_FORMAT_MAP.get(format_key)
            audio_fmt = getattr(AudioFormat, format_name) if format_name else AudioFormat.WAV_22050HZ_MONO_16BIT

            synthesizer = SpeechSynthesizer(
                model=model,
                voice=voice,
                format=audio_fmt,
            )
            audio = synthesizer.call(text)
            return audio

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, _sync_synthesize)

            if audio_data and len(audio_data) > 0:
                logger.info(
                    f"CosyVoice DashScope synthesis OK: {len(audio_data)} bytes, "
                    f"model={model}, voice={voice}"
                )
                return audio_data
            else:
                logger.error("CosyVoice DashScope returned empty audio")
                return None
        except Exception as e:
            logger.error(f"CosyVoice DashScope synthesis error: {e}")
            return None

    def set_engine(self, engine: str):
        """Switch TTS engine at runtime.

        Args:
            engine: Engine name - "voicevox", "cosyvoice_dashscope", or "edge_tts"
        """
        if self.active_language == "japanese":
            self.ja_config["engine"] = engine
        elif self.active_language == "chinese":
            self.zh_config["engine"] = engine

        # Initialize new engine if needed
        if engine == "cosyvoice_dashscope" and not self._cosyvoice_ready:
            self._init_cosyvoice_dashscope()

        logger.info(f"TTS engine switched to: {engine} (lang={self.active_language})")

    def get_engine(self) -> str:
        """Get the currently active TTS engine name."""
        if self.active_language == "japanese":
            return self.ja_config.get("engine", "voicevox")
        elif self.active_language == "chinese":
            return self.zh_config.get("engine", "edge_tts")
        return "edge_tts"

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

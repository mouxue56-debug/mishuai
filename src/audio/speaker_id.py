"""Speaker identification using voice print.

Identifies who is speaking based on voice embeddings,
enabling personalized responses based on speaker profile.

Supports two backends:
1. resemblyzer (lightweight, CPU-only, ~50MB model, pip install)
2. 3D-Speaker via modelscope (heavier, GPU-optional, more accurate)

Requires (one of):
    pip install resemblyzer
    pip install modelscope  # for 3D-Speaker
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config_loader import get_main_config, BASE_DIR
from src.utils.logger import get_logger

logger = get_logger("speaker_id")

# Lazy imports
_resemblyzer_encoder = None
_modelscope_pipeline = None


class SpeakerIdentifier:
    """Identifies speakers using voice print embeddings."""

    def __init__(self):
        config = get_main_config()
        speaker_config = config.get("speaker", {})

        self.enabled = speaker_config.get("enabled", True)
        self.threshold = speaker_config.get("similarity_threshold", 0.68)
        self.engine = speaker_config.get("engine", "resemblyzer")
        self.sample_rate = speaker_config.get("sample_rate", 16000)

        self._encoder = None
        self._embeddings: dict[str, np.ndarray] = {}
        self._profiles_loaded = False

    async def initialize(self):
        """Initialize the speaker identification model."""
        if not self.enabled:
            logger.info("Speaker identification disabled")
            return

        try:
            await self._load_model()
            await self._load_registered_speakers()
        except Exception as e:
            logger.warning(f"Speaker ID init failed: {e}. Feature disabled.")
            self.enabled = False

    async def _load_model(self):
        """Load the speaker embedding model."""
        loop = asyncio.get_event_loop()

        if self.engine == "resemblyzer":
            await loop.run_in_executor(None, self._load_resemblyzer)
        elif self.engine == "3d_speaker":
            await loop.run_in_executor(None, self._load_3d_speaker)
        else:
            logger.warning(f"Unknown speaker engine: {self.engine}, trying resemblyzer")
            self.engine = "resemblyzer"
            await loop.run_in_executor(None, self._load_resemblyzer)

    def _load_resemblyzer(self):
        """Load resemblyzer GE2E encoder (lightweight, CPU)."""
        global _resemblyzer_encoder
        try:
            from resemblyzer import VoiceEncoder
            if _resemblyzer_encoder is None:
                _resemblyzer_encoder = VoiceEncoder("cpu")
            self._encoder = _resemblyzer_encoder
            logger.info("Speaker ID model loaded: resemblyzer (GE2E)")
        except ImportError:
            raise ImportError(
                "resemblyzer not installed. Run: pip install resemblyzer"
            )

    def _load_3d_speaker(self):
        """Load 3D-Speaker model via modelscope."""
        global _modelscope_pipeline
        try:
            from modelscope.pipelines import pipeline as ms_pipeline
            if _modelscope_pipeline is None:
                _modelscope_pipeline = ms_pipeline(
                    task='speaker-verification',
                    model='iic/speech_eres2net_sv_zh-cn_16k-common',
                )
            self._encoder = _modelscope_pipeline
            logger.info("Speaker ID model loaded: 3D-Speaker (ERes2Net)")
        except ImportError:
            raise ImportError(
                "modelscope not installed. Run: pip install modelscope"
            )

    async def _load_registered_speakers(self):
        """Load registered speaker embeddings from disk."""
        embeddings_dir = BASE_DIR / "data" / "speaker_embeddings"
        if not embeddings_dir.exists():
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            return

        for npy_file in embeddings_dir.glob("*.npy"):
            speaker_id = npy_file.stem
            try:
                embedding = np.load(str(npy_file))
                self._embeddings[speaker_id] = embedding
                logger.info(f"Loaded speaker embedding: {speaker_id}")
            except Exception as e:
                logger.error(f"Failed to load embedding {npy_file}: {e}")

        self._profiles_loaded = True
        logger.info(f"Loaded {len(self._embeddings)} speaker embeddings")

    async def identify(self, audio_data: bytes) -> Optional[str]:
        """Identify the speaker from audio data.

        Args:
            audio_data: Raw audio bytes (int16, 16kHz).

        Returns:
            Speaker ID string if identified, None otherwise.
        """
        if not self.enabled or not self._embeddings or not self._encoder:
            return None

        try:
            embedding = await self._extract_embedding(audio_data)
            if embedding is None:
                return None

            best_match = None
            best_score = 0.0

            for speaker_id, ref_embedding in self._embeddings.items():
                score = self._cosine_similarity(embedding, ref_embedding)
                if score > best_score:
                    best_score = score
                    best_match = speaker_id

            if best_score >= self.threshold:
                logger.info(f"Speaker identified: {best_match} (score={best_score:.3f})")
                return best_match
            else:
                logger.debug(f"Speaker not identified (best={best_match}, score={best_score:.3f})")
                return None

        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            return None

    async def _extract_embedding(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio data.

        Args:
            audio_data: Raw audio bytes (int16, 16kHz mono).

        Returns:
            Embedding vector as numpy array.
        """
        if not self._encoder:
            return None

        # Convert bytes to float32 waveform
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Need minimum audio length (~1 second)
        if len(audio_array) < self.sample_rate:
            logger.debug("Audio too short for speaker ID")
            return None

        loop = asyncio.get_event_loop()

        if self.engine == "resemblyzer":
            embedding = await loop.run_in_executor(
                None, self._embed_resemblyzer, audio_array
            )
        elif self.engine == "3d_speaker":
            embedding = await loop.run_in_executor(
                None, self._embed_3d_speaker, audio_array
            )
        else:
            return None

        return embedding

    def _embed_resemblyzer(self, audio_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using resemblyzer."""
        try:
            from resemblyzer import preprocess_wav
            processed = preprocess_wav(audio_array, source_sr=self.sample_rate)
            embedding = self._encoder.embed_utterance(processed)
            return embedding
        except Exception as e:
            logger.error(f"Resemblyzer embedding error: {e}")
            return None

    def _embed_3d_speaker(self, audio_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using 3D-Speaker / modelscope."""
        try:
            result = self._encoder(audio_array)
            return np.array(result['spk_embedding'])
        except Exception as e:
            logger.error(f"3D-Speaker embedding error: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    async def register_speaker(self, speaker_id: str, audio_samples: list[bytes]):
        """Register a new speaker with multiple audio samples.

        Args:
            speaker_id: Unique identifier for the speaker.
            audio_samples: List of raw audio byte arrays (int16, 16kHz).
        """
        if not self.enabled or not self._encoder:
            logger.error("Speaker identification is not available")
            return False

        embeddings = []
        for sample in audio_samples:
            emb = await self._extract_embedding(sample)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.error(f"No valid embeddings extracted for {speaker_id}")
            return False

        # Average the embeddings for robustness
        avg_embedding = np.mean(embeddings, axis=0)
        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # Save to disk
        save_dir = BASE_DIR / "data" / "speaker_embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{speaker_id}.npy"
        np.save(str(save_path), avg_embedding)

        # Update in-memory cache
        self._embeddings[speaker_id] = avg_embedding
        logger.info(f"Speaker registered: {speaker_id} ({len(embeddings)}/{len(audio_samples)} samples)")
        return True

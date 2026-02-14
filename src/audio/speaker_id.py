"""Speaker identification using voice print (3D-Speaker).

Identifies who is speaking based on voice embeddings,
enabling personalized responses based on speaker profile.
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config_loader import get_main_config, BASE_DIR
from src.utils.logger import get_logger

logger = get_logger("speaker_id")


class SpeakerIdentifier:
    """Identifies speakers using voice print embeddings.

    Uses 3D-Speaker model to create and compare voice embeddings.
    """

    def __init__(self):
        config = get_main_config()
        speaker_config = config.get("speaker", {})

        self.enabled = speaker_config.get("enabled", True)
        self.threshold = speaker_config.get("similarity_threshold", 0.68)
        self.model_path = speaker_config.get("model_path", "models/speaker/")

        self._model = None
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
        """Load the 3D-Speaker model."""
        try:
            # 3D-Speaker uses modelscope
            # from modelscope.pipelines import pipeline as ms_pipeline
            # self._model = ms_pipeline(
            #     task='speaker-verification',
            #     model='iic/speech_eres2net_sv_zh-cn_16k-common',
            # )
            logger.info("Speaker ID model: ready (will load on first use)")
        except ImportError:
            logger.warning(
                "modelscope not installed for 3D-Speaker. "
                "Run: pip install modelscope"
            )
            self.enabled = False

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
        if not self.enabled or not self._embeddings:
            return None

        try:
            # Extract embedding from audio
            embedding = await self._extract_embedding(audio_data)
            if embedding is None:
                return None

            # Compare with registered speakers
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
            audio_data: Raw audio bytes.

        Returns:
            Embedding vector as numpy array.
        """
        # Placeholder: Real implementation uses 3D-Speaker model
        # audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        # result = self._model(audio_array)
        # return result['spk_embedding']
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
            audio_samples: List of audio byte arrays for enrollment.
        """
        if not self.enabled:
            logger.error("Speaker identification is disabled")
            return

        embeddings = []
        for sample in audio_samples:
            emb = await self._extract_embedding(sample)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.error(f"No valid embeddings extracted for {speaker_id}")
            return

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Save to disk
        save_dir = BASE_DIR / "data" / "speaker_embeddings"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{speaker_id}.npy"
        np.save(str(save_path), avg_embedding)

        # Update in-memory cache
        self._embeddings[speaker_id] = avg_embedding
        logger.info(f"Speaker registered: {speaker_id} ({len(audio_samples)} samples)")

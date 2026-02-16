"""Pydantic-based configuration validation.

Validates config.yaml and persona.yaml on startup to catch errors early.
"""

from typing import Optional
from pydantic import BaseModel, field_validator


class LLMModelConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.8


class LLMTaskConfig(BaseModel):
    primary: Optional[LLMModelConfig] = None
    fallback: Optional[LLMModelConfig] = None
    budget: Optional[LLMModelConfig] = None


class LLMConfig(BaseModel):
    conversation: Optional[LLMTaskConfig] = None
    tool_use: Optional[LLMTaskConfig] = None
    summarization: Optional[LLMTaskConfig] = None
    active_mode: str = "balanced"

    @field_validator("active_mode")
    @classmethod
    def validate_mode(cls, v):
        valid = {"quality", "balanced", "budget"}
        if v not in valid:
            raise ValueError(f"active_mode must be one of {valid}, got '{v}'")
        return v


class AudioInputConfig(BaseModel):
    device_index: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024


class VADConfig(BaseModel):
    mode: int = 3
    frame_duration_ms: int = 30
    silence_threshold_ms: int = 800


class AudioConfig(BaseModel):
    input: Optional[AudioInputConfig] = None
    vad: Optional[VADConfig] = None


class TTSVoicevoxConfig(BaseModel):
    host: str = "http://127.0.0.1:50021"
    speaker_id: int = 3
    speed_scale: float = 1.1


class TTSLanguageConfig(BaseModel):
    engine: str = "cosyvoice_dashscope"
    voicevox: Optional[TTSVoicevoxConfig] = None


class TTSConfig(BaseModel):
    japanese: Optional[TTSLanguageConfig] = None
    chinese: Optional[TTSLanguageConfig] = None
    active_language: str = "japanese"


class WebSocketConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765


class MemoryConfig(BaseModel):
    database_path: str = "data/memory.db"


class AppConfig(BaseModel):
    """Top-level application config schema."""
    app: Optional[dict] = None
    llm: Optional[LLMConfig] = None
    audio: Optional[AudioConfig] = None
    tts: Optional[TTSConfig] = None
    websocket: Optional[WebSocketConfig] = None
    memory: Optional[MemoryConfig] = None
    mcp: Optional[dict] = None
    notion: Optional[dict] = None
    logging: Optional[dict] = None
    asr: Optional[dict] = None
    speaker: Optional[dict] = None


class CharacterConfig(BaseModel):
    name: str = "ミケ"
    role: str = "AI秘書"
    personality: list[str] = []
    languages: Optional[dict] = None
    cat_trivia: list[str] = []


class PersonaConfig(BaseModel):
    """Top-level persona config schema."""
    character: Optional[CharacterConfig] = None
    emotions: Optional[dict] = None
    speaker_profiles: Optional[dict] = None
    time_behavior: Optional[dict] = None


def validate_config(config_dict: dict) -> AppConfig:
    """Validate the main config dictionary.

    Args:
        config_dict: Parsed YAML config.

    Returns:
        Validated AppConfig.

    Raises:
        ValidationError: If config is invalid.
    """
    return AppConfig(**config_dict)


def validate_persona(persona_dict: dict) -> PersonaConfig:
    """Validate the persona config dictionary.

    Args:
        persona_dict: Parsed YAML persona config.

    Returns:
        Validated PersonaConfig.

    Raises:
        ValidationError: If config is invalid.
    """
    return PersonaConfig(**persona_dict)

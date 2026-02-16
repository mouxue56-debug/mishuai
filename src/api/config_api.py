"""Runtime Configuration API.

Handles reading, validating, applying, and persisting config changes
triggered from the frontend settings UI via WebSocket.

Flow:
    Frontend → WS set_config → ConfigAPI.handle_set_config()
        → validate value
        → update _CONFIG_CACHE in-memory
        → apply to live subsystem
        → persist to YAML file
"""

import copy
from typing import Any, Optional

from src.api.env_manager import EnvManager
from src.utils.config_loader import (
    load_config,
    update_config_value,
    _deep_get,
    _deep_set,
)
from src.utils.logger import get_logger

logger = get_logger("config_api")

# Fields that should NEVER be sent to the frontend
_SENSITIVE_FIELDS = {
    "notion.customer_database_id",
    "memory.database_path",
    "logging.file",
}

# Sections that contain only sensitive data (skip entirely)
_SENSITIVE_SECTIONS = {"mcp", "notion"}

# Validation rules: {dotted_path: {type, min, max, options, ...}}
_VALIDATION_RULES: dict[str, dict] = {
    # TTS - VOICEVOX
    "tts.japanese.voicevox.speed_scale": {"type": "number", "min": 0.5, "max": 2.0},
    "tts.japanese.voicevox.pitch_scale": {"type": "number", "min": -0.15, "max": 0.15},
    "tts.japanese.voicevox.intonation_scale": {"type": "number", "min": 0.0, "max": 2.0},
    "tts.japanese.voicevox.speaker_id": {"type": "int", "min": 0, "max": 100},
    "tts.japanese.voicevox.character": {"type": "str", "options": ["zundamon", "metan"]},
    "tts.japanese.engine": {"type": "str", "options": ["voicevox", "cosyvoice_dashscope"]},
    "tts.chinese.engine": {"type": "str", "options": ["cosyvoice_dashscope", "voicevox"]},
    "tts.active_language": {"type": "str", "options": ["japanese", "chinese"]},
    # LLM
    "llm.active_mode": {"type": "str", "options": ["quality", "balanced", "budget"]},
    # LLM model config (conversation)
    "llm.conversation.primary.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.conversation.primary.model": {"type": "str"},
    "llm.conversation.primary.max_tokens": {"type": "int", "min": 128, "max": 8192},
    "llm.conversation.primary.temperature": {"type": "number", "min": 0.0, "max": 2.0},
    "llm.conversation.fallback.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.conversation.fallback.model": {"type": "str"},
    "llm.conversation.fallback.max_tokens": {"type": "int", "min": 128, "max": 8192},
    "llm.conversation.fallback.temperature": {"type": "number", "min": 0.0, "max": 2.0},
    "llm.conversation.budget.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.conversation.budget.model": {"type": "str"},
    "llm.conversation.budget.max_tokens": {"type": "int", "min": 128, "max": 8192},
    "llm.conversation.budget.temperature": {"type": "number", "min": 0.0, "max": 2.0},
    # LLM model config (tool_use)
    "llm.tool_use.primary.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.tool_use.primary.model": {"type": "str"},
    "llm.tool_use.fallback.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.tool_use.fallback.model": {"type": "str"},
    # LLM model config (summarization)
    "llm.summarization.primary.provider": {"type": "str", "options": ["anthropic", "openai", "google", "moonshot", "deepseek", "qianwen"]},
    "llm.summarization.primary.model": {"type": "str"},
    # App
    "app.debug": {"type": "bool"},
    # Logging
    "logging.level": {"type": "str", "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    # Proactive
    "proactive.enabled": {"type": "bool"},
    "proactive.idle_timeout_min": {"type": "int", "min": 1, "max": 120},
    "proactive.cooldown_sec": {"type": "int", "min": 30, "max": 3600},
    # Vision
    "vision.enabled": {"type": "bool"},
    # Memory
    "memory.short_term.max_turns": {"type": "int", "min": 1, "max": 50},
    # Audio
    "audio.wake_word.enabled": {"type": "bool"},
    "audio.wake_word.keyword": {"type": "str"},
    "audio.vad.silence_threshold_ms": {"type": "int", "min": 200, "max": 3000},
    # Speaker
    "speaker.enabled": {"type": "bool"},
}


class ConfigAPI:
    """Runtime configuration management.

    Receives references to live subsystems so it can apply config
    changes without requiring a restart.
    """

    def __init__(self):
        self._tts = None
        self._llm = None
        self._scheduler = None
        self._vision = None
        self._memory = None
        self._env_manager = EnvManager()

    def register_subsystems(self, tts=None, llm=None, scheduler=None,
                            vision=None, memory=None):
        """Register live subsystem references for runtime updates.

        Called once from main.py after all subsystems are initialized.
        """
        self._tts = tts
        self._llm = llm
        self._scheduler = scheduler
        self._vision = vision
        self._memory = memory
        logger.info("ConfigAPI: subsystems registered")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def handle_get_config(self, sections: Optional[list] = None) -> dict:
        """Return current config for the frontend.

        Args:
            sections: List of top-level sections to include.
                      None = all non-sensitive sections.

        Returns:
            Dict with "config" and optional "cache_stats".
        """
        config = load_config("config")
        result = {}

        if sections:
            for section in sections:
                if section not in _SENSITIVE_SECTIONS:
                    val = config.get(section)
                    if val is not None:
                        result[section] = copy.deepcopy(val)
        else:
            for key, val in config.items():
                if key not in _SENSITIVE_SECTIONS:
                    result[key] = copy.deepcopy(val)

        # Remove sensitive fields
        for field in _SENSITIVE_FIELDS:
            parts = field.split(".", 1)
            if len(parts) == 2 and parts[0] in result:
                sub = result[parts[0]]
                if isinstance(sub, dict):
                    sub.pop(parts[1], None)

        # Attach cache stats if LLM router is available
        cache_stats = None
        if self._llm:
            try:
                cache_stats = self._llm.get_cache_stats()
            except Exception:
                pass

        return {
            "config": result,
            "cache_stats": cache_stats,
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def handle_set_config(self, updates: list[dict]) -> dict:
        """Apply a batch of config updates.

        Each update: {"section": "tts.japanese.voicevox", "key": "speed_scale", "value": 1.3}

        Returns:
            Dict with "success", "applied", "errors".
        """
        applied = []
        errors = []

        for update in updates:
            section = update.get("section", "")
            key = update.get("key", "")
            value = update.get("value")

            if not section or not key:
                errors.append({
                    "section": section, "key": key, "value": value,
                    "error": "section and key are required",
                })
                continue

            full_path = f"{section}.{key}" if section else key

            # Validate
            error = self._validate(full_path, value)
            if error:
                errors.append({
                    "section": section, "key": key, "value": value,
                    "error": error,
                })
                continue

            try:
                # Update in-memory cache + persist to YAML
                await update_config_value("config", full_path, value)

                # Apply to running subsystem
                self._apply_to_runtime(full_path, value)

                applied.append({"section": section, "key": key, "value": value})

            except Exception as e:
                logger.error(f"Failed to apply config {full_path}={value}: {e}")
                errors.append({
                    "section": section, "key": key, "value": value,
                    "error": str(e),
                })

        success = len(errors) == 0
        return {"success": success, "applied": applied, "errors": errors}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    async def handle_clear_cache(self) -> dict:
        """Clear the LLM response cache."""
        if self._llm:
            self._llm.clear_cache()
            return {"success": True, "message": "LLM cache cleared"}
        return {"success": False, "message": "LLM router not available"}

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    async def handle_get_persona(self) -> dict:
        """Return the full persona.yaml config for the frontend.

        Returns:
            Dict with "persona" containing the persona config.
        """
        try:
            persona = load_config("persona")
            return {"success": True, "persona": copy.deepcopy(persona)}
        except Exception as e:
            logger.error(f"Failed to load persona: {e}")
            return {"success": False, "persona": {}, "error": str(e)}

    async def handle_set_persona(self, updates: list[dict]) -> dict:
        """Apply persona config updates.

        Each update: {"path": "character.personality", "value": [...]}

        Returns:
            Dict with "success", "applied", "errors".
        """
        applied = []
        errors = []

        for update in updates:
            path = update.get("path", "")
            value = update.get("value")

            if not path:
                errors.append({"path": path, "error": "path is required"})
                continue

            try:
                await update_config_value("persona", path, value)
                applied.append({"path": path, "value": value})
            except Exception as e:
                logger.error(f"Failed to update persona {path}: {e}")
                errors.append({"path": path, "error": str(e)})

        # Invalidate the LLM router's cached system prompt so it
        # rebuilds with updated persona on next conversation
        if self._llm and applied:
            if hasattr(self._llm, "_system_prompt"):
                self._llm._system_prompt = None
                logger.info("LLM system prompt cache invalidated (persona changed)")

        success = len(errors) == 0
        return {"success": success, "applied": applied, "errors": errors}

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------

    async def handle_get_api_key_status(self) -> dict:
        """Return masked API key status for all providers.

        Returns:
            Dict with "keys" containing status per provider.
        """
        try:
            keys = self._env_manager.get_key_status()
            return {"success": True, "keys": keys}
        except Exception as e:
            logger.error(f"Failed to get API key status: {e}")
            return {"success": False, "keys": {}, "error": str(e)}

    async def handle_set_api_key(self, provider: str, key: str) -> dict:
        """Set an API key for a provider.

        Args:
            provider: Provider name (e.g., "anthropic").
            key: The API key value.

        Returns:
            Dict with success status.
        """
        try:
            result = self._env_manager.set_key(provider, key)
            return result
        except Exception as e:
            logger.error(f"Failed to set API key for {provider}: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, full_path: str, value: Any) -> Optional[str]:
        """Validate a single config value.

        Returns:
            Error message string, or None if valid.
        """
        rule = _VALIDATION_RULES.get(full_path)
        if not rule:
            # No specific rule — allow (for flexibility)
            return None

        expected_type = rule.get("type")

        if expected_type == "number":
            if not isinstance(value, (int, float)):
                return f"Expected number, got {type(value).__name__}"
            if "min" in rule and value < rule["min"]:
                return f"Must be >= {rule['min']}"
            if "max" in rule and value > rule["max"]:
                return f"Must be <= {rule['max']}"

        elif expected_type == "int":
            if not isinstance(value, int):
                return f"Expected integer, got {type(value).__name__}"
            if "min" in rule and value < rule["min"]:
                return f"Must be >= {rule['min']}"
            if "max" in rule and value > rule["max"]:
                return f"Must be <= {rule['max']}"

        elif expected_type == "str":
            if not isinstance(value, str):
                return f"Expected string, got {type(value).__name__}"
            if "options" in rule and value not in rule["options"]:
                return f"Must be one of: {rule['options']}"

        elif expected_type == "bool":
            if not isinstance(value, bool):
                return f"Expected boolean, got {type(value).__name__}"

        return None

    # ------------------------------------------------------------------
    # Runtime Application
    # ------------------------------------------------------------------

    def _apply_to_runtime(self, full_path: str, value: Any):
        """Apply a config change to the running subsystem.

        This is where we map config paths to live object mutations.
        """
        # --- LLM ---
        if full_path == "llm.active_mode" and self._llm:
            from src.core.llm_router import LLMMode
            self._llm.set_mode(LLMMode(value))
            logger.info(f"Runtime: LLM mode → {value}")

        # --- TTS engine ---
        elif full_path == "tts.active_language" and self._tts:
            self._tts.active_language = value
            logger.info(f"Runtime: TTS language → {value}")

        elif full_path.endswith(".engine") and full_path.startswith("tts.") and self._tts:
            # e.g., tts.japanese.engine = "voicevox"
            self._tts.set_engine(value)
            logger.info(f"Runtime: TTS engine → {value}")

        # --- TTS VOICEVOX params ---
        elif full_path.startswith("tts.japanese.voicevox.") and self._tts:
            key = full_path.split(".")[-1]
            if hasattr(self._tts, "ja_config") and self._tts.ja_config:
                voicevox_cfg = self._tts.ja_config.get("voicevox", {})
                voicevox_cfg[key] = value
                logger.info(f"Runtime: VOICEVOX {key} → {value}")

        elif full_path.startswith("tts.chinese.") and self._tts:
            # Update zh_config for Chinese TTS params
            parts = full_path.split(".")
            if len(parts) >= 4 and hasattr(self._tts, "zh_config") and self._tts.zh_config:
                engine = parts[2]  # e.g., "cosyvoice_dashscope"
                key = parts[3]
                engine_cfg = self._tts.zh_config.get(engine, {})
                engine_cfg[key] = value
                logger.info(f"Runtime: TTS Chinese {engine}.{key} → {value}")

        # --- Logging ---
        elif full_path == "logging.level":
            import logging
            level = getattr(logging, value.upper(), logging.INFO)
            logging.getLogger().setLevel(level)
            logger.info(f"Runtime: Log level → {value}")

        # --- Proactive ---
        elif full_path.startswith("proactive.") and self._scheduler:
            key = full_path.split(".")[-1]
            if hasattr(self._scheduler, key):
                setattr(self._scheduler, key, value)
                logger.info(f"Runtime: Proactive {key} → {value}")

        # --- Vision ---
        elif full_path == "vision.enabled" and self._vision:
            self._vision.enabled = value
            logger.info(f"Runtime: Vision enabled → {value}")

        # --- Memory ---
        elif full_path == "memory.short_term.max_turns" and self._memory:
            if hasattr(self._memory, "short_term"):
                self._memory.short_term.max_turns = value
                logger.info(f"Runtime: Short-term max_turns → {value}")

        else:
            # No runtime action needed — config is persisted and will
            # take effect on next subsystem access or restart.
            logger.debug(f"Config persisted (no runtime action): {full_path} = {value}")

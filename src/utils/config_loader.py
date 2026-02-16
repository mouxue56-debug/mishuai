"""Configuration loader for YAML config files.

Supports reading (cached) and writing (comment-preserving via ruamel.yaml).
"""

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_CONFIG_CACHE: dict[str, Any] = {}
_SAVE_LOCK = asyncio.Lock()

# Lazy logger to avoid circular import (logger.py imports config_loader)
_logger = None

def _get_logger():
    global _logger
    if _logger is None:
        from src.utils.logger import get_logger
        _logger = get_logger("config")
    return _logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DATA_DIR = BASE_DIR / "data"


def load_env():
    """Load environment variables from .env file."""
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)


def load_config(config_name: str = "config") -> dict[str, Any]:
    """Load a YAML config file from config/ directory.

    Args:
        config_name: Name of the config file without extension.

    Returns:
        Parsed YAML as a dictionary.
    """
    if config_name in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_name]

    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE[config_name] = config
    return config


def validate_configs() -> tuple[bool, list[str]]:
    """Validate config.yaml and persona.yaml using Pydantic schemas.

    Returns:
        Tuple of (success, list of error messages).
    """
    from src.utils.config_validator import validate_config, validate_persona

    errors = []

    try:
        config = load_config("config")
        validate_config(config)
    except FileNotFoundError:
        errors.append("config.yaml not found")
    except Exception as e:
        errors.append(f"config.yaml validation: {e}")

    try:
        persona = load_config("persona")
        validate_persona(persona)
    except FileNotFoundError:
        errors.append("persona.yaml not found")
    except Exception as e:
        errors.append(f"persona.yaml validation: {e}")

    return (len(errors) == 0, errors)


def get_main_config() -> dict[str, Any]:
    """Load the main config.yaml."""
    return load_config("config")


def get_persona_config() -> dict[str, Any]:
    """Load the persona.yaml."""
    return load_config("persona")


def get_api_key(service: str) -> str:
    """Get an API key from environment variables.

    Args:
        service: Service name (e.g., 'anthropic', 'openai', 'google', 'moonshot')

    Returns:
        The API key string.

    Raises:
        ValueError: If the API key is not set.
    """
    load_env()
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qianwen": "QIANWEN_API_KEY",
        "notion": "NOTION_API_KEY",
        "line": "LINE_CHANNEL_ACCESS_TOKEN",
    }
    env_var = key_map.get(service, f"{service.upper()}_API_KEY")
    value = os.getenv(env_var, "")
    if not value:
        raise ValueError(f"API key not set: {env_var}. Please set it in .env file.")
    return value


def get_knowledge_path(filename: str) -> Path:
    """Get path to a knowledge base file."""
    return KNOWLEDGE_DIR / filename


def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


# --- Config Write-back ---

def _deep_set(d: dict, dotted_path: str, value: Any):
    """Set a value in a nested dict using a dotted path.

    Example:
        _deep_set(config, "tts.japanese.voicevox.speed_scale", 1.3)
    """
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _deep_get(d: dict, dotted_path: str, default=None) -> Any:
    """Get a value from a nested dict using a dotted path."""
    keys = dotted_path.split(".")
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


async def save_config(config_name: str = "config"):
    """Write the cached config back to its YAML file.

    Uses ruamel.yaml to preserve comments and formatting.
    Falls back to PyYAML if ruamel.yaml is unavailable.
    """
    async with _SAVE_LOCK:
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        data = _CONFIG_CACHE.get(config_name)
        if data is None:
            _get_logger().warning(f"No cached config for '{config_name}', skipping save")
            return

        try:
            from ruamel.yaml import YAML
            yaml_rt = YAML()
            yaml_rt.preserve_quotes = True
            yaml_rt.width = 120

            # Load original file to preserve comments
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    original = yaml_rt.load(f)
                # Deep-merge cached values into the comment-preserving structure
                _ruamel_deep_merge(original, data)
                target = original
            else:
                target = data

            with open(config_path, "w", encoding="utf-8") as f:
                yaml_rt.dump(target, f)

            _get_logger().info(f"Config saved: {config_path} (ruamel.yaml, comments preserved)")

        except ImportError:
            # Fallback: PyYAML (loses comments)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                          sort_keys=False)
            _get_logger().warning(f"Config saved: {config_path} (PyYAML fallback, comments lost)")


def _ruamel_deep_merge(target, source):
    """Merge source dict into ruamel.yaml CommentedMap, preserving comments."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _ruamel_deep_merge(target[key], value)
        else:
            target[key] = value


async def update_config_value(
    config_name: str, dotted_path: str, value: Any
) -> dict:
    """Update a single value in the cached config and persist to YAML.

    Args:
        config_name: Config file name ("config" or "persona").
        dotted_path: Dotted path like "tts.japanese.voicevox.speed_scale".
        value: New value to set.

    Returns:
        The updated full config dict.
    """
    config = load_config(config_name)
    _deep_set(config, dotted_path, value)
    await save_config(config_name)
    _get_logger().info(f"Config updated: {config_name}.{dotted_path} = {value}")
    return config


def invalidate_cache(config_name: str = None):
    """Invalidate config cache (all or a specific config).

    Args:
        config_name: If specified, only invalidate this config. Otherwise all.
    """
    if config_name:
        _CONFIG_CACHE.pop(config_name, None)
    else:
        _CONFIG_CACHE.clear()

"""Configuration loader for YAML config files."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_CONFIG_CACHE: dict[str, Any] = {}

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

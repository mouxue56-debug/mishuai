"""Secure .env file management for API keys.

Reads and writes API keys from the project .env file.
Keys are never returned in full — only masked versions are sent to the frontend.
"""

import os
import re
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("env_manager")

# Project root .env file
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"

# Provider → environment variable name
PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qianwen": "QIANWEN_API_KEY",
}

# Provider display names (Japanese)
PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI (GPT)",
    "google": "Google (Gemini)",
    "moonshot": "Moonshot (Kimi)",
    "deepseek": "DeepSeek",
    "qianwen": "Qianwen (通義千問)",
}


class EnvManager:
    """Safe .env file reader/writer for API keys."""

    def __init__(self, env_path: Optional[Path] = None):
        self.env_path = env_path or _ENV_PATH

    def get_key_status(self) -> dict:
        """Return masked key status for all providers.

        Returns:
            Dict of {provider: {is_set, masked, label}}.
        """
        result = {}
        for provider, env_var in PROVIDER_KEY_MAP.items():
            key = os.environ.get(env_var, "")
            result[provider] = {
                "is_set": bool(key and key.strip()),
                "masked": self._mask_key(key) if key.strip() else "",
                "label": PROVIDER_LABELS.get(provider, provider),
                "env_var": env_var,
            }
        return result

    def set_key(self, provider: str, key: str) -> dict:
        """Set an API key for a provider.

        Writes to .env file and updates os.environ.

        Args:
            provider: Provider name (e.g., "anthropic").
            key: The API key value.

        Returns:
            Dict with success status and masked key.
        """
        if provider not in PROVIDER_KEY_MAP:
            return {
                "success": False,
                "error": f"Unknown provider: {provider}",
            }

        key = key.strip()
        if not key:
            return {
                "success": False,
                "error": "API key cannot be empty",
            }

        env_var = PROVIDER_KEY_MAP[provider]

        try:
            # Read existing .env content
            lines = self._read_env_lines()

            # Find and replace or append
            found = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(f"{env_var}=") or stripped.startswith(f"{env_var} ="):
                    lines[i] = f"{env_var}={key}\n"
                    found = True
                    break

            if not found:
                # Append to end
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append(f"{env_var}={key}\n")

            # Write back
            self._write_env_lines(lines)

            # Update runtime environment
            os.environ[env_var] = key

            logger.info(f"API key set for {provider} ({env_var})")
            return {
                "success": True,
                "provider": provider,
                "masked": self._mask_key(key),
            }

        except Exception as e:
            logger.error(f"Failed to set API key for {provider}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _mask_key(self, key: str) -> str:
        """Mask an API key for display.

        Shows recognizable prefix and last 4 characters.
        Examples:
            "sk-ant-api03-abcdefghijk" → "sk-ant-****hijk"
            "AIzaSyABC123" → "AIza****C123"
            "sk-1234567890abcdef" → "sk-1****cdef"
        """
        if not key or len(key) < 8:
            return "****"

        # Find a natural prefix (up to first 6 chars or before a long run)
        # Common patterns: "sk-ant-", "sk-", "AIza", etc.
        prefix_match = re.match(r'^([a-zA-Z]{2,4}[-_]?[a-zA-Z]{0,4}[-_]?)', key)
        if prefix_match:
            prefix = prefix_match.group(1)
            if len(prefix) > 6:
                prefix = prefix[:6]
        else:
            prefix = key[:4]

        last4 = key[-4:]
        return f"{prefix}****{last4}"

    def _read_env_lines(self) -> list[str]:
        """Read .env file as lines, preserving comments and order."""
        if not self.env_path.exists():
            return []
        with open(self.env_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def _write_env_lines(self, lines: list[str]):
        """Write lines back to .env file."""
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

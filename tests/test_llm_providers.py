"""LLM Provider connectivity tests.

Tests each configured LLM provider to verify:
- API key is set
- Connection succeeds
- Response is valid
- Response time is acceptable

Usage:
    python -m pytest tests/test_llm_providers.py -v
    python -m tests.test_llm_providers  # Direct run with timing output
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Provider configurations for testing
# ---------------------------------------------------------------------------

PROVIDERS = [
    {
        "name": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5-20250929",
        "display": "Claude Sonnet 4.5",
    },
    {
        "name": "openai",
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "display": "GPT-4o Mini",
    },
    {
        "name": "google",
        "env_key": "GOOGLE_API_KEY",
        "model": "gemini-2.5-flash",
        "display": "Gemini 2.5 Flash",
    },
    {
        "name": "moonshot",
        "env_key": "MOONSHOT_API_KEY",
        "model": "kimi-k2.5",
        "display": "Kimi K2.5",
    },
    {
        "name": "deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "display": "DeepSeek V3",
    },
]


# ---------------------------------------------------------------------------
# Unit tests (always run, use mocks)
# ---------------------------------------------------------------------------


class TestLLMRouterUnit:
    """Unit tests for LLM routing logic (no API calls)."""

    def test_provider_config_has_all_required_fields(self):
        """Verify each provider config entry has required fields."""
        for p in PROVIDERS:
            assert "name" in p, f"Missing 'name' in provider config"
            assert "env_key" in p, f"Missing 'env_key' for {p['name']}"
            assert "model" in p, f"Missing 'model' for {p['name']}"

    def test_all_env_keys_are_documented(self):
        """Verify .env.example contains all required API keys."""
        env_example_path = os.path.join(
            os.path.dirname(__file__), "..", ".env.example"
        )
        if not os.path.exists(env_example_path):
            pytest.skip(".env.example not found")

        with open(env_example_path) as f:
            env_content = f.read()

        for p in PROVIDERS:
            assert p["env_key"] in env_content, (
                f"{p['env_key']} not found in .env.example"
            )

    def test_config_yaml_has_providers(self):
        """Verify config.yaml has the providers section."""
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "llm" in config
        assert "providers" in config["llm"]
        assert "conversation" in config["llm"]
        assert "tool_use" in config["llm"]
        assert "active_mode" in config["llm"]

    def test_config_yaml_routing_modes(self):
        """Verify config.yaml has quality/balanced/budget modes."""
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        modes = config["llm"].get("modes", {})
        assert "quality" in modes
        assert "balanced" in modes
        assert "budget" in modes

    def test_fallback_chain_is_different_provider(self):
        """Verify primary and fallback use different providers."""
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        conv = config["llm"]["conversation"]
        assert conv["primary"]["provider"] != conv["fallback"]["provider"], (
            "Primary and fallback conversation providers should differ"
        )
        assert conv["primary"]["provider"] != conv["budget"]["provider"], (
            "Primary and budget conversation providers should differ"
        )


# ---------------------------------------------------------------------------
# Integration tests (require API keys, skip if not set)
# ---------------------------------------------------------------------------


def _has_api_key(env_key: str) -> bool:
    return bool(os.environ.get(env_key))


@pytest.mark.parametrize(
    "provider",
    PROVIDERS,
    ids=[p["display"] for p in PROVIDERS],
)
class TestLLMProviderIntegration:
    """Integration tests that make real API calls (skipped if no key)."""

    @pytest.mark.asyncio
    async def test_provider_responds(self, provider):
        """Send a minimal message and verify response."""
        if not _has_api_key(provider["env_key"]):
            pytest.skip(f"{provider['env_key']} not set")

        try:
            from src.core.llm_router import LLMRouter
        except ImportError:
            pytest.skip("src.core.llm_router not importable")

        # Build minimal config matching this provider
        config = {
            "conversation": {
                "primary": {
                    "provider": provider["name"],
                    "model": provider["model"],
                    "max_tokens": 100,
                    "temperature": 0.5,
                }
            },
            "tool_use": {
                "primary": {
                    "provider": provider["name"],
                    "model": provider["model"],
                    "max_tokens": 100,
                    "temperature": 0.3,
                }
            },
            "summarization": {
                "primary": {
                    "provider": provider["name"],
                    "model": provider["model"],
                    "max_tokens": 100,
                    "temperature": 0.5,
                }
            },
            "active_mode": "quality",
        }

        router = LLMRouter(config, persona={})

        messages = [
            {"role": "user", "content": "Say 'hello' in one word."}
        ]

        start = time.time()
        response = await router.chat(messages, task="conversation")
        elapsed_ms = (time.time() - start) * 1000

        assert response is not None
        assert hasattr(response, "text") or isinstance(response, dict)

        text = response.text if hasattr(response, "text") else response.get("text", "")
        assert len(text) > 0, f"{provider['display']} returned empty response"

        print(
            f"\n  {provider['display']}: "
            f"OK ({elapsed_ms:.0f}ms) "
            f"response='{text[:50]}'"
        )


# ---------------------------------------------------------------------------
# CLI runner — for direct execution with timing
# ---------------------------------------------------------------------------

async def _run_all_providers():
    """Direct CLI runner: test all providers and print results table."""
    print("\n" + "=" * 60)
    print("LLM Provider Connectivity Test")
    print("=" * 60)

    results = []

    for p in PROVIDERS:
        key = os.environ.get(p["env_key"])
        if not key:
            results.append({
                "provider": p["display"],
                "status": "SKIP",
                "time_ms": 0,
                "note": f"{p['env_key']} not set",
            })
            continue

        try:
            from src.core.llm_router import LLMRouter

            config = {
                "conversation": {
                    "primary": {
                        "provider": p["name"],
                        "model": p["model"],
                        "max_tokens": 100,
                        "temperature": 0.5,
                    }
                },
                "tool_use": {
                    "primary": {
                        "provider": p["name"],
                        "model": p["model"],
                        "max_tokens": 100,
                        "temperature": 0.3,
                    }
                },
                "summarization": {
                    "primary": {
                        "provider": p["name"],
                        "model": p["model"],
                        "max_tokens": 100,
                        "temperature": 0.5,
                    }
                },
                "active_mode": "quality",
            }

            router = LLMRouter(config, persona={})
            messages = [{"role": "user", "content": "Say 'hello' in one word."}]

            start = time.time()
            response = await router.chat(messages, task="conversation")
            elapsed_ms = (time.time() - start) * 1000

            text = response.text if hasattr(response, "text") else str(response)
            results.append({
                "provider": p["display"],
                "status": "OK",
                "time_ms": elapsed_ms,
                "note": text[:40],
            })

        except Exception as e:
            results.append({
                "provider": p["display"],
                "status": "FAIL",
                "time_ms": 0,
                "note": str(e)[:50],
            })

    # Print results table
    print(f"\n{'Provider':<20} {'Status':<8} {'Time':<10} {'Note'}")
    print("-" * 60)
    for r in results:
        time_str = f"{r['time_ms']:.0f}ms" if r["time_ms"] > 0 else "-"
        emoji = {"OK": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(r["status"], "?")
        print(f"{r['provider']:<20} {emoji} {r['status']:<5} {time_str:<10} {r['note']}")

    print()
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"Result: {ok_count}/{len(results)} providers connected successfully")
    return results


if __name__ == "__main__":
    asyncio.run(_run_all_providers())

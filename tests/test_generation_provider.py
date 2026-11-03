"""Provider config ve `.env` sırrı çözümlemesini sınar."""

import json

import pytest

from src.generation.providers import OpenAICompatibleChatProvider, load_chat_provider


def test_api_provider_reads_key_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KVKK_TEST_API_KEY", raising=False)
    (tmp_path / ".env").write_text("KVKK_TEST_API_KEY=local-secret\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "generation.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "model": "test-model",
                "base_url": "https://api.example.test/v1",
                "api_key_env": "KVKK_TEST_API_KEY",
            }
        ),
        encoding="utf-8",
    )

    provider = load_chat_provider(config_path)

    assert isinstance(provider, OpenAICompatibleChatProvider)
    assert provider.api_key == "local-secret"
    monkeypatch.delenv("KVKK_TEST_API_KEY", raising=False)


def test_api_provider_requires_configured_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KVKK_MISSING_API_KEY", raising=False)
    config_path = tmp_path / "generation.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "model": "test-model",
                "base_url": "https://api.example.test/v1",
                "api_key_env": "KVKK_MISSING_API_KEY",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KVKK_MISSING_API_KEY"):
        load_chat_provider(config_path)


def test_provider_base_url_can_be_overridden_by_environment(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "generation.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": "local-model",
                "base_url": "http://localhost:11434",
                "base_url_env": "KVKK_TEST_OLLAMA_URL",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KVKK_TEST_OLLAMA_URL", "http://ollama.internal:11434")

    provider = load_chat_provider(config_path)

    assert provider.base_url == "http://ollama.internal:11434"

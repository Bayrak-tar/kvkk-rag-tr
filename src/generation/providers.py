"""Ollama ve OpenAI uyumlu API'ler için sohbet sağlayıcıları."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen


class ChatProvider(Protocol):
    """Sistem ve kullanıcı mesajlarından tek bir metin yanıtı üretir."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Sohbet sağlayıcısını çağırıp yanıt metnini döndürür."""


def _load_dotenv(path: Path) -> None:
    """Basit KEY=VALUE `.env` satırlarını mevcut ortamı ezmeden yükler."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaChatProvider:
    """Ollama'nın yerel `/api/chat` endpoint'ini çağırır."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        think: bool | None = None,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.think = think
        self.keep_alive = keep_alive
        self.options = options or {}

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Ollama'dan stream kapalı yanıt alır."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": self.options,
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = _post_json(
            f"{self.base_url}/api/chat",
            payload,
            headers={},
            timeout=self.timeout,
        )
        return str(response["message"]["content"]).strip()


class OpenAICompatibleChatProvider:
    """OpenAI Chat Completions biçimini destekleyen API'leri çağırır."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Uyumlu `/chat/completions` API'sinden yanıt metnini alır."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = _post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers=headers,
            timeout=self.timeout,
        )
        return str(response["choices"][0]["message"]["content"]).strip()


def load_chat_provider(path: Path) -> ChatProvider:
    """Sohbet sağlayıcısını JSON config ve gerektiğinde ortam değişkeniyle kurar."""
    _load_dotenv(path.resolve().parents[1] / ".env")
    config = json.loads(path.read_text(encoding="utf-8"))
    provider = config["provider"]
    base_url_env = config.get("base_url_env")
    common = {
        "base_url": os.getenv(base_url_env, config["base_url"])
        if base_url_env
        else config["base_url"],
        "model": config["model"],
        "timeout": float(config.get("timeout_seconds", 120)),
    }
    if provider == "ollama":
        ollama_options = {
            key: config[key]
            for key in ("think", "keep_alive", "options")
            if key in config
        }
        return OllamaChatProvider(**common, **ollama_options)
    if provider == "openai-compatible":
        key_name = config.get("api_key_env")
        api_key = os.getenv(key_name) if key_name else None
        if key_name and not api_key:
            raise ValueError(f"API anahtarı {key_name} ortam değişkeninde bulunamadı.")
        return OpenAICompatibleChatProvider(**common, api_key=api_key)
    raise ValueError(f"Desteklenmeyen generation sağlayıcısı: {provider}")

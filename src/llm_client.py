from __future__ import annotations

import requests

from src.config import settings


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None, timeout: int = 120) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str | None = None, model: str | None = None) -> str:
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

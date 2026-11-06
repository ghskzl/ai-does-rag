import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama

from src.config import Settings
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama.host
        self.timeout = httpx.Timeout(float(settings.ollama.timeout))

    async def health_check(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "version": version_data.get("version", "unknown"),
                    }
                raise OllamaException(f"Ollama returned status {response.status_code}")
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {e}")

    def get_langchain_model(
        self,
        model: str,
        temperature: float = 0.0,
        format: Optional[str] = None,
    ) -> ChatOllama:
        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "base_url": self.base_url,
        }
        if format:
            kwargs["format"] = format
        return ChatOllama(**kwargs)

from typing import Optional

from src.config import Settings, get_settings

from .base import BaseEmbeddingsClient
from .bailian_client import BailianEmbeddingsClient
from .ollama_embeddings_client import OllamaEmbeddingsClient


def make_embeddings_service(settings: Optional[Settings] = None) -> BaseEmbeddingsClient:
    if settings is None:
        settings = get_settings()

    provider = settings.embedding.provider

    if provider == "bailian":
        api_key = settings.embedding.bailian_api_key
        if not api_key:
            raise ValueError("EMBEDDING__BAILIAN_API_KEY is required")
        return BailianEmbeddingsClient(
            api_key=api_key,
            model=settings.embedding.bailian_model,
            base_url=settings.embedding.bailian_base_url,
            dimensions=settings.embedding.dimensions,
        )

    if provider == "ollama":
        return OllamaEmbeddingsClient(
            model=settings.embedding.ollama_model,
            base_url=settings.ollama.host,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")

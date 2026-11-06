import logging
from typing import List

import httpx

from src.services.embeddings.base import BaseEmbeddingsClient

logger = logging.getLogger(__name__)


class OllamaEmbeddingsClient(BaseEmbeddingsClient):
    def __init__(
        self,
        model: str = "qwen3-embedding",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        logger.info("Ollama embeddings client initialized (model=%s)", model)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        response = await self.client.post(
            f"{self.base_url}/api/embed",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]

    async def embed_passages(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        result = await self._embed([query])
        return result[0]

    async def close(self):
        await self.client.aclose()

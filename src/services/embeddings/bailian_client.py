import logging
from typing import Any, Dict, List

import httpx

from src.services.embeddings.base import BaseEmbeddingsClient

logger = logging.getLogger(__name__)


class BailianEmbeddingsClient(BaseEmbeddingsClient):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.dimensions = dimensions
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info("Bailian embeddings client initialized (model=%s)", model)

    async def _embed(self, texts: List[str], task: str = "document") -> List[List[float]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        response = await self.client.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        sorted_data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._embed(batch, task="document")
            embeddings.extend(batch_embeddings)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        result = await self._embed([query], task="query")
        return result[0]

    async def close(self):
        await self.client.aclose()

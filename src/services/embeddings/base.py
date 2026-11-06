from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingsClient(ABC):
    @abstractmethod
    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]: ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]: ...

    @abstractmethod
    async def close(self): ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

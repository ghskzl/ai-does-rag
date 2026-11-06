from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
else:
    try:
        from fastapi import Depends, Request
    except ImportError:
        pass

from src.config import Settings
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.factory import make_agentic_rag_service
from src.services.cache.client import CacheClient
from src.services.documents.factory import make_ingestion_service
from src.services.documents.ingestion_service import DocumentIngestionService
from src.services.embeddings.base import BaseEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_embeddings_service(request: Request) -> BaseEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state."""
    return request.app.state.ollama_client


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


SettingsDep = Annotated[Settings, Depends(get_settings)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
EmbeddingsDep = Annotated[BaseEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]


def get_agentic_rag_service(
    opensearch: OpenSearchDep,
    ollama: OllamaDep,
    embeddings: EmbeddingsDep,
    langfuse: LangfuseDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticRAGService:
    """Get agentic RAG service."""
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        ollama_client=ollama,
        embeddings_client=embeddings,
        langfuse_tracer=langfuse,
    )


AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]


def get_ingestion_service(
    opensearch: OpenSearchClient = Depends(get_opensearch_client),
    embeddings: BaseEmbeddingsClient = Depends(get_embeddings_service),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestionService:
    """Get document ingestion service."""
    return make_ingestion_service(
        opensearch_client=opensearch,
        embeddings_client=embeddings,
        settings=settings,
    )


IngestionDep = Annotated[DocumentIngestionService, Depends(get_ingestion_service)]

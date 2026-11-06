from src.services.documents.ingestion_service import DocumentIngestionService
from src.services.embeddings.base import BaseEmbeddingsClient
from src.config import Settings
from src.services.opensearch.client import OpenSearchClient


def make_ingestion_service(
    opensearch_client: OpenSearchClient,
    embeddings_client: BaseEmbeddingsClient,
    settings: Settings,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        settings=settings,
    )

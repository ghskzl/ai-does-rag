from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    if settings is None:
        settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)

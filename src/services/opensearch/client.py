import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from src.config import Settings
from src.services.opensearch.index_config import DOCS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from src.services.opensearch.query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"
        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=5,
            max_retries=1,
            retry_on_timeout=True,
        )
        logger.info(f"OpenSearch client initialized: {host}")

    def health_check(self) -> bool:
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}
            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]
            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_hybrid_index(self, force: bool = False) -> bool:
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index: {self.index_name}")
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=DOCS_CHUNKS_MAPPING)
                logger.info(f"Created index: {self.index_name}")
                return True
            logger.info(f"Index already exists: {self.index_name}")
            return False
        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index already exists (race condition): {self.index_name}")
                return False
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]
            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                except Exception:
                    pass
            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }
            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)
            logger.info(f"Created RRF pipeline: {pipeline_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        try:
            if not query_embedding or not use_hybrid:
                return self._search_bm25(query=query, size=size, from_=from_)
            return self._search_hybrid(query=query, query_embedding=query_embedding, size=size, min_score=min_score)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25(self, query: str, size: int, from_: int) -> Dict[str, Any]:
        builder = QueryBuilder(query=query, size=size, from_=from_, search_chunks=True)
        search_body = builder.build()
        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_hits(response)

    def _search_hybrid(
        self, query: str, query_embedding: List[float], size: int, min_score: float
    ) -> Dict[str, Any]:
        builder = QueryBuilder(query=query, size=size * 2, from_=0, search_chunks=True)
        bm25_body = builder.build()
        bm25_query = bm25_body["query"]

        hybrid_query = {
            "hybrid": {
                "queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}]
            }
        }
        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_body["_source"],
            "highlight": bm25_body["highlight"],
        }
        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )
        return self._format_hits(response, min_score=min_score)

    def _format_hits(self, response: Dict, min_score: float = 0.0) -> Dict[str, Any]:
        results: Dict[str, Any] = {"total": response["hits"]["total"]["value"], "hits": []}
        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]
            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]
            results["hits"].append(chunk)
        results["total"] = len(results["hits"])
        return results

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        try:
            chunk_data["embedding"] = embedding
            response = self.client.index(index=self.index_name, body=chunk_data, refresh=True)
            return response["result"] in ["created", "updated"]
        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def delete_document(self, doc_id: str) -> int:
        try:
            response = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"doc_id": doc_id}}},
                refresh=True,
            )
            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for document '{doc_id}'")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {e}")
            return 0

    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {"match_all": {}},
                    "_source": ["doc_id", "title", "source", "chunk_word_count", "created_at", "updated_at"],
                    "size": 10000,
                },
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        from opensearchpy import helpers

        actions = []
        for chunk in chunks:
            chunk_data = chunk["chunk_data"].copy()
            chunk_data["embedding"] = chunk["embedding"]
            actions.append({"_index": self.index_name, "_source": chunk_data})
        success, failed = helpers.bulk(self.client, actions, refresh=True)
        return {"success": success, "failed": len(failed)}

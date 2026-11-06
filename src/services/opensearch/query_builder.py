import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        search_chunks: bool = True,
    ):
        self.query = query
        self.size = size
        self.from_ = from_
        self.search_chunks = search_chunks
        self.fields = fields or ["chunk_text^3", "title^2"]

    def build(self) -> Dict[str, Any]:
        return {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": True,
            "_source": {"excludes": ["embedding"]},
            "highlight": {
                "fields": {
                    "chunk_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "title": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            },
        }

    def _build_query(self) -> Dict[str, Any]:
        must_clauses = []
        if self.query.strip():
            must_clauses.append({
                "multi_match": {
                    "query": self.query,
                    "fields": self.fields,
                    "type": "best_fields",
                    "operator": "or",
                    "fuzziness": "AUTO",
                    "prefix_length": 2,
                }
            })
        if not must_clauses:
            must_clauses = [{"match_all": {}}]
        return {"bool": {"must": must_clauses}}

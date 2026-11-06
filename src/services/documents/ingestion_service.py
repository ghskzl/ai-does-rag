import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ChunkingSettings, Settings
from src.services.documents.chunker import TextChunk, TextChunker
from src.services.documents.document_parser import RawChunk, get_document_parser
from src.services.embeddings.base import BaseEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


@dataclass
class IngestedDocument:
    doc_id: str
    title: str
    source: str
    chunk_count: int
    total_words: int
    created_at: str
    updated_at: str = ""


@dataclass
class IngestionResult:
    doc_id: str
    title: str
    source: str
    chunks_created: int
    total_words: int
    status: str
    error: Optional[str] = None


class DocumentIngestionService:
    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        embeddings_client: BaseEmbeddingsClient,
        settings: Settings,
    ):
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.settings = settings
        self.chunker = TextChunker(settings.chunking)
        self.upload_dir = Path(settings.documents.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def parse_file(self, file_path: Path) -> List[RawChunk]:
        parser = get_document_parser(file_path)
        return parser.parse(file_path)

    def chunk_raw(self, raw_chunks: List[RawChunk]) -> List[TextChunk]:
        return self.chunker.chunk_document(raw_chunks)

    async def embed_and_index_chunks(
        self,
        chunks: List[TextChunk],
        doc_id: str,
        title: str,
        source: str,
    ) -> int:
        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embeddings.embed_passages(chunk_texts)

        now = datetime.now(timezone.utc).isoformat()
        emb = self.settings.embedding
        embedding_model = emb.bailian_model if emb.provider == "bailian" else emb.ollama_model

        bulk_items: List[Dict[str, Any]] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_data = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.text,
                "chunk_word_count": chunk.word_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "title": title,
                "source": source,
                "section_title": chunk.section_title,
                "embedding_model": embedding_model,
                "created_at": now,
                "updated_at": now,
            }
            bulk_items.append({"chunk_data": chunk_data, "embedding": embedding})

        result = self.opensearch.bulk_index_chunks(bulk_items)
        logger.info(f"Indexed {result['success']} chunks for doc '{doc_id}'")
        return result["success"]

    def delete_document(self, doc_id: str) -> int:
        return self.opensearch.delete_document(doc_id)

    def list_documents(self) -> List[IngestedDocument]:
        docs = self.opensearch.get_all_documents()
        doc_map: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            did = doc.get("doc_id", "unknown")
            if did not in doc_map:
                doc_map[did] = {
                    "doc_id": did,
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "chunk_count": 0,
                    "total_words": 0,
                    "created_at": doc.get("created_at", ""),
                    "updated_at": doc.get("updated_at", ""),
                }
            doc_map[did]["chunk_count"] += 1
            doc_map[did]["total_words"] += doc.get("chunk_word_count", 0)

        return [IngestedDocument(**v) for v in doc_map.values()]


def generate_doc_id(filename: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9\-_]", "_", Path(filename).stem)[:50]
    return f"{timestamp}_{safe_name}"

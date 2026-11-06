import logging
from dataclasses import dataclass, field
from typing import List

from src.config import ChunkingSettings
from src.services.documents.document_parser import RawChunk

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    section_title: str = ""
    page_number: int = 0
    metadata: dict = field(default_factory=dict)


class TextChunker:
    def __init__(self, settings: ChunkingSettings):
        self.chunk_size = settings.chunk_size
        self.overlap_size = settings.overlap_size
        self.min_chunk_size = settings.min_chunk_size
        self.section_based = settings.section_based

    def chunk_document(self, raw_chunks: List[RawChunk]) -> List[TextChunk]:
        if self.section_based:
            return self._section_based_chunk(raw_chunks)
        return self._flat_chunk(raw_chunks)

    def _section_based_chunk(self, raw_chunks: List[RawChunk]) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        chunk_index = 0

        for raw_chunk in raw_chunks:
            text = raw_chunk.text.strip()
            if not text:
                continue

            if len(text) <= self.chunk_size:
                if len(text) >= self.min_chunk_size:
                    chunks.append(
                        TextChunk(
                            text=text,
                            chunk_index=chunk_index,
                            start_char=raw_chunk.start_char,
                            end_char=raw_chunk.end_char,
                            word_count=len(text.split()),
                            section_title=raw_chunk.section_title,
                            page_number=raw_chunk.page_number,
                        )
                    )
                    chunk_index += 1
                continue

            sub_chunks = self._split_text(text, raw_chunk.start_char)
            for sub_text, start, end in sub_chunks:
                if len(sub_text) >= self.min_chunk_size:
                    chunks.append(
                        TextChunk(
                            text=sub_text,
                            chunk_index=chunk_index,
                            start_char=start,
                            end_char=end,
                            word_count=len(sub_text.split()),
                            section_title=raw_chunk.section_title,
                            page_number=raw_chunk.page_number,
                        )
                    )
                    chunk_index += 1

        logger.info(f"Section-based chunking produced {len(chunks)} chunks")
        return chunks

    def _flat_chunk(self, raw_chunks: List[RawChunk]) -> List[TextChunk]:
        full_text = "\n".join(rc.text for rc in raw_chunks if rc.text.strip())
        if not full_text.strip():
            return []

        splits = self._split_text(full_text, raw_chunks[0].start_char if raw_chunks else 0)
        chunks: List[TextChunk] = []
        for i, (text, start, end) in enumerate(splits):
            if len(text) >= self.min_chunk_size:
                chunks.append(
                    TextChunk(
                        text=text,
                        chunk_index=i,
                        start_char=start,
                        end_char=end,
                        word_count=len(text.split()),
                    )
                )

        logger.info(f"Flat chunking produced {len(chunks)} chunks")
        return chunks

    def _split_text(self, text: str, base_offset: int) -> List[tuple]:
        results: List[tuple] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos
            chunk_text = text[start:end].strip()
            if chunk_text:
                results.append((chunk_text, base_offset + start, base_offset + end))
            start = end - self.overlap_size if end < len(text) else end
        return results

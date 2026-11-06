import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pymupdf

logger = logging.getLogger(__name__)


class RawChunk:
    __slots__ = ("text", "page_number", "section_title", "start_char", "end_char")

    def __init__(
        self,
        text: str,
        page_number: int = 0,
        section_title: str = "",
        start_char: int = 0,
        end_char: int = 0,
    ):
        self.text = text
        self.page_number = page_number
        self.section_title = section_title
        self.start_char = start_char
        self.end_char = end_char


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> List[RawChunk]: ...


class PDFDocumentParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> List[RawChunk]:
        doc = pymupdf.open(str(file_path))
        try:
            chunks: List[RawChunk] = []
            current_offset = 0
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if not text or not text.strip():
                    current_offset += len(text)
                    continue
                chunks.append(
                    RawChunk(
                        text=text.strip(),
                        page_number=page_num + 1,
                        section_title=f"Page {page_num + 1}",
                        start_char=current_offset,
                        end_char=current_offset + len(text),
                    )
                )
                current_offset += len(text)
            logger.info(f"Parsed PDF '{file_path.name}': {len(chunks)} pages extracted")
            return chunks
        finally:
            doc.close()


class MarkdownDocumentParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> List[RawChunk]:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return []

        chunks: List[RawChunk] = []
        current_section = ""
        current_text_lines: List[str] = []
        current_start = 0
        offset = 0

        for line in content.split("\n"):
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                if current_text_lines:
                    text = "\n".join(current_text_lines).strip()
                    if text:
                        chunks.append(
                            RawChunk(
                                text=text,
                                section_title=current_section,
                                start_char=current_start,
                                end_char=current_start + len(text),
                            )
                        )
                current_section = heading_match.group(2).strip()
                current_text_lines = []
                current_start = offset
            else:
                current_text_lines.append(line)
            offset += len(line) + 1

        if current_text_lines:
            text = "\n".join(current_text_lines).strip()
            if text:
                chunks.append(
                    RawChunk(
                        text=text,
                        section_title=current_section,
                        start_char=current_start,
                        end_char=current_start + len(text),
                    )
                )

        logger.info(f"Parsed Markdown '{file_path.name}': {len(chunks)} sections extracted")
        return chunks


class PlainTextParser(BaseDocumentParser):
    def parse(self, file_path: Path) -> List[RawChunk]:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            return []
        return [RawChunk(text=content.strip(), start_char=0, end_char=len(content))]


def get_document_parser(file_path: Path) -> BaseDocumentParser:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFDocumentParser()
    if suffix in (".md", ".markdown"):
        return MarkdownDocumentParser()
    if suffix in (".txt", ".rst", ".text"):
        return PlainTextParser()
    raise ValueError(f"Unsupported file type: {suffix}")

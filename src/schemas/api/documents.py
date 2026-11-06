from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    doc_id: str
    title: str
    source: str
    chunks_created: int
    total_words: int
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "20260106120000_attention_is_all_you_need",
                "title": "Attention Is All You Need",
                "source": "attention.pdf",
                "chunks_created": 15,
                "total_words": 8500,
                "status": "success",
            }
        }


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    source: str
    chunk_count: int
    total_words: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    status: str


class ErrorResponse(BaseModel):
    detail: str

import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from src.dependencies import IngestionDep
from src.schemas.api.documents import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from src.services.documents.ingestion_service import generate_doc_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".rst"}


def _check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    service: IngestionDep,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = _check_extension(file.filename)
    doc_id = generate_doc_id(file.filename)
    doc_title = title or Path(file.filename).stem

    upload_path = service.upload_dir / f"{doc_id}{ext}"

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        raw_chunks = service.parse_file(upload_path)
        if not raw_chunks:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")

        text_chunks = service.chunk_raw(raw_chunks)
        if not text_chunks:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Document too short to create meaningful chunks")

        chunks_count = await service.embed_and_index_chunks(
            chunks=text_chunks,
            doc_id=doc_id,
            title=doc_title,
            source=file.filename,
        )
        total_words = sum(c.word_count for c in text_chunks)

        logger.info(f"Document uploaded: {doc_id} ({chunks_count} chunks, {total_words} words)")

        return DocumentUploadResponse(
            doc_id=doc_id,
            title=doc_title,
            source=file.filename,
            chunks_created=chunks_count,
            total_words=total_words,
            status="success",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {type(e).__name__}: {str(e)}")


@router.get("/", response_model=DocumentListResponse)
def list_documents(service: IngestionDep):
    docs = service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                doc_id=d.doc_id,
                title=d.title,
                source=d.source,
                chunk_count=d.chunk_count,
                total_words=d.total_words,
                created_at=d.created_at,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, service: IngestionDep):
    docs = service.list_documents()
    for d in docs:
        if d.doc_id == doc_id:
            return DocumentInfo(
                doc_id=d.doc_id,
                title=d.title,
                source=d.source,
                chunk_count=d.chunk_count,
                total_words=d.total_words,
                created_at=d.created_at,
            )
    raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(doc_id: str, service: IngestionDep):
    try:
        deleted = service.delete_document(doc_id)

        for ext in ALLOWED_EXTENSIONS:
            file_path = service.upload_dir / f"{doc_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                break

        logger.info(f"Document deleted: {doc_id} ({deleted} chunks removed)")

        return DocumentDeleteResponse(
            doc_id=doc_id,
            chunks_deleted=deleted,
            status="success",
        )
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

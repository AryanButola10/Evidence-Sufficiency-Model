"""
api/routers/documents.py

Document upload and management endpoints.

Endpoints:
    POST /api/documents/upload  — Upload a PDF, extract + chunk it
    GET  /api/documents         — List all uploaded documents
    GET  /api/documents/{id}    — Get summary for one document
    GET  /api/documents/{id}/chunks — Get all chunks for a document
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.dependencies import InMemoryDocumentStore, get_document_store
from api.schemas.response_schemas import (
    ChunkListResponse,
    DocumentChunkResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from config import get_logger
from src.document_processing.pdf_extractor import PDFExtractor

logger = get_logger(__name__)

router = APIRouter()

# One shared extractor instance (stateless — safe to share)
_extractor = PDFExtractor()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a PDF document",
    description=(
        "Upload a PDF file. The system will:\n"
        "1. Extract text page-by-page using PyMuPDF\n"
        "2. Clean and normalize the text\n"
        "3. Split into sentence-aware overlapping chunks (400 words, 80 word overlap)\n"
        "4. Store chunks with metadata for retrieval\n\n"
        "Returns document statistics and a unique document ID."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and process."),
    store: InMemoryDocumentStore = Depends(get_document_store),
) -> DocumentUploadResponse:
    """
    Upload a PDF and process it into evidence chunks.

    This endpoint drives the document processing pipeline:
    PDFExtractor → TextCleaner → SentenceAwareChunker → InMemoryStore
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are accepted. Received: '{file.filename}'",
        )

    logger.info(
        "Upload request received — file='{file}', size={size}",
        file=file.filename,
        size=file.size or "unknown",
    )

    # Read file bytes
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {e}",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Process the PDF
    try:
        processed = _extractor.process_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            source_metadata={"upload_source": "api"},
        )
    except ValueError as e:
        # User error (bad PDF, encrypted, etc.)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        # Processing failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Store the processed document
    store.add_document(processed)

    avg_words = round(processed.total_words / max(processed.total_chunks, 1), 1)

    logger.info(
        "Document stored — id={id}, chunks={chunks}",
        id=processed.document_id[:8],
        chunks=processed.total_chunks,
    )

    return DocumentUploadResponse(
        document_id=processed.document_id,
        document_name=processed.document_name,
        total_pages=processed.total_pages,
        total_chunks=processed.total_chunks,
        total_words=processed.total_words,
        avg_words_per_chunk=avg_words,
        message=(
            f"Successfully processed '{processed.document_name}': "
            f"{processed.total_chunks} evidence chunks created from "
            f"{processed.total_pages} pages."
        ),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents(
    store: InMemoryDocumentStore = Depends(get_document_store),
) -> DocumentListResponse:
    """Return a list of all documents currently in the store."""
    docs = store.list_documents()
    return DocumentListResponse(
        total_documents=len(docs),
        documents=[
            DocumentListItem(
                document_id=d.document_id,
                document_name=d.document_name,
                total_pages=d.total_pages,
                total_chunks=d.total_chunks,
                total_words=d.total_words,
            )
            for d in docs
        ],
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get all chunks for a document",
    description="Returns all evidence chunks extracted from a document. Useful for inspecting chunking quality.",
)
async def get_document_chunks(
    document_id: str,
    store: InMemoryDocumentStore = Depends(get_document_store),
) -> ChunkListResponse:
    """Return all chunks for a given document ID."""
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found. Upload it first.",
        )

    return ChunkListResponse(
        document_id=doc.document_id,
        document_name=doc.document_name,
        total_chunks=doc.total_chunks,
        chunks=[
            DocumentChunkResponse(**chunk.to_dict())
            for chunk in doc.chunks
        ],
    )

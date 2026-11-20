"""
api/schemas/response_schemas.py
Pydantic response models for all API endpoints.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class DocumentChunkResponse(BaseModel):
    """A single evidence chunk as returned by the API."""
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    page_number: Optional[int]
    start_char: int
    end_char: int
    word_count: int
    source_metadata: dict


class DocumentUploadResponse(BaseModel):
    """Response after successfully uploading and processing a PDF."""
    document_id: str = Field(description="Unique ID assigned to this document.")
    document_name: str = Field(description="Original filename.")
    total_pages: int = Field(description="Total pages in the PDF.")
    total_chunks: int = Field(description="Total evidence chunks created.")
    total_words: int = Field(description="Total words across all chunks.")
    avg_words_per_chunk: float = Field(description="Average chunk size in words.")
    message: str = Field(description="Human-readable status message.")


class DocumentListItem(BaseModel):
    """Summary of a document in the document store."""
    document_id: str
    document_name: str
    total_pages: int
    total_chunks: int
    total_words: int


class DocumentListResponse(BaseModel):
    """List of all uploaded documents."""
    total_documents: int
    documents: list[DocumentListItem]


class ChunkListResponse(BaseModel):
    """All chunks for a specific document."""
    document_id: str
    document_name: str
    total_chunks: int
    chunks: list[DocumentChunkResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    phase: str
    llm_provider: str
    active_model: str
    qdrant_mode: str
    components_active: dict

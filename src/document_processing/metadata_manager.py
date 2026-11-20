"""
src/document_processing/metadata_manager.py

DocumentChunk — the core data model for all evidence in this system.

Every piece of text that the retrieval, evidence evaluation, and ESM
modules work with is a DocumentChunk. This model is the contract between:
    - Document processing (creates chunks)
    - Retrieval (stores and fetches chunks)
    - Evidence evaluation (scores chunks against a query)
    - Provenance mapping (links answers back to chunks)

Design notes:
    - Uses Python dataclasses for lightweight, type-safe data containers
    - chunk_id is a UUID — stable and unique across the whole system
    - word_count is precomputed at creation time (not recomputed on access)
    - source_metadata is a flexible dict for any extra document properties
      (author, date, URL, domain, etc.) — used by source reliability scoring
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentChunk:
    """
    A single text chunk extracted from a document.

    This is the fundamental unit of evidence in the RAG pipeline.
    Every retrieved passage, every evidence score, and every citation
    maps back to a DocumentChunk.

    Attributes:
        chunk_id:        Unique identifier (UUID4 string).
        document_id:     ID of the parent document this chunk came from.
        document_name:   Original filename (e.g., "climate_report.pdf").
        chunk_index:     Position of this chunk within the document (0-based).
        text:            The actual chunk content — what gets retrieved.
        page_number:     PDF page number (1-based). None if not applicable.
        start_char:      Character offset where this chunk starts in the
                         full document text. Used for precise provenance.
        end_char:        Character offset where this chunk ends.
        word_count:      Number of whitespace-delimited words in the chunk.
        source_metadata: Flexible dict for document-level metadata
                         (author, date, url, domain, trust_score, etc.).
                         Used by the source reliability scorer in Phase 4.
    """

    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    page_number: Optional[int]
    start_char: int
    end_char: int
    word_count: int
    source_metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        document_id: str,
        document_name: str,
        chunk_index: int,
        text: str,
        page_number: Optional[int] = None,
        start_char: int = 0,
        end_char: int = 0,
        source_metadata: Optional[dict] = None,
    ) -> "DocumentChunk":
        """
        Factory method to create a DocumentChunk with auto-generated ID
        and precomputed word count.

        Use this instead of the constructor directly — it handles ID
        generation and word counting automatically.

        Args:
            document_id:     UUID of the parent document.
            document_name:   Original filename.
            chunk_index:     Position in document (0-based).
            text:            Chunk text content.
            page_number:     PDF page number (1-based), or None.
            start_char:      Character start offset in full document.
            end_char:        Character end offset in full document.
            source_metadata: Optional dict of document-level metadata.

        Returns:
            A new DocumentChunk instance.
        """
        return cls(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            document_name=document_name,
            chunk_index=chunk_index,
            text=text,
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            word_count=len(text.split()),
            source_metadata=source_metadata or {},
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary (for JSON responses and storage)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_count": self.word_count,
            "source_metadata": self.source_metadata,
        }

    def preview(self, max_chars: int = 120) -> str:
        """Return a short preview of the chunk text for logging."""
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars].rstrip() + "…"

    def __repr__(self) -> str:
        return (
            f"DocumentChunk("
            f"id={self.chunk_id[:8]}…, "
            f"doc='{self.document_name}', "
            f"idx={self.chunk_index}, "
            f"page={self.page_number}, "
            f"words={self.word_count})"
        )


@dataclass
class ProcessedDocument:
    """
    Result of processing a single document through the pipeline.

    Contains the document-level metadata and all extracted chunks.
    Returned by the PDF extractor and stored in the document store.
    """

    document_id: str
    document_name: str
    total_pages: int
    total_chunks: int
    total_words: int
    chunks: list[DocumentChunk]
    source_metadata: dict = field(default_factory=dict)

    @classmethod
    def from_chunks(
        cls,
        document_id: str,
        document_name: str,
        chunks: list[DocumentChunk],
        total_pages: int = 0,
        source_metadata: Optional[dict] = None,
    ) -> "ProcessedDocument":
        """Create a ProcessedDocument by aggregating chunk statistics."""
        return cls(
            document_id=document_id,
            document_name=document_name,
            total_pages=total_pages,
            total_chunks=len(chunks),
            total_words=sum(c.word_count for c in chunks),
            chunks=chunks,
            source_metadata=source_metadata or {},
        )

    def to_summary_dict(self) -> dict:
        """Serialise document-level info without chunk texts (for list endpoints)."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "total_words": self.total_words,
            "source_metadata": self.source_metadata,
        }

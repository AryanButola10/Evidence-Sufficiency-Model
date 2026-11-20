"""
api/dependencies.py

Shared FastAPI dependencies — in-memory document store for Phase 2.

In Phase 3, the DocumentStore will be replaced by Qdrant (vector DB)
for persistent storage. This in-memory version is for development only.

Usage in routers:
    from api.dependencies import get_document_store
    store: DocumentStore = Depends(get_document_store)
"""
from __future__ import annotations
from typing import Optional
from src.document_processing.metadata_manager import ProcessedDocument, DocumentChunk


class InMemoryDocumentStore:
    """
    Thread-safe in-memory store for processed documents.

    Stores ProcessedDocument objects keyed by document_id.
    This will be replaced by Qdrant in Phase 3.
    """

    def __init__(self) -> None:
        self._documents: dict[str, ProcessedDocument] = {}

    def add_document(self, doc: ProcessedDocument) -> None:
        self._documents[doc.document_id] = doc

    def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        return self._documents.get(document_id)

    def list_documents(self) -> list[ProcessedDocument]:
        return list(self._documents.values())

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        doc = self._documents.get(document_id)
        return doc.chunks if doc else []

    def get_all_chunks(self) -> list[DocumentChunk]:
        chunks = []
        for doc in self._documents.values():
            chunks.extend(doc.chunks)
        return chunks

    @property
    def total_documents(self) -> int:
        return len(self._documents)

    @property
    def total_chunks(self) -> int:
        return sum(len(doc.chunks) for doc in self._documents.values())


# Singleton store — lives for the duration of the server process
_document_store = InMemoryDocumentStore()


def get_document_store() -> InMemoryDocumentStore:
    """FastAPI dependency that returns the shared document store."""
    return _document_store

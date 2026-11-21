"""
src/document_processing/pdf_extractor.py

PDF text extraction using PyMuPDF (fitz).

PyMuPDF is chosen because:
    - Excellent Apple Silicon support (native ARM binary)
    - Preserves text reading order better than pdfplumber/pdfminer
    - Page-level extraction with metadata (page number, dimensions)
    - Handles multi-column layouts with configurable sort order
    - Fast: processes 100-page documents in under 2 seconds

What this module does NOT do:
    - OCR (images-only PDFs are not supported in Phase 2)
    - Table extraction (not needed for text-based evidence)
    - Formula rendering (LaTeX/math equations are extracted as text)

Usage:
    extractor = PDFExtractor()
    result = extractor.process_pdf(
        pdf_bytes=file_bytes,
        filename="paper.pdf",
    )
    print(result.total_chunks, "chunks from", result.total_pages, "pages")
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pymupdf # PyMuPDF

from config import get_logger
from src.document_processing.chunker import SentenceAwareChunker, ChunkingConfig
from src.document_processing.metadata_manager import ProcessedDocument
from src.document_processing.text_cleaner import clean_text

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF files and produces ProcessedDocument instances.

    This is the entry point for all document processing. It orchestrates:
        1. PDF opening (from bytes or file path)
        2. Page-by-page text extraction
        3. Text cleaning (via text_cleaner)
        4. Chunking (via SentenceAwareChunker)
        5. Metadata assembly (via ProcessedDocument)
    """

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None) -> None:
        """
        Initialize the PDF extractor.

        Args:
            chunking_config: Optional chunking parameters.
                             Defaults to ChunkingConfig() — 400 word chunks
                             with 80 word overlap.
        """
        self.chunker = SentenceAwareChunker(chunking_config)
        logger.debug("PDFExtractor initialized")

    def process_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        source_metadata: Optional[dict] = None,
    ) -> ProcessedDocument:
        """
        Process a PDF from raw bytes into a ProcessedDocument with chunks.

        This is the main method called by the API upload endpoint.

        Args:
            pdf_bytes:       Raw PDF file content (from file upload).
            filename:        Original filename (used for display and citation).
            source_metadata: Optional metadata dict (author, url, domain, etc.).
                             Stored on every chunk for source reliability scoring.

        Returns:
            ProcessedDocument containing all chunks and document statistics.

        Raises:
            ValueError: If the PDF cannot be opened or is password-protected.
            RuntimeError: If text extraction completely fails.
        """
        document_id = str(uuid.uuid4())
        logger.info(
            "Processing PDF — id={id}, file='{file}', size={size:.1f} KB",
            id=document_id[:8],
            file=filename,
            size=len(pdf_bytes) / 1024,
        )

        # Build source metadata including document identity
        full_metadata = {
            "document_id": document_id,
            "filename": filename,
            **(source_metadata or {}),
        }

        # Open PDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            raise ValueError(
                f"Cannot open PDF '{filename}'. "
                f"File may be corrupted or password-protected. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{filename}': {e}") from e

        if doc.is_encrypted:
            doc.close()
            raise ValueError(
                f"PDF '{filename}' is password-protected. "
                "Please provide an unencrypted PDF."
            )

        total_pages = len(doc)
        logger.info(
            "PDF opened — id={id}, pages={pages}",
            id=document_id[:8],
            pages=total_pages,
        )

        # Extract text page by page
        pages: list[tuple[int, str]] = []  # (page_number_1based, cleaned_text)
        empty_pages: list[int] = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1  # Convert to 1-based

            # Extract text with layout preservation
            # "text" mode: simple text extraction with line breaks
            # "blocks" mode (alternative): extracts text blocks with coordinates
            raw_text = page.get_text("text")

            cleaned = clean_text(raw_text)

            if not cleaned or len(cleaned.split()) < 5:
                empty_pages.append(page_number)
                logger.debug(
                    "Page {page} has no usable text — skipping",
                    page=page_number,
                )
                continue

            pages.append((page_number, cleaned))

        doc.close()

        if not pages:
            raise RuntimeError(
                f"No text could be extracted from '{filename}'. "
                "The PDF may contain only images. "
                "OCR support is planned for a future phase."
            )

        if empty_pages:
            logger.warning(
                "'{file}': {n} page(s) had no extractable text: {pages}",
                file=filename,
                n=len(empty_pages),
                pages=empty_pages[:10],  # Don't log too many
            )

        # Chunk all pages
        chunks = self.chunker.chunk_pages(
            pages=pages,
            document_id=document_id,
            document_name=filename,
            source_metadata=full_metadata,
        )

        if not chunks:
            raise RuntimeError(
                f"Text was extracted from '{filename}' but produced no valid chunks. "
                f"The document may contain only very short text segments."
            )

        result = ProcessedDocument.from_chunks(
            document_id=document_id,
            document_name=filename,
            chunks=chunks,
            total_pages=total_pages,
            source_metadata=full_metadata,
        )

        logger.info(
            "PDF processing complete — id={id}, pages={pages}, "
            "chunks={chunks}, words={words}",
            id=document_id[:8],
            pages=total_pages,
            chunks=result.total_chunks,
            words=result.total_words,
        )

        return result

    def process_pdf_file(
        self,
        file_path: str | Path,
        source_metadata: Optional[dict] = None,
    ) -> ProcessedDocument:
        """
        Process a PDF from a file path (convenience wrapper).

        Useful for batch processing and evaluation scripts.

        Args:
            file_path:       Path to the PDF file.
            source_metadata: Optional metadata dict.

        Returns:
            ProcessedDocument with all chunks.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        pdf_bytes = path.read_bytes()
        return self.process_pdf(
            pdf_bytes=pdf_bytes,
            filename=path.name,
            source_metadata=source_metadata,
        )

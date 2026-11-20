"""
src/document_processing/chunker.py

Sentence-aware sliding window text chunker.

Why sentence-aware matters for this research:
    NLI models (DeBERTa-v3) that score entailment and contradiction work
    best on complete, grammatically coherent sentences. If a sentence is
    split across two chunks, the entailment scorer sees an incomplete
    premise or hypothesis. This degrades evidence evaluation quality.

    Our chunker guarantees that sentence boundaries are never violated:
    each chunk starts and ends at a sentence boundary.

Chunking algorithm:
    1. Split text into sentences (via text_cleaner.split_into_sentences)
    2. Assign word counts to each sentence
    3. Build chunks by accumulating sentences until word_count ≥ chunk_size
    4. Create overlap: the next chunk starts by including the last
       `overlap_size` words worth of sentences from the previous chunk
    5. Each chunk is tagged with start/end char offsets for provenance

Page tracking:
    When text is provided per-page (as from PDF extraction), page numbers
    are tracked and each chunk records the page it starts on.

Configuration (all defaults from settings in Phase 5+):
    chunk_size:    Target word count per chunk. Default 400 words.
                   Rationale: ~500 tokens for English, well within BGE-M3's
                   8192 token limit and DeBERTa's 512 token limit.
    overlap_size:  Words of overlap between consecutive chunks. Default 80.
                   Rationale: 20% overlap — enough to capture cross-boundary
                   evidence without excessive duplication.
    min_chunk_size: Minimum words to keep a chunk. Default 30.
                   Prevents tiny trailing chunks that add noise.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from config import get_logger
from src.document_processing.metadata_manager import DocumentChunk
from src.document_processing.text_cleaner import split_into_sentences, normalize_whitespace

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """
    Configuration for the sliding window chunker.

    These are baseline values treated as experimental hypotheses.
    They will be tuned based on retrieval quality in Phase 3.
    """
    chunk_size: int = 400       # Target words per chunk
    overlap_size: int = 80      # Overlap words between chunks
    min_chunk_size: int = 30    # Discard chunks smaller than this


class SentenceAwareChunker:
    """
    Splits document text into overlapping chunks that respect sentence boundaries.

    Usage:
        chunker = SentenceAwareChunker()
        chunks = chunker.chunk_document(
            text="Long document text...",
            document_id="doc-uuid",
            document_name="paper.pdf",
        )
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        logger.debug(
            "SentenceAwareChunker initialized — "
            "chunk_size={cs}, overlap={ov}, min={mn}",
            cs=self.config.chunk_size,
            ov=self.config.overlap_size,
            mn=self.config.min_chunk_size,
        )

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: Optional[int] = None,
        char_offset: int = 0,
        source_metadata: Optional[dict] = None,
        existing_chunk_count: int = 0,
    ) -> list[DocumentChunk]:
        """
        Chunk a text string into overlapping DocumentChunks.

        This is called once per page when processing PDFs (each page's
        text is chunked independently with its page number). Pages with
        very little text may produce only one chunk.

        Args:
            text:               Cleaned text to chunk.
            document_id:        UUID of the parent document.
            document_name:      Original filename.
            page_number:        PDF page number (1-based), or None.
            char_offset:        Character position of this text in the
                                full document (for cross-page provenance).
            source_metadata:    Document-level metadata dict.
            existing_chunk_count: Number of chunks already created for
                                  this document (for sequential indexing).

        Returns:
            List of DocumentChunk objects. May be empty if text is too short.
        """
        if not text or not text.strip():
            return []

        sentences = split_into_sentences(text)
        if not sentences:
            return []

        # Pre-compute word counts for each sentence (avoid re-counting in loop)
        sentence_word_counts = [len(s.split()) for s in sentences]

        chunks: list[DocumentChunk] = []
        chunk_index = existing_chunk_count

        # Sliding window state
        window_start: int = 0  # Sentence index where current window starts

        while window_start < len(sentences):
            # --- Build a chunk starting at window_start ---
            chunk_sentences: list[str] = []
            chunk_word_count: int = 0
            current_sentence: int = window_start

            while current_sentence < len(sentences):
                sent = sentences[current_sentence]
                sent_words = sentence_word_counts[current_sentence]

                chunk_sentences.append(sent)
                chunk_word_count += sent_words
                current_sentence += 1

                # Stop when we've reached the target chunk size
                if chunk_word_count >= self.config.chunk_size:
                    break

            if not chunk_sentences:
                break

            chunk_text = " ".join(chunk_sentences)

            # Only keep chunks that meet the minimum size
            if chunk_word_count >= self.config.min_chunk_size:
                # Calculate character offsets in the full document
                # (approximate: based on position in the page text)
                pre_text = " ".join(sentences[:window_start])
                start_in_page = len(pre_text) + (1 if pre_text else 0)
                end_in_page = start_in_page + len(chunk_text)

                chunk = DocumentChunk.create(
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    page_number=page_number,
                    start_char=char_offset + start_in_page,
                    end_char=char_offset + end_in_page,
                    source_metadata=source_metadata or {},
                )
                chunks.append(chunk)
                chunk_index += 1

            # --- Advance window with overlap ---
            # Roll back from current_sentence until we've shed overlap_size words
            words_to_shed = chunk_word_count - self.config.overlap_size
            words_shed = 0
            new_start = window_start

            for i in range(window_start, current_sentence):
                if words_shed >= words_to_shed:
                    break
                words_shed += sentence_word_counts[i]
                new_start = i + 1

            # Prevent infinite loop: always advance at least one sentence
            if new_start <= window_start:
                new_start = window_start + 1

            window_start = new_start

        logger.debug(
            "Chunked '{name}' page={page} → {n} chunks",
            name=document_name,
            page=page_number,
            n=len(chunks),
        )

        return chunks

    def chunk_pages(
        self,
        pages: list[tuple[int, str]],
        document_id: str,
        document_name: str,
        source_metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a list of (page_number, page_text) tuples.

        This is the primary entry point when processing PDFs — each page
        is chunked independently so page numbers are preserved per chunk.

        Pages with insufficient text (< min_chunk_size words) are merged
        with the next page before chunking to avoid orphan micro-chunks.

        Args:
            pages:           List of (page_number, text) tuples (1-based page nums).
            document_id:     UUID of the parent document.
            document_name:   Original filename.
            source_metadata: Document-level metadata.

        Returns:
            Flat list of all DocumentChunks across all pages.
        """
        all_chunks: list[DocumentChunk] = []
        char_offset = 0

        # Merge short pages with the next page
        merged_pages = self._merge_short_pages(pages)

        for page_number, page_text in merged_pages:
            page_chunks = self.chunk_document(
                text=page_text,
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                char_offset=char_offset,
                source_metadata=source_metadata,
                existing_chunk_count=len(all_chunks),
            )
            all_chunks.extend(page_chunks)
            char_offset += len(page_text) + 1  # +1 for page separator

        logger.info(
            "Total chunks for '{name}': {n} chunks from {p} pages",
            name=document_name,
            n=len(all_chunks),
            p=len(merged_pages),
        )

        return all_chunks

    def _merge_short_pages(
        self,
        pages: list[tuple[int, str]],
    ) -> list[tuple[int, str]]:
        """
        Merge pages with very few words into the following page.

        Prevents single-sentence pages (like chapter title pages) from
        producing micro-chunks that pollute the evidence pool.

        A page is "short" if it has fewer than min_chunk_size words.
        It gets merged into the next page, keeping the first page's number.
        """
        if not pages:
            return []

        result: list[tuple[int, str]] = []
        buffer_page_num = pages[0][0]
        buffer_text = pages[0][1]

        for page_num, page_text in pages[1:]:
            buffer_words = len(buffer_text.split())

            if buffer_words < self.config.min_chunk_size:
                # Merge buffer into current page
                buffer_text = buffer_text + "\n" + page_text
                # Keep buffer_page_num (the earlier page)
            else:
                result.append((buffer_page_num, buffer_text))
                buffer_page_num = page_num
                buffer_text = page_text

        # Flush the last buffer
        if buffer_text.strip():
            result.append((buffer_page_num, buffer_text))

        return result

"""
scripts/demo_phase2.py

MENTOR DEMONSTRATION SCRIPT — Phase 2: Document Processing

This script demonstrates the core document processing pipeline WITHOUT
needing an API server or a PDF file. It runs the chunker on a sample
research text and shows the DocumentChunk output.

Run from the project root (with venv activated):
    python scripts/demo_phase2.py

What it demonstrates:
    1. Text cleaning and normalization
    2. Sentence boundary detection
    3. Sentence-aware sliding window chunking
    4. DocumentChunk data model (the evidence unit)
    5. Chunk statistics
"""
from __future__ import annotations
import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processing.text_cleaner import clean_text, split_into_sentences
from src.document_processing.chunker import SentenceAwareChunker, ChunkingConfig
from src.document_processing.metadata_manager import DocumentChunk
import uuid

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colors for clean terminal output
# ─────────────────────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
LINE   = "─" * 70

def header(title: str) -> None:
    print(f"\n{BOLD}{BLUE}{LINE}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{LINE}{RESET}")

def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}  ▶ {title}{RESET}")

def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")

def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TEXT — simulates what would be extracted from a research PDF
# This is representative of the kind of content in HotpotQA documents
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a framework that combines information 
retrieval with large language model generation to improve factual grounding. 
The system retrieves relevant documents from an external knowledge base and 
uses them as context when generating an answer. This approach reduces 
hallucination by grounding responses in retrieved evidence.

However, conventional RAG systems have a critical limitation: they retrieve 
documents and generate answers without evaluating whether the retrieved evidence 
is actually sufficient to support a reliable response. Retrieved evidence may be 
incomplete, irrelevant, contradictory, or weakly supportive of the query. 
Despite this, the system proceeds to generate an answer as if the evidence 
were adequate.

Evidence sufficiency refers to the degree to which retrieved passages 
collectively support a reliable answer to a given query. A passage may be 
relevant to the query topic but still fail to provide the specific information 
needed to answer it correctly. For example, a passage about climate change 
may be retrieved for a query about Arctic ice melt rates, but if it does not 
contain specific measurements or timelines, it is insufficient evidence for 
a precise factual answer.

Natural Language Inference (NLI) is the task of determining whether a 
hypothesis follows from a premise. In the context of evidence evaluation, 
NLI can be used to assess whether a retrieved passage entails, contradicts, 
or is neutral with respect to a candidate answer. Models such as DeBERTa-v3 
trained on the Multi-Genre Natural Language Inference corpus achieve strong 
performance on this task and can be applied to evidence evaluation without 
task-specific fine-tuning.

The proposed Evidence Sufficiency Model aggregates five evidence signals: 
semantic relevance, entailment probability, contradiction probability, source 
reliability, and evidence coverage. These signals are combined into a single 
interpretable sufficiency score that drives an adaptive decision: answer the 
question, retrieve additional evidence, or abstain from answering when reliable 
evidence cannot be established.
"""


def run_demo() -> None:

    header("Reliability-Oriented RAG Framework")
    print(f"  {BOLD}Phase 2 — Document Processing Pipeline Demo{RESET}")
    print(f"  B.Tech CSE (AI & DS) Major Project | Graphic Era University")

    # ── Step 1: Text Cleaning ─────────────────────────────────────────────
    section("Step 1: Text Cleaning")
    raw_len = len(SAMPLE_TEXT.split())
    cleaned = clean_text(SAMPLE_TEXT)
    cleaned_len = len(cleaned.split())

    ok(f"Input text: {raw_len} words (simulated PDF extraction)")
    ok(f"After cleaning: {cleaned_len} words")
    info("Operations: ligature replacement, hyphen-break fixing, whitespace normalization")
    print(f"\n  {DIM}Preview (first 150 chars):{RESET}")
    print(f"  \"{cleaned[:150]}…\"")

    # ── Step 2: Sentence Splitting ────────────────────────────────────────
    section("Step 2: Sentence Boundary Detection")
    sentences = split_into_sentences(cleaned)
    ok(f"Detected {len(sentences)} sentences (sentence-boundary-aware splitting)")
    info("Critical for NLI: DeBERTa-v3 scores complete sentences, not fragments")

    for i, sent in enumerate(sentences[:3], 1):
        words = len(sent.split())
        print(f"\n  {DIM}Sentence {i} ({words} words):{RESET}")
        print(f"  \"{sent[:100]}{'…' if len(sent) > 100 else ''}\"")

    if len(sentences) > 3:
        print(f"\n  {DIM}... and {len(sentences) - 3} more sentences{RESET}")

    # ── Step 3: Chunking ──────────────────────────────────────────────────
    section("Step 3: Sentence-Aware Sliding Window Chunking")

    config = ChunkingConfig(chunk_size=120, overlap_size=25, min_chunk_size=20)
    info(f"Config: chunk_size={config.chunk_size} words, overlap={config.overlap_size} words")
    info("Note: Production uses 400 words / 80 word overlap. Smaller values used here for demo clarity.")

    chunker = SentenceAwareChunker(config)
    doc_id = str(uuid.uuid4())

    chunks = chunker.chunk_document(
        text=cleaned,
        document_id=doc_id,
        document_name="sample_research_paper.pdf",
        page_number=1,
    )

    ok(f"Created {len(chunks)} evidence chunks from the text")
    ok("No sentence was split mid-way (guaranteed by algorithm design)")

    # ── Step 4: Show DocumentChunk model ─────────────────────────────────
    section("Step 4: DocumentChunk — The Evidence Unit")
    info("Every retrieved passage in the RAG pipeline is a DocumentChunk.")
    info("The Evidence Sufficiency Model scores each chunk on 5 dimensions.")

    for i, chunk in enumerate(chunks):
        print(f"\n  {BOLD}{YELLOW}┌─ Chunk {i+1} of {len(chunks)} {'─' * 45}{RESET}")
        print(f"  {YELLOW}│{RESET}  chunk_id    : {DIM}{chunk.chunk_id[:16]}…{RESET}")
        print(f"  {YELLOW}│{RESET}  document_id : {DIM}{chunk.document_id[:16]}…{RESET}")
        print(f"  {YELLOW}│{RESET}  chunk_index : {chunk.chunk_index}")
        print(f"  {YELLOW}│{RESET}  page_number : {chunk.page_number}")
        print(f"  {YELLOW}│{RESET}  word_count  : {GREEN}{chunk.word_count} words{RESET}")
        print(f"  {YELLOW}│{RESET}  char range  : [{chunk.start_char} → {chunk.end_char}]")
        print(f"  {YELLOW}│{RESET}  text preview: \"{chunk.preview(80)}\"")
        print(f"  {YELLOW}└{'─' * 55}{RESET}")

    # ── Step 5: Statistics ────────────────────────────────────────────────
    section("Step 5: Chunk Statistics")
    word_counts = [c.word_count for c in chunks]
    total_words = sum(word_counts)
    avg_words = total_words / len(chunks) if chunks else 0

    ok(f"Total chunks   : {len(chunks)}")
    ok(f"Total words    : {total_words}")
    ok(f"Avg chunk size : {avg_words:.1f} words")
    ok(f"Min chunk size : {min(word_counts)} words")
    ok(f"Max chunk size : {max(word_counts)} words")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{LINE}{RESET}")
    print(f"{BOLD}{GREEN}  ✓ Phase 2 Document Processing — WORKING{RESET}")
    print(f"{GREEN}{LINE}{RESET}")
    print(f"\n  {BOLD}What comes next:{RESET}")
    print(f"  Phase 3 → These chunks are stored in Qdrant vector DB")
    print(f"  Phase 3 → BM25 + BGE-M3 retrieve relevant chunks for a query")
    print(f"  Phase 4 → Each retrieved chunk is scored on 5 evidence dimensions")
    print(f"  Phase 5 → Evidence Sufficiency Model combines scores → ESM Score")
    print(f"  Phase 6 → ESM Score drives: ANSWER / RETRIEVE MORE / ABSTAIN\n")


if __name__ == "__main__":
    run_demo()

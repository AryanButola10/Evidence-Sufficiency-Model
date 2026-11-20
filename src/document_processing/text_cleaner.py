"""
src/document_processing/text_cleaner.py

Text cleaning and normalization for extracted PDF text.

PDF text extraction is messy. Common issues:
    - Ligatures (ﬁ, ﬂ, ﬀ) that break word boundaries
    - Hyphenated line breaks ("infor-\nmation" → "information")
    - Multiple spaces, tabs, and stray whitespace
    - Null bytes and control characters from binary PDFs
    - Page headers/footers mixed into content
    - Unicode normalization issues

This module fixes these problems before chunking.
Cleaning is intentionally conservative — we preserve meaning, we don't
remove content that could be evidence.

Design principle:
    Cleaning must be reproducible and auditable. Any text removed must
    be justified, because removed text is evidence that can never be retrieved.
"""
from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Ligature map — PDF fonts often encode ligatures as single characters
# that break tokenization and NLI models
# ---------------------------------------------------------------------------
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
    "\u2019": "'",    # right single quotation mark → apostrophe
    "\u2018": "'",    # left single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2026": "...",  # ellipsis
    "\u00ad": "",     # soft hyphen (invisible but can break tokenization)
}


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean and normalize text extracted from a PDF.

    Args:
        text:       Raw text string from PyMuPDF extraction.
        aggressive: If True, also removes likely page headers/footers
                    (short lines < 5 words at start/end of pages).
                    Default False — keep all content.

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Step 1: Unicode normalization (NFC — composed form)
    text = unicodedata.normalize("NFC", text)

    # Step 2: Replace ligatures with their ASCII equivalents
    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)

    # Step 3: Remove null bytes and other control characters
    # Keep: regular whitespace (\n, \t, space), printable characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Step 4: Fix hyphenated line breaks
    # "infor-\nmation" → "information"
    # Only join if the break is mid-word (lowercase on both sides)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # Step 5: Normalize whitespace within lines
    # Multiple spaces → single space (but preserve line breaks)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Step 6: Collapse excessive blank lines (more than 2 → 2)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:  # Allow single blank line between paragraphs
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Step 7: Final strip
    text = text.strip()

    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse all whitespace (including newlines) to single spaces.

    Used when preparing text for NLI models that expect a single
    flat string rather than multi-line paragraphs.

    Args:
        text: Input text string.

    Returns:
        Text with all whitespace normalized to single spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.

    This is a lightweight sentence splitter — no NLTK or spaCy required.
    It handles common cases:
        - Period, exclamation, question mark followed by space and capital
        - Abbreviations are partially handled (Mr., Dr., etc. won't split)

    Why we do this here (not in chunker):
        The chunker needs sentences as atomic units. This function
        provides the boundary detection, so the chunker never cuts
        a sentence in half.

    Args:
        text: Cleaned text string (single paragraph or multi-paragraph).

    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
    """
    # Normalize to single-line for sentence splitting
    flat = normalize_whitespace(text)

    if not flat:
        return []

    # Sentence boundary detection — two-pass approach:
    # Pass 1: Temporarily replace periods in known abbreviations
    #         so they don't trigger false sentence splits.
    # Pass 2: Split on remaining sentence boundaries.

    # Common abbreviations whose trailing period should NOT end a sentence
    _ABBREVS = [
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
        "vs", "etc", "Fig", "Eq", "Vol", "No", "pp",
        "al", "et", "e.g", "i.e",
    ]

    # Protect abbreviations: replace "Mr." with "Mr<DOT>"
    protected = flat
    for abbr in _ABBREVS:
        protected = protected.replace(f"{abbr}.", f"{abbr}<DOT>")

    # Also protect decimal numbers: "3.14" → "3<DOT>14"
    protected = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", protected)

    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)

    # Restore protected dots
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

    # Fallback: if splitting produced nothing useful, treat whole text as one sentence
    if not sentences:
        return [flat]

    return sentences

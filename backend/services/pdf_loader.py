"""
backend/services/pdf_loader.py
------------------------------
Turns an uploaded PDF into clean, per-page text. No chunking, no embedding.

Handles three kinds of document:

  1. Normal PDFs with a real text layer      -> pypdf extraction
  2. PDFs whose extraction loses word breaks -> pypdf layout-mode retry
  3. Scanned / image-only PDFs               -> OCR (services/ocr.py)

Case 3 is decided PER PAGE, not per document, so a report that is mostly
typed text with a few scanned exhibit pages only pays the OCR cost on
those few pages.
"""

from dataclasses import dataclass
import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.config import settings
from backend.services.ocr import OCRUnavailableError, ocr_available, ocr_pdf_pages

# A page yielding less than this many characters is treated as "no real text"
# and becomes an OCR candidate in auto mode.
_MIN_CHARS_FOR_REAL_TEXT = 40


class InvalidPDFError(Exception):
    """Raised when a file cannot be parsed or yields no usable text."""


@dataclass
class PageText:
    document_name: str
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str
    via_ocr: bool = False


def _clean_text(raw: str) -> str:
    """Remove common extraction artifacts without touching real content."""
    if not raw:
        return ""
    text = raw.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)   # de-hyphenate line-wrapped words
    text = re.sub(r"[ \t]+", " ", text)         # collapse runs of spaces/tabs
    text = re.sub(r"\n{2,}", "\n", text)        # collapse blank lines
    text = re.sub(r" *\n *", "\n", text)        # trim spaces around newlines
    return text.strip()


def _looks_space_starved(text: str) -> bool:
    """Real prose has roughly one space per 5-6 characters. Far fewer than
    that usually means word breaks were lost during extraction."""
    if not text or len(text) < 40:
        return False
    return (text.count(" ") / len(text)) < 0.03


def _extract_page_text(page) -> str:
    try:
        raw = page.extract_text() or ""
    except Exception:
        return ""

    if _looks_space_starved(raw):
        try:
            layout_text = page.extract_text(extraction_mode="layout") or ""
            if layout_text and not _looks_space_starved(layout_text):
                return layout_text
        except Exception:
            pass  # older pypdf versions may not support extraction_mode
    return raw


def load_pdf(
    file_bytes: bytes,
    document_name: str,
    max_mb: int = 20,
    ocr_mode: str = "auto",
) -> list[PageText]:
    """
    Extract text from a PDF, page by page.

    Args:
        ocr_mode: "off"   never OCR; image-only pages are simply dropped.
                  "auto"  OCR only the pages that yielded no real text (default).
                  "force" OCR every page, ignoring any existing text layer.
                          Useful when a PDF has a text layer that is garbage.

    Raises:
        InvalidPDFError: file is empty, too large, unreadable, encrypted, or
                          yields no usable text at all.
    """
    if not file_bytes:
        raise InvalidPDFError(f"'{document_name}' is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise InvalidPDFError(
            f"'{document_name}' is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit."
        )

    if ocr_mode not in {"off", "auto", "force"}:
        raise ValueError(f"ocr_mode must be 'off', 'auto', or 'force' (got '{ocr_mode}').")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as e:
        raise InvalidPDFError(f"'{document_name}' is not a readable PDF: {e}") from e

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise InvalidPDFError(f"'{document_name}' is password-protected.")

    total_pages = len(reader.pages)

    # --- pass 1: native text extraction -------------------------------------
    extracted: dict[int, str] = {}
    if ocr_mode != "force":
        for i, page in enumerate(reader.pages, start=1):
            extracted[i] = _clean_text(_extract_page_text(page))
    else:
        extracted = {i: "" for i in range(1, total_pages + 1)}

    # --- pass 2: OCR the pages that need it ---------------------------------
    ocr_pages: set[int] = set()
    ocr_note = ""

    if ocr_mode == "force":
        candidates = list(range(1, total_pages + 1))
    elif ocr_mode == "auto":
        candidates = [
            i for i, text in extracted.items()
            if len(text) < _MIN_CHARS_FOR_REAL_TEXT
        ]
    else:
        candidates = []

    if candidates:
        try:
            recovered = ocr_pdf_pages(file_bytes, page_numbers=candidates)
            for page_number, text in recovered.items():
                cleaned = _clean_text(text)
                if cleaned:
                    extracted[page_number] = cleaned
                    ocr_pages.add(page_number)
        except OCRUnavailableError as e:
            if ocr_mode == "force":
                # The caller explicitly asked for OCR, so failing loudly is right.
                raise InvalidPDFError(f"OCR was requested but is unavailable: {e}") from e
            # In auto mode, carry on with whatever real text we found and
            # explain the gap only if we end up with nothing usable.
            ocr_note = (
                f" {len(candidates)} page(s) appear to be scanned images, but OCR "
                f"is unavailable: {e}"
            )

    # --- assemble -----------------------------------------------------------
    pages = [
        PageText(
            document_name=document_name,
            page_number=i,
            text=extracted[i],
            via_ocr=i in ocr_pages,
        )
        for i in sorted(extracted)
        if extracted[i]
    ]

    if not pages:
        if ocr_mode == "off":
            raise InvalidPDFError(
                f"'{document_name}' has no extractable text. It looks like a scanned "
                f"or image-based PDF \u2014 re-upload it with OCR set to Auto or Always."
            )
        raise InvalidPDFError(
            f"'{document_name}' produced no readable text.{ocr_note or ' The pages may be blank, or the scan quality may be too low to recognize.'}"
        )

    return pages


def ocr_status() -> tuple[bool, str]:
    """Re-exported so the API layer doesn't need to import the OCR module directly."""
    return ocr_available()

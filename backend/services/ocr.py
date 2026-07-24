"""
backend/services/ocr.py
-------------------------
Optical Character Recognition for PDFs that have no usable text layer —
scans, photographed documents, or image-heavy exports where the text is
baked into pictures rather than stored as selectable characters.

How it works: each page is rasterized to an image with pypdfium2 (a pip-only
renderer, so no Poppler/system PDF tooling is needed), then passed to
Tesseract via pytesseract to recover the characters.

The ONE thing that can't be installed with pip is the Tesseract binary
itself, since it's a compiled C++ program rather than a Python package.
See README for the installer link. If it isn't present, this module
degrades gracefully: ocr_available() reports false, the API surfaces that
in /api/health, and the UI disables OCR rather than crashing.

This module deliberately does NOT import pdf_loader — it returns plain
{page_number: text} dicts so the two stay decoupled and there's no
circular import.
"""

from __future__ import annotations

import io

from backend.config import settings


class OCRUnavailableError(Exception):
    """Raised when OCR is requested but Tesseract isn't installed/reachable."""


def _configure_tesseract():
    """Apply an explicit tesseract.exe path if the user set one in .env.
    Needed on Windows, where the installer often doesn't add it to PATH."""
    import pytesseract

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    return pytesseract


def ocr_available() -> tuple[bool, str]:
    """
    Check whether OCR can actually run right now.

    Returns (is_available, human_readable_status). Never raises — this is
    called by the health endpoint and the UI on every page load.
    """
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return False, "pypdfium2 is not installed (pip install pypdfium2)"

    try:
        pytesseract = _configure_tesseract()
    except ImportError:
        return False, "pytesseract is not installed (pip install pytesseract)"

    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version}"
    except Exception:
        return False, (
            "Tesseract binary not found. Install it and, on Windows, set "
            "TESSERACT_CMD in .env to the full path of tesseract.exe"
        )


def ocr_pdf_pages(
    file_bytes: bytes,
    page_numbers: list[int] | None = None,
    dpi: int | None = None,
    language: str | None = None,
) -> dict[int, str]:
    """
    Run OCR over a PDF and return {page_number (1-indexed): recovered_text}.

    Args:
        file_bytes:    the raw PDF.
        page_numbers:  only OCR these pages (1-indexed). None = every page.
                        Passing a subset is what makes hybrid documents cheap:
                        pages that already had real text are skipped.
        dpi:           rasterization resolution. Higher is more accurate and
                        much slower; 300 is the usual sweet spot for body text.
        language:      Tesseract language code(s), e.g. "eng" or "eng+deu".

    Raises:
        OCRUnavailableError: Tesseract or a Python dependency is missing.
    """
    ok, status = ocr_available()
    if not ok:
        raise OCRUnavailableError(status)

    import pypdfium2 as pdfium

    pytesseract = _configure_tesseract()

    dpi = dpi or settings.ocr_dpi
    language = language or settings.ocr_language
    scale = dpi / 72.0  # pypdfium2 renders at 72 DPI baseline

    results: dict[int, str] = {}
    pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))

    try:
        total_pages = len(pdf)
        targets = page_numbers if page_numbers else range(1, total_pages + 1)

        for page_number in targets:
            if page_number < 1 or page_number > total_pages:
                continue

            page = pdf[page_number - 1]
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("L")  # grayscale reads cleaner
                text = pytesseract.image_to_string(image, lang=language) or ""
                results[page_number] = text.strip()
            except Exception:
                # One unreadable page shouldn't sink the whole document.
                results[page_number] = ""
            finally:
                try:
                    page.close()
                except Exception:
                    pass
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return results

"""
backend/services/text_chunker.py
---------------------------------
Splits extracted page text into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter (paragraph -> sentence -> word aware).
Chunking happens PER PAGE (not on one giant concatenated document string)
so every chunk keeps an exact page number for source attribution.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.services.pdf_loader import PageText


@dataclass
class Chunk:
    chunk_id: str          # e.g. "handbook.pdf::p12::c0"
    document_name: str
    page_number: int
    text: str


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk every page's text and return a flat list of Chunk objects."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{page.document_name}::p{page.page_number}::c{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=piece,
                )
            )
    return chunks

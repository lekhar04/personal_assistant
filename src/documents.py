"""Document text extraction and chunking."""
import io
import re

from . import config


class DocumentError(Exception):
    """Raised when an uploaded file cannot be read or contains no usable text."""


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Plain-text files are returned as a single page 1."""
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"'{filename}' is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise DocumentError(f"'{filename}' is empty.")

    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError(f"'{filename}' is password-protected.")
            pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
        except DocumentError:
            raise
        except Exception as e:  # pypdf raises many different exception types
            raise DocumentError(f"Could not read PDF '{filename}': {e}") from e
    elif name.endswith((".txt", ".md")):
        text = None
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise DocumentError(f"Could not decode '{filename}' as text.")
        pages = [(1, text.strip())]
    else:
        raise DocumentError(f"Unsupported file type: '{filename}'. Upload a PDF, TXT or MD file.")

    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise DocumentError(
            f"No extractable text found in '{filename}'. Scanned/image-only PDFs are not supported."
        )
    return pages


def _split_long(segment: str, size: int) -> list[str]:
    return [segment[i : i + size] for i in range(0, len(segment), size)]


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Split text into ~size-character chunks on sentence/paragraph boundaries with overlap."""
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    segments: list[str] = []
    for seg in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        seg = seg.strip()
        if not seg:
            continue
        segments.extend(_split_long(seg, size) if len(seg) > size else [seg])

    chunks: list[str] = []
    current = ""
    for seg in segments:
        if current and len(current) + len(seg) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            # start the next chunk at a word boundary inside the overlap
            tail = tail.split(" ", 1)[1] if " " in tail else tail
            current = f"{tail} {seg}".strip()
        else:
            current = f"{current} {seg}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Return [(chunk_text, page_number)]; chunks never span pages so citations stay accurate."""
    out: list[tuple[str, int]] = []
    for page, text in pages:
        out.extend((c, page) for c in chunk_text(text))
    return out

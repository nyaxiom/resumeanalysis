"""Lightweight TXT/PDF/DOCX parsing, independent of LLM entity extraction."""

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Protocol

from resumeanalysis.recommendation.schema import SourceDocument


class DocumentError(ValueError):
    """Redacted document failure; never includes source text or parser internals."""


class DocumentParser(Protocol):
    def parse(self, data: bytes, filename: str, *, source_id: str) -> SourceDocument: ...


class LocalDocumentParser:
    def __init__(self, *, max_bytes=10 * 1024 * 1024, max_characters=200_000, max_pages=100):
        if any(type(v) is not int or v < 1 for v in (max_bytes, max_characters, max_pages)):
            raise ValueError("Document limits must be positive integers")
        self.max_bytes, self.max_characters, self.max_pages = max_bytes, max_characters, max_pages

    def parse_file(self, path, *, source_id):
        # Read a bounded amount even when the on-disk size changes concurrently.
        with Path(path).open("rb") as stream:
            data = stream.read(self.max_bytes + 1)
        return self.parse(data, Path(path).name, source_id=source_id)

    def parse(self, data, filename, *, source_id):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".pdf", ".docx"}:
            raise DocumentError("Unsupported document format; use TXT, PDF or DOCX")
        if not isinstance(data, bytes) or not data or len(data) > self.max_bytes:
            raise DocumentError("Document is empty or exceeds the byte limit")
        pages = []
        try:
            if suffix == ".txt":
                if b"\0" in data:
                    raise DocumentError("TXT input must contain UTF-8 text")
                text = data.decode("utf-8-sig")
            elif suffix == ".pdf":
                from pypdf import PdfReader

                if not data.startswith(b"%PDF-"):
                    raise DocumentError("The document does not have a PDF signature")
                reader = PdfReader(io.BytesIO(data), strict=True)
                if reader.is_encrypted:
                    raise DocumentError("Encrypted PDFs require prior local decryption")
                if len(reader.pages) > self.max_pages:
                    raise DocumentError("PDF exceeds the page limit")
                chunks, offset = [], 0
                for index, page in enumerate(reader.pages, 1):
                    chunk = page.extract_text() or ""
                    if offset + len(chunk) > self.max_characters:
                        raise DocumentError("Document text exceeds the character limit")
                    pages.append({"page": index, "start": offset, "end": offset + len(chunk)})
                    chunks.append(chunk)
                    offset += len(chunk) + 1
                text = "\n".join(chunks)
            else:
                from docx import Document

                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    entries = archive.infolist()
                    if (
                        len(entries) > 2048
                        or sum(item.file_size for item in entries) > 32 * 1024 * 1024
                        or any(
                            item.file_size / max(item.compress_size, 1) > 200 for item in entries
                        )
                    ):
                        raise DocumentError("DOCX exceeds archive expansion limits")
                    if "word/document.xml" not in archive.namelist():
                        raise DocumentError("The archive is not a DOCX document")
                document = Document(io.BytesIO(data))
                parts = []
                for block in document.iter_inner_content():
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.extend(
                            "\t".join(cell.text for cell in row.cells) for row in block.rows
                        )
                text = "\n".join(parts)
            if not text.strip():
                raise DocumentError(
                    "No extractable text; image-only documents require a separate OCR parser"
                )
            if len(text) > self.max_characters:
                raise DocumentError("Document text exceeds the character limit")
        except DocumentError:
            raise
        except ImportError:
            raise DocumentError("Install the documents extra for PDF/DOCX parsing") from None
        except Exception:
            raise DocumentError(
                "Document parsing failed; check format and file integrity"
            ) from None
        return SourceDocument(
            source_id=source_id,
            kind=suffix[1:],
            text=text,
            original_sha256=hashlib.sha256(data).hexdigest(),
            pages=pages,
        )

"""
Multi-format knowledge document parser and PII/transactional dump guard for RAG ingestion.

Supports: PDF, DOCX, TXT/MD, and CSV without ingesting raw transactional customer PII dumps.
"""
from __future__ import annotations

import csv
import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from typing import Tuple

from platform_core.errors.exceptions import PermissionDeniedError, ValidationError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

_CARD_NUMBER_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SENSITIVE_HEADER_RE = re.compile(
    r"\b(card_number|credit_card|cvv|cvc|aadhaar|ssn|password|secret_key|bank_account)\b",
    re.IGNORECASE,
)


def assert_no_transactional_pii_dump(text: str, filename: str = "document") -> None:
    """
    Prevent accidental ingestion of raw transactional PII / payment dumps into vector embeddings.
    """
    sample = (text or "")[:20000]
    if _SENSITIVE_HEADER_RE.search(sample):
        # Allow policy docs describing "AADHAAR/PAN" if not CSV/tabular dump; block CSV-like PII columns
        lines = [ln.strip() for ln in sample.splitlines()[:10] if ln.strip()]
        if lines and ("," in lines[0] or "\t" in lines[0]):
            first_row = lines[0].lower()
            if any(tok in first_row for tok in ("card_number", "cvv", "password", "ssn", "aadhaar", "secret_key")):
                raise PermissionDeniedError(
                    f"Rejected document '{filename}': detected transactional PII columns that must not be embedded in RAG."
                )

    card_hits = _CARD_NUMBER_RE.findall(sample)
    # Count sequences that look like raw unmasked 16-digit PANs (not dates/order IDs)
    unmasked_16 = [h for h in card_hits if len(re.sub(r"\D", "", h)) in (15, 16) and "*" not in h]
    if len(unmasked_16) >= 3:
        raise PermissionDeniedError(
            f"Rejected document '{filename}': detected multiple unmasked payment card numbers (PII guard)."
        )


def _extract_pdf_bytes(raw: bytes) -> str:
    # 1. Try optional pypdf / PyPDF2 / pymupdf if present
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except ImportError:
        pass
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=raw, filetype="pdf")
        return "\n\n".join(page.get_text("text") for page in doc).strip()
    except ImportError:
        pass

    # 2. Pure-Python fallback for uncompressed / FlateDecode PDF content streams
    texts = []
    streams = re.findall(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw, re.DOTALL)
    for stream in streams:
        decoded = stream
        try:
            decoded = zlib.decompress(stream)
        except Exception:
            pass
        text_blocks = re.findall(rb"BT(.*?)ET", decoded, re.DOTALL)
        for blk in text_blocks:
            for lit in re.findall(rb"\((.*?)\)\s*T[jJ]", blk, re.DOTALL):
                try:
                    s = lit.decode("latin-1").replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
                    if s.strip():
                        texts.append(s)
                except Exception:
                    continue
    extracted = " ".join(texts).strip()
    if not extracted:
        raise ValidationError(
            "Unable to extract text from PDF without 'pypdf' installed or PDF contains only scanned images."
        )
    return extracted


def _extract_docx_bytes(raw: bytes) -> str:
    try:
        import docx  # type: ignore

        doc = docx.Document(io.BytesIO(raw))
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_txt = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_txt:
                    parts.append(row_txt)
        return "\n".join(parts).strip()
    except ImportError:
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        paragraphs = []
        for p in root.iter():
            if p.tag.endswith("}p"):
                runs = [node.text for node in p.iter() if node.tag.endswith("}t") and node.text]
                if runs:
                    paragraphs.append("".join(runs))
        return "\n".join(paragraphs).strip()
    except Exception as exc:
        raise ValidationError(f"Invalid DOCX file: {exc}") from exc


def _extract_csv_bytes(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames:
        rows = []
        for idx, row in enumerate(reader, start=1):
            pairs = [f"{k}: {v}" for k, v in row.items() if k and v is not None and str(v).strip()]
            if pairs:
                rows.append(f"Row {idx}: " + ", ".join(pairs))
        return "\n".join(rows).strip()
    return text.strip()


def extract_document_text(filename: str, raw_bytes: bytes) -> Tuple[str, str]:
    """
    Validate size & extension and extract normalized plain text from PDF, DOCX, CSV, or TXT/MD.
    Returns `(source_type, extracted_text)`.
    """
    if not raw_bytes:
        raise ValidationError("Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValidationError(f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    ext = os.path.splitext((filename or "").strip().lower())[1]
    if ext in (".txt", ".md"):
        source_type = "text"
        extracted = raw_bytes.decode("utf-8", errors="replace").strip()
    elif ext == ".csv":
        source_type = "csv"
        extracted = _extract_csv_bytes(raw_bytes)
    elif ext == ".docx":
        source_type = "docx"
        extracted = _extract_docx_bytes(raw_bytes)
    elif ext == ".pdf":
        source_type = "pdf"
        extracted = _extract_pdf_bytes(raw_bytes)
    else:
        raise ValidationError(f"Unsupported document format '{ext}'. Supported formats: .pdf, .docx, .txt, .md, .csv.")

    if not extracted:
        raise ValidationError(f"No readable text content found in '{filename}'.")

    assert_no_transactional_pii_dump(extracted, filename=filename)
    return source_type, extracted


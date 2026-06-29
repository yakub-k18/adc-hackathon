"""File type detection and content extraction."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
from docx import Document
from pypdf import PdfReader

from core.config import STRUCTURED_EXTENSIONS, SUPPORTED_EXTENSIONS, UNSTRUCTURED_EXTENSIONS

FileKind = Literal["structured", "unstructured", "unsupported"]


def get_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def detect_file_kind(file_name: str, mode: str = "auto") -> FileKind:
    ext = get_extension(file_name)

    if mode == "structured":
        return "structured" if ext in STRUCTURED_EXTENSIONS else "unsupported"
    if mode == "unstructured":
        return "unstructured" if ext in UNSTRUCTURED_EXTENSIONS else "unsupported"

    if ext in STRUCTURED_EXTENSIONS:
        return "structured"
    if ext in UNSTRUCTURED_EXTENSIONS:
        return "unstructured"
    return "unsupported"


def validate_supported(file_name: str) -> None:
    ext = get_extension(file_name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def read_csv(uploaded_file: Union[io.BytesIO, Any]) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def read_xlsx(uploaded_file: Union[io.BytesIO, Any]) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, engine="openpyxl")


def read_txt(uploaded_file: Union[io.BytesIO, Any]) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def read_docx(uploaded_file: Union[io.BytesIO, Any]) -> str:
    uploaded_file.seek(0)
    document = Document(uploaded_file)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_pdf(uploaded_file: Union[io.BytesIO, Any]) -> str:
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    if not pages:
        raise ValueError("Could not extract text from PDF. The file may be scanned or image-based.")
    return "\n\n".join(pages)


def extract_structured_dataframe(uploaded_file: Union[io.BytesIO, Any], file_name: str) -> pd.DataFrame:
    ext = get_extension(file_name)
    if ext == ".csv":
        return read_csv(uploaded_file)
    if ext == ".xlsx":
        return read_xlsx(uploaded_file)
    raise ValueError(f"Not a structured file: {file_name}")


def extract_unstructured_text(uploaded_file: Union[io.BytesIO, Any], file_name: str) -> str:
    ext = get_extension(file_name)
    if ext == ".txt":
        return read_txt(uploaded_file)
    if ext == ".docx":
        return read_docx(uploaded_file)
    if ext == ".pdf":
        return read_pdf(uploaded_file)
    raise ValueError(f"Not an unstructured file: {file_name}")


def get_sample_values(series: pd.Series, limit: int = 5) -> list[str]:
    values: list[str] = []
    for value in series.dropna().astype(str).head(limit * 3):
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in {"nan", "none", "null"}:
            values.append(cleaned)
        if len(values) >= limit:
            break
    return values


def build_column_text(column_name: str, sample_values: list[str]) -> str:
    samples = ", ".join(sample_values[:5])
    return f"Column: {column_name}. Sample values: {samples}"


def dataframe_preview(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    return df.head(max_rows).copy()

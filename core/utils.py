"""Shared utility helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from core.schemas import DetectionResult, SummaryStats


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def extract_json_from_text(text: str) -> Optional[dict[str, Any] | list[Any]]:
    """Extract and parse JSON from LLM output, handling fences and extra prose."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    candidates: list[str] = [cleaned]

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return None


def _repair_json(text: str) -> Optional[str]:
    """Best-effort JSON repair for common LLM mistakes."""
    repaired = text.strip()
    repaired = repaired.replace("'", '"')
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    return repaired


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def get_context_window(text: str, start: int, end: int, window: int = 120) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def normalize_classification(value: str) -> str:
    from core.config import NON_NPPI, NPPI, SENSITIVE_NPPI

    if not value:
        return NON_NPPI

    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "NPPI": NPPI,
        "SENSITIVE_NPPI": SENSITIVE_NPPI,
        "SENSITIVENPPI": SENSITIVE_NPPI,
        "SENSITIVE": SENSITIVE_NPPI,
        "NON_NPPI": NON_NPPI,
        "NONNPPI": NON_NPPI,
        "PUBLIC": NON_NPPI,
        "NOT_NPPI": NON_NPPI,
    }
    return mapping.get(normalized, NON_NPPI)


def clamp_confidence(value: Any, default: float = 70.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, num))


def findings_to_dataframe_rows(findings: list[DetectionResult]) -> list[dict[str, Any]]:
    return [
        {
            "Source": f.source,
            "Column / Entity": f.column_or_entity,
            "Sample Value / Value": f.sample_value,
            "Entity Type": f.entity_type,
            "Final Classification": f.classification,
            "Confidence": round(f.confidence, 1),
            "Explanation": f.explanation,
            "Context Snippet": f.context_snippet,
        }
        for f in findings
    ]


def compute_summary(findings: list[DetectionResult]) -> SummaryStats:
    from core.config import NON_NPPI, NPPI, SENSITIVE_NPPI

    stats = SummaryStats(total_findings=len(findings))
    for finding in findings:
        if finding.classification == SENSITIVE_NPPI:
            stats.sensitive_nppi_count += 1
        elif finding.classification == NPPI:
            stats.nppi_count += 1
        elif finding.classification == NON_NPPI:
            stats.non_nppi_count += 1
    return stats


def column_name_heuristic(column_name: str) -> tuple[str, float, str]:
    """Fallback classification based on column name keywords."""
    from core.config import (
        HEURISTIC_NON_NPPI_KEYWORDS,
        HEURISTIC_NPPI_KEYWORDS,
        HEURISTIC_SENSITIVE_KEYWORDS,
        NON_NPPI,
        NPPI,
        SENSITIVE_NPPI,
    )

    name = column_name.lower().replace("-", "_")

    for keyword in HEURISTIC_SENSITIVE_KEYWORDS:
        if keyword in name:
            return SENSITIVE_NPPI, 82.0, f"Column name contains sensitive keyword '{keyword}'"

    if (
        name.endswith("_id")
        or name.endswith("_code")
        or name.endswith("_sku")
        or name in {"id", "sku", "product_id", "customer_id", "order_id"}
    ):
        if not any(k in name for k in ("phone", "mobile", "email", "aadhaar", "pan", "passport")):
            return NON_NPPI, 80.0, "Column name indicates a business or system identifier"

    for keyword in HEURISTIC_NON_NPPI_KEYWORDS:
        if keyword in name:
            return NON_NPPI, 78.0, f"Column name suggests business identifier ('{keyword}')"

    for keyword in HEURISTIC_NPPI_KEYWORDS:
        if keyword in name:
            return NPPI, 75.0, f"Column name suggests personal data ('{keyword}')"

    return NON_NPPI, 55.0, "No strong column-name signal; defaulting to NON_NPPI"


def is_high_confidence_column(
    column_name: str,
    entity_types: list[str],
    presidio_result: dict[str, Any] | None = None,
) -> tuple[bool, tuple[str, float, str]]:
    """Return True when heuristics + Presidio agree strongly enough to skip LLM."""
    from core.config import (
        HEURISTIC_SKIP_LLM_CONFIDENCE,
        NON_NPPI,
        NPPI,
        SENSITIVE_NPPI,
        SENSITIVE_ENTITY_TYPES,
    )
    from core.entity_disambiguation import has_type_conflict

    classification, confidence, reason = column_name_heuristic(column_name)

    if confidence < HEURISTIC_SKIP_LLM_CONFIDENCE:
        return False, (classification, confidence, reason)

    if has_type_conflict(entity_types):
        return False, (classification, confidence, reason)

    if entity_types:
        if any(t in SENSITIVE_ENTITY_TYPES for t in entity_types):
            if classification != SENSITIVE_NPPI:
                return False, (classification, confidence, reason)
            return True, (SENSITIVE_NPPI, max(confidence, 85.0), reason)

        personal = {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"}
        if any(t in personal for t in entity_types):
            if classification == NON_NPPI:
                return False, (classification, confidence, reason)
            return True, (NPPI, max(confidence, 80.0), reason)

        if classification in {NPPI, SENSITIVE_NPPI} and not any(
            t in personal | SENSITIVE_ENTITY_TYPES for t in entity_types
        ):
            return False, (classification, confidence, reason)

    if classification == NON_NPPI and confidence >= HEURISTIC_SKIP_LLM_CONFIDENCE:
        return True, (classification, confidence, reason)

    if classification in {NPPI, SENSITIVE_NPPI} and confidence >= HEURISTIC_SKIP_LLM_CONFIDENCE:
        return True, (classification, confidence, reason)

    return False, (classification, confidence, reason)

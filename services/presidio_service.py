"""Microsoft Presidio integration for candidate NPPI detection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from core.config import MAX_SAMPLE_VALUES
from core.utils import clean_string, get_context_window
from core.entity_disambiguation import (
    consolidate_entity_hits,
    refine_entity_types_list,
)
from recognizers.india_recognizers import register_india_recognizers
from services.file_processor import build_column_text, get_sample_values

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_analyzer_engine() -> AnalyzerEngine:
    """Create and cache Presidio analyzer with custom India recognizers."""
    try:
        analyzer = AnalyzerEngine()
    except Exception as exc:
        logger.warning("Presidio init failed (%s). Install: python -m spacy download en_core_web_sm", exc)
        raise RuntimeError(
            "Presidio analyzer failed to start. Run: python -m spacy download en_core_web_sm"
        ) from exc

    register_india_recognizers(analyzer)
    return analyzer


class PresidioService:
    def __init__(self) -> None:
        self.analyzer = get_analyzer_engine()

    def analyze_text(self, text: str, language: str = "en") -> list[RecognizerResult]:
        if not text or not text.strip():
            return []
        try:
            return self.analyzer.analyze(text=text, language=language)
        except Exception as exc:
            logger.exception("Presidio analyze failed: %s", exc)
            return []

    def analyze_column(self, column_name: str, series: Any) -> dict[str, Any]:
        sample_values = get_sample_values(series, MAX_SAMPLE_VALUES)
        column_text = build_column_text(column_name, sample_values)
        combined_text = column_text
        if sample_values:
            combined_text = f"{column_text}\n" + " | ".join(sample_values)

        full_results = self.analyze_text(combined_text)
        entity_hits: list[dict[str, Any]] = []

        for result in full_results:
            value = combined_text[result.start : result.end]
            entity_hits.append(
                {
                    "value": clean_string(value),
                    "entity_type": result.entity_type,
                    "score": float(result.score),
                    "start": result.start,
                    "end": result.end,
                }
            )

        entity_hits = consolidate_entity_hits(entity_hits, column_name=column_name)
        entity_types = sorted({h["entity_type"] for h in entity_hits})
        entity_types = refine_entity_types_list(
            entity_types, sample_values, column_name=column_name
        )

        return {
            "sample_values": sample_values,
            "column_text": column_text,
            "entity_types": entity_types,
            "entity_hits": entity_hits,
            "recognizer_results": full_results,
        }

    @staticmethod
    def result_to_dict(result: RecognizerResult, text: str) -> dict[str, Any]:
        value = text[result.start : result.end]
        return {
            "value": clean_string(value),
            "entity_type": result.entity_type,
            "score": float(result.score),
            "start": result.start,
            "end": result.end,
        }

    def deduplicate_results(
        self,
        results: list[RecognizerResult],
        text: str,
        context: str = "",
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for result in sorted(results, key=lambda r: (r.start, -r.score)):
            item = self.result_to_dict(result, text)
            item["context_snippet"] = get_context_window(
                text, result.start, result.end, 80
            )
            hits.append(item)

        return consolidate_entity_hits(hits, context=context)

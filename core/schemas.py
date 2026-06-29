"""Data schemas for analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DetectionResult:
    """Single entity or column finding."""

    source: str
    column_or_entity: str
    sample_value: str
    entity_type: str
    classification: str
    confidence: float
    explanation: str
    context_snippet: str = ""
    risk_impact: str = "Low"
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "column_or_entity": self.column_or_entity,
            "sample_value": self.sample_value,
            "entity_type": self.entity_type,
            "classification": self.classification,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "context_snippet": self.context_snippet,
            "risk_impact": self.risk_impact,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class ColumnAnalysisResult:
    """Structured column-level analysis."""

    column_name: str
    sample_values: list[str]
    candidate_entity_types: list[str]
    classification: str
    confidence: float
    explanation: str
    risk_impact: str = "Medium"
    used_ai: bool = False

    def to_detection(self, file_name: str) -> DetectionResult:
        sample = self.sample_values[0] if self.sample_values else ""
        entity_types = ", ".join(self.candidate_entity_types) if self.candidate_entity_types else "COLUMN"
        return DetectionResult(
            source=file_name,
            column_or_entity=self.column_name,
            sample_value=sample,
            entity_type=entity_types,
            classification=self.classification,
            confidence=self.confidence,
            explanation=self.explanation,
            context_snippet=f"Samples: {', '.join(self.sample_values[:3])}",
            risk_impact=self.risk_impact,
        )


@dataclass
class SummaryStats:
    total_findings: int = 0
    nppi_count: int = 0
    sensitive_nppi_count: int = 0
    non_nppi_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_findings": self.total_findings,
            "nppi_count": self.nppi_count,
            "sensitive_nppi_count": self.sensitive_nppi_count,
            "non_nppi_count": self.non_nppi_count,
        }


@dataclass
class RiskAssessment:
    risk_level: str
    risk_score: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "explanation": self.explanation,
        }


@dataclass
class FileAnalysisResult:
    file_name: str
    file_type: str
    risk_level: str
    risk_score: int
    summary: SummaryStats
    findings: list[DetectionResult] = field(default_factory=list)
    raw_preview: Any = None
    extracted_text: str = ""
    ai_warnings: list[str] = field(default_factory=list)
    risk_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "summary": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "raw_preview": self.raw_preview,
            "ai_warnings": self.ai_warnings,
            "risk_explanation": self.risk_explanation,
        }

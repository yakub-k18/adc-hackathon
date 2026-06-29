"""File-level risk scoring based on NPPI findings."""

from __future__ import annotations

from core.config import (
    NON_NPPI,
    NPPI,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    SENSITIVE_NPPI,
    SENSITIVE_ENTITY_TYPES,
)
from core.schemas import DetectionResult, RiskAssessment, SummaryStats


def assess_risk(findings: list[DetectionResult], summary: SummaryStats) -> RiskAssessment:
    if summary.total_findings == 0:
        return RiskAssessment(
            risk_level=RISK_LOW,
            risk_score=5,
            explanation="No NPPI-related findings detected in this file.",
        )

    score = 0
    reasons: list[str] = []

    score += summary.nppi_count * 8
    score += summary.sensitive_nppi_count * 18

    sensitive_types_found = {
        f.entity_type.split(",")[0].strip()
        for f in findings
        if f.classification == SENSITIVE_NPPI
    }
    for entity_type in sensitive_types_found:
        if entity_type in SENSITIVE_ENTITY_TYPES:
            score += 12
            reasons.append(f"Sensitive identifier detected: {entity_type}")

    nppi_types = {
        f.entity_type.split(",")[0].strip()
        for f in findings
        if f.classification in {NPPI, SENSITIVE_NPPI}
    }
    if len(nppi_types) >= 3:
        score += 15
        reasons.append("Multiple NPPI entity categories present")

    if summary.nppi_count + summary.sensitive_nppi_count >= 8:
        score += 20
        reasons.append("High volume of NPPI findings")

    if summary.sensitive_nppi_count >= 2:
        score += 15
        reasons.append("Multiple sensitive NPPI elements found")

    non_ratio = summary.non_nppi_count / max(summary.total_findings, 1)
    if non_ratio >= 0.8 and summary.sensitive_nppi_count == 0 and summary.nppi_count <= 1:
        score = max(score - 15, 10)

    score = max(0, min(100, score))
    risk_level = _score_to_level(score, summary)

    if not reasons:
        if summary.sensitive_nppi_count:
            reasons.append("Sensitive NPPI data present")
        elif summary.nppi_count:
            reasons.append("Personal data (NPPI) detected")
        else:
            reasons.append("Mostly non-personal identifiers")

    return RiskAssessment(
        risk_level=risk_level,
        risk_score=score,
        explanation="; ".join(reasons),
    )


def _score_to_level(score: int, summary: SummaryStats) -> str:
    if summary.sensitive_nppi_count >= 2 or score >= 85:
        return RISK_CRITICAL
    if summary.sensitive_nppi_count >= 1 or score >= 65:
        return RISK_HIGH
    if summary.nppi_count >= 2 or score >= 40:
        return RISK_MEDIUM
    if score >= 20 or summary.nppi_count >= 1:
        return RISK_MEDIUM
    return RISK_LOW

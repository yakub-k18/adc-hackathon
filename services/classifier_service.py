"""Main classification orchestration pipeline."""

from __future__ import annotations

import io
import logging
from typing import Any, Union

from core.config import (
    CONTEXT_WINDOW_CHARS,
    DEFAULT_LLM_PROVIDER,
    MAX_LLM_ENTITIES,
    MAX_TEXT_PREVIEW_CHARS,
    MAX_UNSTRUCTURED_BATCH,
    NON_NPPI,
    NON_NPPI_ENTITY_TYPES,
    NPPI,
    PROVIDER_OLLAMA,
    SENSITIVE_ENTITY_TYPES,
    SENSITIVE_NPPI,
    SPEED_BALANCED,
    SPEED_FAST,
    SPEED_FULL,
)
from core.entity_disambiguation import (
    disambiguate_entity_type,
    refine_llm_entity_type,
)
from core.schemas import ColumnAnalysisResult, DetectionResult, FileAnalysisResult
from core.utils import (
    clean_string,
    column_name_heuristic,
    compute_summary,
    get_context_window,
    is_high_confidence_column,
    clamp_confidence,
)
from services.file_processor import (
    dataframe_preview,
    detect_file_kind,
    extract_structured_dataframe,
    extract_unstructured_text,
    validate_supported,
)
from services.llm_service import (
    LLMService,
    normalize_structured_batch_results,
    normalize_unstructured_llm_results,
)
from services.presidio_service import PresidioService
from services.risk_service import assess_risk

logger = logging.getLogger(__name__)


class ClassifierService:
    def __init__(
        self,
        model_name: str | None = None,
        provider: str = DEFAULT_LLM_PROVIDER,
    ) -> None:
        self.presidio = PresidioService()
        self.llm = LLMService(model_name=model_name, provider=provider)
        self.provider = provider

    def analyze_uploaded_file(
        self,
        uploaded_file: Union[io.BytesIO, Any],
        file_name: str,
        use_ai_validation: bool = True,
        mode: str = "auto",
        speed_mode: str = SPEED_BALANCED,
    ) -> FileAnalysisResult:
        validate_supported(file_name)
        file_kind = detect_file_kind(file_name, mode)
        if file_kind == "unsupported":
            raise ValueError(f"File '{file_name}' is not supported for mode '{mode}'.")

        ai_warnings: list[str] = []
        use_ai = use_ai_validation and speed_mode != SPEED_FAST
        llm_available = use_ai and self.llm.is_available()

        if use_ai and not llm_available:
            ai_warnings.append(
                f"{self.llm.availability_message()} Continuing with Presidio + heuristic fallback."
            )

        if speed_mode == SPEED_FAST:
            ai_warnings.append("Fast mode: skipped AI validation for speed.")

        uploaded_file.seek(0)
        if file_kind == "structured":
            return self._analyze_structured(
                uploaded_file, file_name, llm_available, ai_warnings, speed_mode
            )
        return self._analyze_unstructured(
            uploaded_file, file_name, llm_available, ai_warnings, speed_mode
        )

    def _analyze_structured(
        self,
        uploaded_file: Any,
        file_name: str,
        use_ai: bool,
        ai_warnings: list[str],
        speed_mode: str,
    ) -> FileAnalysisResult:
        df = extract_structured_dataframe(uploaded_file, file_name)
        column_profiles: list[dict[str, Any]] = []

        for column in df.columns:
            presidio_result = self.presidio.analyze_column(str(column), df[column])
            column_profiles.append(
                {
                    "column_name": str(column),
                    "presidio_result": presidio_result,
                    "sample_values": presidio_result["sample_values"],
                    "entity_types": presidio_result["entity_types"],
                }
            )

        findings: list[DetectionResult] = []
        llm_queue: list[dict[str, Any]] = []

        for profile in column_profiles:
            column_name = profile["column_name"]
            presidio_result = profile["presidio_result"]
            entity_types = profile["entity_types"]

            if not use_ai:
                findings.append(
                    self._heuristic_column_classification(
                        column_name, profile["sample_values"], entity_types, presidio_result
                    ).to_detection(file_name)
                )
                continue

            if speed_mode == SPEED_BALANCED:
                skip, resolved = is_high_confidence_column(column_name, entity_types, presidio_result)
                if skip:
                    classification, confidence, reason = resolved
                    findings.append(
                        ColumnAnalysisResult(
                            column_name=column_name,
                            sample_values=profile["sample_values"],
                            candidate_entity_types=entity_types,
                            classification=classification,
                            confidence=confidence,
                            explanation=f"{reason} (fast heuristic)",
                            risk_impact="High" if classification == SENSITIVE_NPPI else "Medium",
                            used_ai=False,
                        ).to_detection(file_name)
                    )
                    continue

            llm_queue.append(profile)

        if use_ai and llm_queue:
            findings.extend(
                self._classify_structured_batch(llm_queue, file_name, ai_warnings)
            )

        summary = compute_summary(findings)
        risk = assess_risk(findings, summary)

        return FileAnalysisResult(
            file_name=file_name,
            file_type="structured",
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            summary=summary,
            findings=findings,
            raw_preview=dataframe_preview(df),
            extracted_text="",
            ai_warnings=ai_warnings,
            risk_explanation=risk.explanation,
        )

    def _classify_structured_batch(
        self,
        profiles: list[dict[str, Any]],
        file_name: str,
        ai_warnings: list[str],
    ) -> list[DetectionResult]:
        batch_input = [
            {
                "column_name": p["column_name"],
                "sample_values": p["sample_values"],
                "entity_types": p["entity_types"],
            }
            for p in profiles
        ]

        llm_raw, error = self.llm.validate_structured_columns_batch(batch_input)
        findings: list[DetectionResult] = []

        if llm_raw:
            by_name = normalize_structured_batch_results(llm_raw)
            for profile in profiles:
                column_name = profile["column_name"]
                presidio_result = profile["presidio_result"]
                normalized = by_name.get(column_name.lower())

                if normalized:
                    entity_types = normalized.get("entity_types") or profile["entity_types"]
                    if profile["sample_values"]:
                        refined_type, refine_reason = refine_llm_entity_type(
                            profile["sample_values"][0],
                            entity_types[0] if entity_types else "UNKNOWN",
                            column_name=column_name,
                        )
                        if refined_type != "UNKNOWN":
                            entity_types = [refined_type]
                        if refine_reason and "Corrected" in refine_reason:
                            clf, conf, rsn = self._classification_from_entity_type(
                                refined_type, column_name, refine_reason
                            )
                            normalized["classification"] = clf
                            normalized["confidence"] = conf
                            normalized["reason"] = rsn
                        elif refine_reason:
                            normalized["reason"] = refine_reason

                    findings.append(
                        ColumnAnalysisResult(
                            column_name=column_name,
                            sample_values=profile["sample_values"],
                            candidate_entity_types=entity_types,
                            classification=normalized["classification"],
                            confidence=normalized["confidence"],
                            explanation=normalized["reason"],
                            risk_impact=normalized["risk_impact"],
                            used_ai=True,
                        ).to_detection(file_name)
                    )
                else:
                    findings.append(
                        self._heuristic_column_classification(
                            column_name,
                            profile["sample_values"],
                            profile["entity_types"],
                            presidio_result,
                        ).to_detection(file_name)
                    )
            return findings

        if error:
            if "timed out" in error.lower():
                ai_warnings.append(
                    f"{error} Results below use Presidio/heuristic fallback for affected columns."
                )
            else:
                ai_warnings.append(f"Batch LLM fallback: {error}")

        for profile in profiles:
            findings.append(
                self._heuristic_column_classification(
                    profile["column_name"],
                    profile["sample_values"],
                    profile["entity_types"],
                    profile["presidio_result"],
                ).to_detection(file_name)
            )
        return findings

    def _heuristic_column_classification(
        self,
        column_name: str,
        sample_values: list[str],
        entity_types: list[str],
        presidio_result: dict[str, Any],
    ) -> ColumnAnalysisResult:
        classification, confidence, reason = column_name_heuristic(column_name)

        if entity_types and sample_values:
            resolved_type, disambig_reason = disambiguate_entity_type(
                sample_values[0],
                candidate_types=entity_types,
                column_name=column_name,
            )
            if resolved_type != "UNKNOWN":
                entity_types = [resolved_type]
                classification, confidence, reason = self._classification_from_entity_type(
                    resolved_type, column_name, disambig_reason
                )

        hits = presidio_result.get("entity_hits") or []
        if hits and classification == NON_NPPI:
            top = hits[0]
            hit_type = top.get("entity_type", "")
            if hit_type in SENSITIVE_ENTITY_TYPES and hit_type != "US_BANK_NUMBER":
                classification = SENSITIVE_NPPI
                confidence = 88.0
                reason = f"Value pattern matches {hit_type}"
            elif hit_type in {"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION"}:
                classification = NPPI
                confidence = 85.0
                reason = top.get("disambiguation_reason") or f"Identified as {hit_type}"

        return ColumnAnalysisResult(
            column_name=column_name,
            sample_values=sample_values,
            candidate_entity_types=entity_types,
            classification=classification,
            confidence=confidence,
            explanation=reason,
            risk_impact="High" if classification == SENSITIVE_NPPI else "Medium",
            used_ai=False,
        )

    @staticmethod
    def _classification_from_entity_type(
        entity_type: str,
        column_name: str,
        reason: str = "",
    ) -> tuple[str, float, str]:
        if entity_type in SENSITIVE_ENTITY_TYPES:
            return SENSITIVE_NPPI, 88.0, reason or f"Sensitive entity: {entity_type}"
        if entity_type in {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"}:
            return NPPI, 85.0, reason or f"Personal data: {entity_type}"
        if entity_type in NON_NPPI_ENTITY_TYPES:
            return NON_NPPI, 80.0, reason or f"Non-personal identifier: {entity_type}"
        if entity_type == "US_BANK_NUMBER":
            return SENSITIVE_NPPI, 88.0, reason or "Bank account number"
        return NON_NPPI, 70.0, reason or f"Entity type: {entity_type}"

    def _analyze_unstructured(
        self,
        uploaded_file: Any,
        file_name: str,
        use_ai: bool,
        ai_warnings: list[str],
        speed_mode: str,
    ) -> FileAnalysisResult:
        text = extract_unstructured_text(uploaded_file, file_name)
        if not text.strip():
            raise ValueError("No text could be extracted from the uploaded file.")

        recognizer_results = self.presidio.analyze_text(text)
        entities = self.presidio.deduplicate_results(recognizer_results, text)

        for entity in entities:
            if not entity.get("context_snippet"):
                entity["context_snippet"] = get_context_window(
                    text, entity["start"], entity["end"], CONTEXT_WINDOW_CHARS
                )

        findings: list[DetectionResult] = []

        if use_ai and entities:
            llm_entities = entities[:MAX_LLM_ENTITIES]
            if len(entities) > MAX_LLM_ENTITIES:
                ai_warnings.append(
                    f"Truncated AI validation to top {MAX_LLM_ENTITIES} entities for speed."
                )

            if speed_mode == SPEED_BALANCED:
                heuristic_findings: list[DetectionResult] = []
                ambiguous: list[dict[str, Any]] = []
                for entity in llm_entities:
                    finding = self._heuristic_entity_finding(entity, file_name)
                    if self._is_clear_entity(entity, finding):
                        heuristic_findings.append(finding)
                    else:
                        ambiguous.append(entity)
                findings.extend(heuristic_findings)
                if ambiguous:
                    findings.extend(
                        self._classify_unstructured_with_ai(
                            text, ambiguous, file_name, ai_warnings, single_batch=True
                        )
                    )
            else:
                findings.extend(
                    self._classify_unstructured_with_ai(
                        text, llm_entities, file_name, ai_warnings, single_batch=True
                    )
                )
        else:
            findings.extend(self._classify_unstructured_heuristic(entities, file_name))

        if not entities and not findings:
            ai_warnings.append("Presidio found no entities. File marked as low risk.")

        summary = compute_summary(findings)
        risk = assess_risk(findings, summary)

        return FileAnalysisResult(
            file_name=file_name,
            file_type="unstructured",
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            summary=summary,
            findings=findings,
            raw_preview=text[:MAX_TEXT_PREVIEW_CHARS],
            extracted_text=text,
            ai_warnings=ai_warnings,
            risk_explanation=risk.explanation,
        )

    @staticmethod
    def _is_clear_entity(entity: dict[str, Any], finding: DetectionResult) -> bool:
        from core.entity_disambiguation import is_email, is_indian_mobile, is_url

        entity_type = entity.get("entity_type", "")
        value = clean_string(entity.get("value", ""))
        score = float(entity.get("score", 0.0))

        if is_indian_mobile(value):
            return entity_type == "PHONE_NUMBER" and finding.classification == NPPI
        if is_email(value):
            return entity_type == "EMAIL_ADDRESS" and finding.classification == NPPI
        if is_url(value):
            return entity_type == "URL" and finding.classification == NON_NPPI

        if entity_type in {"IN_PAN", "IN_AADHAAR", "CREDIT_CARD"} and score >= 0.7:
            return True
        if entity_type in {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"} and score >= 0.85:
            return True
        if finding.classification == NON_NPPI and entity_type in {"DATE_TIME", "NRP", "URL"}:
            return True
        return False

    def _classify_unstructured_with_ai(
        self,
        text: str,
        entities: list[dict[str, Any]],
        file_name: str,
        ai_warnings: list[str],
        single_batch: bool = True,
    ) -> list[DetectionResult]:
        findings: list[DetectionResult] = []
        excerpt = text[:MAX_TEXT_PREVIEW_CHARS]
        batch_size = len(entities) if single_batch else MAX_UNSTRUCTURED_BATCH

        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            llm_raw, error = self.llm.validate_unstructured_entities(excerpt, batch)

            if llm_raw:
                ai_results = normalize_unstructured_llm_results(llm_raw)
                ai_by_value = {r["value"].lower(): r for r in ai_results}
                for entity in batch:
                    value = entity["value"]
                    ai_item = ai_by_value.get(value.lower())
                    if ai_item:
                        refined_type, refine_reason = refine_llm_entity_type(
                            value,
                            ai_item["entity_type"],
                            context=entity.get("context_snippet", ""),
                        )
                        explanation = ai_item["reason"]
                        classification = ai_item["classification"]
                        if refine_reason:
                            explanation = refine_reason
                            if "Corrected" in refine_reason:
                                classification, _, explanation = ClassifierService._classification_from_entity_type(
                                    refined_type, "", refine_reason
                                )

                        findings.append(
                            DetectionResult(
                                source=file_name,
                                column_or_entity=value,
                                sample_value=value,
                                entity_type=refined_type,
                                classification=classification,
                                confidence=ai_item["confidence"],
                                explanation=explanation,
                                context_snippet=ai_item.get("context_summary")
                                or entity.get("context_snippet", ""),
                                risk_impact=ai_item["risk_impact"],
                                start=entity.get("start"),
                                end=entity.get("end"),
                            )
                        )
                    else:
                        findings.append(self._heuristic_entity_finding(entity, file_name))
            else:
                if error:
                    if "timed out" in error.lower():
                        ai_warnings.append(
                            f"{error} Using Presidio/heuristic fallback for this batch."
                        )
                    else:
                        ai_warnings.append(f"LLM batch fallback: {error}")
                for entity in batch:
                    findings.append(self._heuristic_entity_finding(entity, file_name))

        return findings

    def _classify_unstructured_heuristic(
        self,
        entities: list[dict[str, Any]],
        file_name: str,
    ) -> list[DetectionResult]:
        return [self._heuristic_entity_finding(entity, file_name) for entity in entities]

    def _heuristic_entity_finding(self, entity: dict[str, Any], file_name: str) -> DetectionResult:
        value = clean_string(entity.get("value", ""))
        context = entity.get("context_snippet", "")
        original_type = entity.get("entity_type", "UNKNOWN")
        candidate_types = entity.get("original_types") or [original_type]

        entity_type, disambig_reason = disambiguate_entity_type(
            value,
            candidate_types=candidate_types,
            context=context,
        )
        if entity_type == "UNKNOWN":
            entity_type = original_type

        classification, _, reason = self._classification_from_entity_type(
            entity_type, "", disambig_reason
        )
        score = clamp_confidence(float(entity.get("score", 0.7)) * 100, 70.0)
        risk_impact = "High" if classification == SENSITIVE_NPPI else (
            "Medium" if classification == NPPI else "Low"
        )

        return DetectionResult(
            source=file_name,
            column_or_entity=value,
            sample_value=value,
            entity_type=entity_type,
            classification=classification,
            confidence=score,
            explanation=reason,
            context_snippet=context,
            risk_impact=risk_impact,
            start=entity.get("start"),
            end=entity.get("end"),
        )

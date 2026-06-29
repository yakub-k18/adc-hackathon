"""LLM integration: Ollama (local), OpenAI, and Anthropic (Claude)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_TIMEOUT_SECONDS,
    CLOUD_MAX_COLUMNS_PER_BATCH,
    CLOUD_MAX_ENTITIES_PER_BATCH,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_COLUMNS_PER_BATCH,
    OLLAMA_MAX_ENTITIES_PER_BATCH,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_WARMUP_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)
from core.utils import extract_json_from_text, normalize_classification, clamp_confidence

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: str = PROVIDER_OLLAMA,
    ) -> None:
        self.provider = provider
        self.model_name = model_name or self._default_model(provider)
        self.endpoint = OLLAMA_CHAT_ENDPOINT
        self._warmed_up = False

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults = {
            PROVIDER_OLLAMA: OLLAMA_DEFAULT_MODEL,
            PROVIDER_OPENAI: OPENAI_DEFAULT_MODEL,
            PROVIDER_ANTHROPIC: ANTHROPIC_DEFAULT_MODEL,
        }
        return defaults.get(provider, OLLAMA_DEFAULT_MODEL)

    @property
    def _columns_batch_size(self) -> int:
        if self.provider == PROVIDER_OLLAMA:
            return OLLAMA_MAX_COLUMNS_PER_BATCH
        return CLOUD_MAX_COLUMNS_PER_BATCH

    @property
    def _entities_batch_size(self) -> int:
        if self.provider == PROVIDER_OLLAMA:
            return OLLAMA_MAX_ENTITIES_PER_BATCH
        return CLOUD_MAX_ENTITIES_PER_BATCH

    def is_available(self) -> bool:
        if self.provider == PROVIDER_OLLAMA:
            try:
                base = self.endpoint.replace("/api/chat", "")
                response = requests.get(
                    f"{base}/api/tags",
                    timeout=(OLLAMA_CONNECT_TIMEOUT, 10),
                )
                return response.status_code == 200
            except requests.RequestException:
                return False

        if self.provider == PROVIDER_OPENAI:
            return bool(OPENAI_API_KEY.strip())

        if self.provider == PROVIDER_ANTHROPIC:
            return bool(ANTHROPIC_API_KEY.strip())

        return False

    def availability_message(self) -> str:
        if self.is_available():
            return f"{self.provider.title()} ready"

        if self.provider == PROVIDER_OLLAMA:
            return "Ollama not running. Start the Ollama app or switch to OpenAI/Claude."
        if self.provider == PROVIDER_OPENAI:
            return "Set OPENAI_API_KEY in .env or environment variables."
        if self.provider == PROVIDER_ANTHROPIC:
            return "Set ANTHROPIC_API_KEY in .env or environment variables."
        return "LLM provider unavailable."

    def warmup(self) -> None:
        """Pre-load Ollama model only; cloud APIs do not need warmup."""
        if self.provider != PROVIDER_OLLAMA or self._warmed_up:
            return

        self._warmed_up = True
        prompt = (
            'Return JSON only: {"results":[{"column_name":"test","classification":"NON_NPPI",'
            '"confidence":90,"entity_types":[],"reason":"warmup","risk_impact":"Low"}]}'
        )
        try:
            self.chat(prompt, STRUCTURED_SYSTEM_PROMPT, timeout=OLLAMA_WARMUP_TIMEOUT)
            logger.info("Ollama warmup completed for %s", self.model_name)
        except Exception as exc:
            logger.warning("Ollama warmup skipped: %s", exc)

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if self.provider == PROVIDER_OPENAI:
            return self._chat_openai(prompt, system_prompt, timeout)
        if self.provider == PROVIDER_ANTHROPIC:
            return self._chat_anthropic(prompt, system_prompt, timeout)
        return self._chat_ollama(prompt, system_prompt, timeout)

    def _chat_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        read_timeout = timeout or OLLAMA_TIMEOUT_SECONDS
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=(OLLAMA_CONNECT_TIMEOUT, read_timeout),
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return content, None
        except requests.Timeout:
            return None, (
                f"Ollama timed out after {read_timeout}s. "
                "Try OpenAI/Claude in the sidebar, or Analysis speed = Fast."
            )
        except requests.RequestException as exc:
            logger.warning("Ollama request failed: %s", exc)
            return None, str(exc)

    def _chat_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if not OPENAI_API_KEY.strip():
            return None, "OPENAI_API_KEY is not set."

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        read_timeout = timeout or OPENAI_TIMEOUT_SECONDS
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, read_timeout),
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content, None
        except requests.Timeout:
            return None, f"OpenAI timed out after {read_timeout}s."
        except (requests.RequestException, KeyError, IndexError) as exc:
            logger.warning("OpenAI request failed: %s", exc)
            detail = exc
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    detail = exc.response.json()
                except Exception:
                    detail = exc.response.text
            return None, f"OpenAI error: {detail}"

    def _chat_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if not ANTHROPIC_API_KEY.strip():
            return None, "ANTHROPIC_API_KEY is not set."

        read_timeout = timeout or ANTHROPIC_TIMEOUT_SECONDS
        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 1200,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{ANTHROPIC_BASE_URL.rstrip('/')}/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, read_timeout),
            )
            response.raise_for_status()
            data = response.json()
            parts = data.get("content", [])
            content = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
            return content, None
        except requests.Timeout:
            return None, f"Claude timed out after {read_timeout}s."
        except (requests.RequestException, KeyError) as exc:
            logger.warning("Anthropic request failed: %s", exc)
            detail = exc
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    detail = exc.response.json()
                except Exception:
                    detail = exc.response.text
            return None, f"Claude error: {detail}"

    def parse_json_response(self, content: Optional[str]) -> Optional[dict[str, Any] | list[Any]]:
        if not content:
            return None
        return extract_json_from_text(content)

    def validate_structured_columns_batch(
        self,
        columns: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not columns:
            return {"results": []}, None

        merged_results: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        batch_size = self._columns_batch_size

        for i in range(0, len(columns), batch_size):
            chunk = columns[i : i + batch_size]
            parsed, error = self._validate_structured_chunk(chunk)
            if parsed:
                merged_results.extend(parsed.get("results", []))
            else:
                last_error = error
                for col in chunk:
                    single, single_error = self.validate_structured_column(
                        col["column_name"],
                        col.get("sample_values", []),
                        col.get("entity_types", []),
                    )
                    if single:
                        merged_results.append(single)
                    elif single_error:
                        last_error = single_error

        if merged_results:
            return {"results": merged_results}, last_error

        return None, last_error or "Invalid JSON from LLM"

    def _validate_structured_chunk(
        self,
        columns: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        prompt = validate_structured_columns_batch_prompt(columns)
        content, error = self.chat(prompt, STRUCTURED_SYSTEM_PROMPT)
        parsed = self.parse_json_response(content)
        if parsed and isinstance(parsed, dict):
            return parsed, error
        return None, error or "Invalid JSON from LLM"

    def validate_structured_column(
        self,
        column_name: str,
        sample_values: list[str],
        candidate_entity_types: list[str],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        prompt = validate_structured_column_prompt(column_name, sample_values, candidate_entity_types)
        content, error = self.chat(prompt, STRUCTURED_SYSTEM_PROMPT)
        parsed = self.parse_json_response(content)
        if parsed and isinstance(parsed, dict):
            return parsed, error
        return None, error or "Invalid JSON from LLM"

    def validate_unstructured_entities(
        self,
        document_excerpt: str,
        entities: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not entities:
            return {"results": []}, None

        merged_results: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        batch_size = self._entities_batch_size

        for i in range(0, len(entities), batch_size):
            chunk = entities[i : i + batch_size]
            prompt = validate_unstructured_entities_prompt(document_excerpt, chunk)
            content, error = self.chat(prompt, UNSTRUCTURED_SYSTEM_PROMPT)
            parsed = self.parse_json_response(content)
            if parsed and isinstance(parsed, dict):
                merged_results.extend(parsed.get("results", []))
            else:
                last_error = error or "Invalid JSON from LLM"

        if merged_results:
            return {"results": merged_results}, last_error

        return None, last_error or "Invalid JSON from LLM"


DISAMBIGUATION_RULES = """
CRITICAL disambiguation rules (never confuse these):
- 10-digit numbers starting with 6-9 (e.g. 9876543210) = PHONE_NUMBER (Indian mobile), NOT bank account
- Bank account numbers are typically 11+ digits OR explicitly labeled account/ifsc/bank
- user@domain.com = EMAIL_ADDRESS, NOT URL
- http:// or https:// or www. = URL, NOT email
- 12-digit Aadhaar is NOT a 10-digit phone number
- Return the CORRECT entity_type in your JSON response
"""


STRUCTURED_SYSTEM_PROMPT = f"""You are an enterprise data privacy classifier.
Classify structured data columns as NPPI, SENSITIVE_NPPI, or NON_NPPI.
{DISAMBIGUATION_RULES}
Return ONLY valid JSON with no markdown or extra commentary."""

UNSTRUCTURED_SYSTEM_PROMPT = f"""You are an enterprise data privacy classifier.
Validate detected entities using surrounding context.
{DISAMBIGUATION_RULES}
Return ONLY valid JSON with no markdown or extra commentary."""


def validate_structured_columns_batch_prompt(columns: list[dict[str, Any]]) -> str:
    lines = []
    for col in columns:
        name = col.get("column_name", "")
        samples = ", ".join(col.get("sample_values", [])[:2]) or "(empty)"
        entities = ", ".join(col.get("entity_types", [])[:3]) or "none"
        lines.append(f"- {name} | samples: {samples} | presidio: {entities}")

    column_block = "\n".join(lines)
    return f"""Classify ALL columns below in one response.

Columns:
{column_block}

Rules:
- name/phone/email/address -> NPPI
- Aadhaar/PAN/passport/payment -> SENSITIVE_NPPI
- product_id/sku/order_id/inventory -> NON_NPPI
- 10-digit Indian mobile (9876543210) -> PHONE_NUMBER, never US_BANK_NUMBER
- user@domain.com -> EMAIL_ADDRESS; http:// links -> URL (NON_NPPI)

Return ONLY JSON:
{{
  "results": [
    {{
      "column_name": "example",
      "classification": "NPPI",
      "confidence": 90,
      "entity_types": ["PHONE_NUMBER"],
      "reason": "brief reason",
      "risk_impact": "Medium"
    }}
  ]
}}
"""


def validate_structured_column_prompt(
    column_name: str,
    sample_values: list[str],
    candidate_entity_types: list[str],
) -> str:
    samples = ", ".join(sample_values[:2]) or "(empty)"
    entities = ", ".join(candidate_entity_types[:3]) if candidate_entity_types else "none"
    return f"""Classify column "{column_name}".
Samples: {samples}
Presidio types: {entities}

Rules: personal data->NPPI, Aadhaar/PAN/payment->SENSITIVE_NPPI, product/sku/order id->NON_NPPI

Return ONLY JSON:
{{
  "column_name": "{column_name}",
  "classification": "NPPI",
  "confidence": 90,
  "entity_types": ["PHONE_NUMBER"],
  "reason": "brief reason",
  "risk_impact": "Medium"
}}
"""


def validate_unstructured_entities_prompt(
    document_excerpt: str,
    entities: list[dict[str, Any]],
) -> str:
    entity_lines = []
    for idx, entity in enumerate(entities, start=1):
        snippet = str(entity.get("context_snippet", ""))[:60]
        entity_lines.append(
            f"{idx}. value='{entity.get('value', '')}', "
            f"type={entity.get('entity_type', '')}, context=\"{snippet}\""
        )
    entity_block = "\n".join(entity_lines) if entity_lines else "No entities"
    excerpt = document_excerpt[:800]

    return f"""Validate entities using context.

Excerpt:
\"\"\"
{excerpt}
\"\"\"

Entities:
{entity_block}

Rules: personal->NPPI, Aadhaar/PAN/payment->SENSITIVE_NPPI, product/sku/order id->NON_NPPI

Return ONLY JSON:
{{
  "results": [
    {{
      "value": "example",
      "entity_type": "PERSON",
      "classification": "NPPI",
      "confidence": 90,
      "reason": "brief",
      "context_summary": "brief",
      "risk_impact": "Medium"
    }}
  ]
}}
"""


def normalize_structured_batch_results(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = raw.get("results") or []
    by_name: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column_name", "")).strip()
        if not name:
            continue
        by_name[name.lower()] = normalize_structured_llm_result(item)
    return by_name


def normalize_structured_llm_result(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "column_name": raw.get("column_name", ""),
        "classification": normalize_classification(str(raw.get("classification", ""))),
        "confidence": clamp_confidence(raw.get("confidence", 80)),
        "entity_types": raw.get("entity_types") or [],
        "reason": str(raw.get("reason", "AI classification")),
        "risk_impact": str(raw.get("risk_impact", "Medium")),
    }


def normalize_unstructured_llm_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    results = raw.get("results") or []
    normalized: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "value": str(item.get("value", "")),
                "entity_type": str(item.get("entity_type", "UNKNOWN")),
                "classification": normalize_classification(str(item.get("classification", ""))),
                "confidence": clamp_confidence(item.get("confidence", 75)),
                "reason": str(item.get("reason", "AI classification")),
                "context_summary": str(item.get("context_summary", "")),
                "risk_impact": str(item.get("risk_impact", "Medium")),
            }
        )
    return normalized

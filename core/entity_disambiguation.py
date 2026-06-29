"""Resolve conflicting Presidio entity types using patterns and context."""

from __future__ import annotations

import re
from typing import Any, Optional

# --- Value patterns ---
INDIAN_MOBILE_RE = re.compile(r"^(?:\+91[\s-]?)?[6-9]\d{9}$")
EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
    re.IGNORECASE,
)
URL_RE = re.compile(
    r"^(?:https?://|www\.)[^\s/$.?#].[^\s]*$|https?://",
    re.IGNORECASE,
)
PAN_RE = re.compile(r"^[A-Za-z]{5}\d{4}[A-Za-z]$")
AADHAAR_RE = re.compile(r"^[2-9]\d{3}\s?\d{4}\s?\d{4}$|^[2-9]\d{11}$")
CREDIT_CARD_RE = re.compile(r"^(?:\d[\s\-]?){13,19}$")

PHONE_CONTEXT = ("phone", "mobile", "contact", "cell", "tel", "whatsapp")
EMAIL_CONTEXT = ("email", "e-mail", "e_mail", "mail id", "mailid")
URL_CONTEXT = ("url", "website", "web site", "link", "homepage", "domain")
BANK_CONTEXT = ("bank", "account", "ifsc", "iban", "routing", "acct")

CONFLICTING_PAIRS = (
    frozenset({"PHONE_NUMBER", "US_BANK_NUMBER"}),
    frozenset({"EMAIL_ADDRESS", "URL"}),
    frozenset({"PHONE_NUMBER", "IN_AADHAAR"}),
)


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def is_url(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    if "@" in v and "://" not in v and not v.lower().startswith("www."):
        return False
    return bool(URL_RE.search(v) or v.lower().startswith("www."))


def is_email(value: str) -> bool:
    v = value.strip()
    if not v or "@" not in v:
        return False
    if is_url(v):
        return False
    return bool(EMAIL_RE.match(v))


def is_indian_mobile(value: str) -> bool:
    v = value.strip()
    digits = digits_only(v)
    if len(digits) == 10 and digits[0] in "6789":
        return True
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return True
    return bool(INDIAN_MOBILE_RE.match(v.replace(" ", "")))


def is_aadhaar(value: str) -> bool:
    compact = digits_only(value)
    if len(compact) == 12 and compact[0] in "23456789":
        return True
    return bool(AADHAAR_RE.match(value.strip()))


def is_pan(value: str) -> bool:
    return bool(PAN_RE.match(value.strip()))


def is_credit_card(value: str) -> bool:
    digits = digits_only(value)
    return 13 <= len(digits) <= 19 and bool(CREDIT_CARD_RE.match(value.strip()))


def is_bank_account_value(value: str, context: str = "") -> bool:
    """Bank account: long numeric, or explicit bank context — never 10-digit Indian mobile."""
    if is_indian_mobile(value):
        return False
    digits = digits_only(value)
    ctx = context.lower()
    if not digits.isdigit():
        return False
    if any(k in ctx for k in BANK_CONTEXT):
        return len(digits) >= 9
    return len(digits) >= 11


def has_type_conflict(entity_types: list[str]) -> bool:
    types = set(entity_types)
    for pair in CONFLICTING_PAIRS:
        if pair.issubset(types):
            return True
    return False


def disambiguate_entity_type(
    value: str,
    candidate_types: Optional[list[str]] = None,
    context: str = "",
    column_name: str = "",
) -> tuple[str, str]:
    """
    Pick the single best entity type for a value.
    Returns (entity_type, reason).
    """
    value = value.strip()
    ctx = f"{context} {column_name}".lower()
    col = column_name.lower().replace("-", "_")
    candidates = list(candidate_types or [])

    # --- Definitive value patterns (highest priority) ---
    if is_url(value):
        return "URL", "Contains URL scheme (http/https/www) — not an email"

    if is_email(value):
        return "EMAIL_ADDRESS", "Valid email format (user@domain) — not a URL"

    if is_pan(value):
        return "IN_PAN", "Matches PAN format (AAAAA9999A)"

    if is_aadhaar(value) and not is_indian_mobile(value):
        return "IN_AADHAAR", "12-digit Aadhaar pattern"

    if is_credit_card(value):
        return "CREDIT_CARD", "Credit/debit card number pattern"

    if is_indian_mobile(value):
        if any(k in ctx for k in BANK_CONTEXT) and not any(k in ctx for k in PHONE_CONTEXT):
            if not any(k in col for k in PHONE_CONTEXT):
                pass  # fall through only if strong bank signal
            else:
                return "PHONE_NUMBER", "10-digit Indian mobile — column/context indicates phone"
        return "PHONE_NUMBER", "10-digit Indian mobile number — not a bank account"

    if is_bank_account_value(value, ctx):
        return "US_BANK_NUMBER", "Long numeric value with bank/account context"

    # --- Column name hints ---
    if any(k in col for k in PHONE_CONTEXT):
        digits = digits_only(value)
        if digits.isdigit() and 8 <= len(digits) <= 13:
            return "PHONE_NUMBER", f"Column '{column_name}' indicates phone number"

    if any(k in col for k in EMAIL_CONTEXT):
        if "@" in value and not is_url(value):
            return "EMAIL_ADDRESS", f"Column '{column_name}' indicates email"

    if any(k in col for k in URL_CONTEXT):
        if is_url(value) or value.lower().startswith("www."):
            return "URL", f"Column '{column_name}' indicates URL"

    if any(k in col for k in BANK_CONTEXT):
        digits = digits_only(value)
        if digits.isdigit() and len(digits) >= 9 and not is_indian_mobile(value):
            return "US_BANK_NUMBER", f"Column '{column_name}' indicates bank account"

    # --- Resolve Presidio candidate conflicts ---
    if candidates:
        return _resolve_candidates(value, candidates, ctx)

    return "UNKNOWN", "No strong pattern match"


def _resolve_candidates(value: str, candidates: list[str], context: str) -> tuple[str, str]:
    types = set(candidates)
    digits = digits_only(value)

    if {"PHONE_NUMBER", "US_BANK_NUMBER"}.issubset(types):
        if is_indian_mobile(value) or (len(digits) == 10 and digits[0] in "6789"):
            return "PHONE_NUMBER", "10-digit number is a phone, not a bank account"
        if any(k in context for k in PHONE_CONTEXT):
            return "PHONE_NUMBER", "Surrounding context indicates phone"
        if any(k in context for k in BANK_CONTEXT) and len(digits) >= 11:
            return "US_BANK_NUMBER", "Surrounding context indicates bank account"
        if len(digits) == 10:
            return "PHONE_NUMBER", "Default: 10-digit values treated as phone"

    if {"EMAIL_ADDRESS", "URL"}.issubset(types):
        if is_email(value):
            return "EMAIL_ADDRESS", "Value is email format, not URL"
        if is_url(value):
            return "URL", "Value is URL format, not email"
        if "@" in value and "://" not in value:
            return "EMAIL_ADDRESS", "Contains @ without URL scheme → email"

    if {"PHONE_NUMBER", "IN_AADHAAR"}.issubset(types):
        if len(digits) == 10:
            return "PHONE_NUMBER", "10-digit value is phone, not 12-digit Aadhaar"
        if len(digits) == 12:
            return "IN_AADHAAR", "12-digit value matches Aadhaar"

    priority = [
        "IN_PAN",
        "IN_AADHAAR",
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PERSON",
        "LOCATION",
        "URL",
        "US_BANK_NUMBER",
        "DATE_TIME",
        "NRP",
    ]
    for entity_type in priority:
        if entity_type in types:
            return entity_type, f"Selected {entity_type} from Presidio candidates"

    return candidates[0], "Presidio candidate (first match)"


def refine_entity_types_list(
    entity_types: list[str],
    sample_values: list[str],
    column_name: str = "",
) -> list[str]:
    """Return deduplicated entity types after disambiguation across sample values."""
    if not entity_types and not sample_values:
        return []

    refined: set[str] = set()
    for value in sample_values[:5]:
        resolved, _ = disambiguate_entity_type(
            value,
            candidate_types=entity_types,
            column_name=column_name,
        )
        if resolved != "UNKNOWN":
            refined.add(resolved)

    if not refined:
        return entity_types

    return sorted(refined)


def consolidate_entity_hits(
    hits: list[dict[str, Any]],
    column_name: str = "",
    context: str = "",
) -> list[dict[str, Any]]:
    """
    Merge multiple Presidio hits for the same value into one disambiguated hit.
    """
    by_value: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        key = hit.get("value", "").strip().lower()
        if not key:
            continue
        by_value.setdefault(key, []).append(hit)

    consolidated: list[dict[str, Any]] = []
    for _key, group in by_value.items():
        value = group[0]["value"]
        candidate_types = [h["entity_type"] for h in group]
        hit_context = context or " ".join(
            h.get("context_snippet", "") for h in group if h.get("context_snippet")
        )

        entity_type, reason = disambiguate_entity_type(
            value,
            candidate_types=candidate_types,
            context=hit_context,
            column_name=column_name,
        )

        best_score = max(float(h.get("score", 0)) for h in group)
        consolidated.append(
            {
                "value": value,
                "entity_type": entity_type,
                "score": best_score,
                "start": min(h.get("start", 0) for h in group),
                "end": max(h.get("end", 0) for h in group),
                "disambiguation_reason": reason,
                "original_types": candidate_types,
            }
        )

    return consolidated


def refine_llm_entity_type(
    value: str,
    llm_entity_type: str,
    context: str = "",
    column_name: str = "",
) -> tuple[str, str]:
    """Post-process LLM output — override obvious misclassifications."""
    resolved, reason = disambiguate_entity_type(
        value,
        candidate_types=[llm_entity_type],
        context=context,
        column_name=column_name,
    )

    if resolved == "UNKNOWN":
        return llm_entity_type, ""

    if resolved != llm_entity_type:
        return resolved, f"Corrected {llm_entity_type} → {resolved}: {reason}"

    return llm_entity_type, reason

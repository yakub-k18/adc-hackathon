"""Custom Presidio recognizers for India-specific identifiers."""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


def get_aadhaar_recognizer() -> PatternRecognizer:
    """Recognize Indian Aadhaar numbers (12 digits, optional spaces)."""
    patterns = [
        Pattern(
            name="aadhaar_plain",
            regex=r"\b[2-9]\d{11}\b",
            score=0.65,
        ),
        Pattern(
            name="aadhaar_spaced",
            regex=r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
            score=0.75,
        ),
    ]
    return PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=patterns,
        context=["aadhaar", "aadhar", "uid", "uidai", "identity"],
        supported_language="en",
    )


def get_pan_recognizer() -> PatternRecognizer:
    """Recognize Indian PAN numbers (AAAAA9999A)."""
    patterns = [
        Pattern(
            name="pan_standard",
            regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            score=0.8,
        ),
        Pattern(
            name="pan_lower",
            regex=r"\b[a-z]{5}[0-9]{4}[a-z]\b",
            score=0.75,
        ),
    ]
    return PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=patterns,
        context=["pan", "permanent account", "tax", "income tax"],
        supported_language="en",
    )


def register_india_recognizers(analyzer_engine) -> None:
    """Register India-specific recognizers with a Presidio analyzer engine."""
    analyzer_engine.registry.add_recognizer(get_aadhaar_recognizer())
    analyzer_engine.registry.add_recognizer(get_pan_recognizer())

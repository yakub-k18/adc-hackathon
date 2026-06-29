"""Application configuration and constants."""

from __future__ import annotations

import os
from typing import Final

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Classification labels
NPPI: Final[str] = "NPPI"
SENSITIVE_NPPI: Final[str] = "SENSITIVE_NPPI"
NON_NPPI: Final[str] = "NON_NPPI"

CLASSIFICATION_LABELS: Final[list[str]] = [NPPI, SENSITIVE_NPPI, NON_NPPI]

# Risk levels
RISK_LOW: Final[str] = "LOW"
RISK_MEDIUM: Final[str] = "MEDIUM"
RISK_HIGH: Final[str] = "HIGH"
RISK_CRITICAL: Final[str] = "CRITICAL"

RISK_LEVELS: Final[list[str]] = [RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL]

# Supported file extensions
STRUCTURED_EXTENSIONS: Final[set[str]] = {".csv", ".xlsx"}
UNSTRUCTURED_EXTENSIONS: Final[set[str]] = {".txt", ".pdf", ".docx"}
SUPPORTED_EXTENSIONS: Final[set[str]] = STRUCTURED_EXTENSIONS | UNSTRUCTURED_EXTENSIONS

# Ollama
OLLAMA_BASE_URL: Final[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT: Final[str] = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_DEFAULT_MODEL: Final[str] = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS: Final[int] = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_CONNECT_TIMEOUT: Final[int] = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
OLLAMA_WARMUP_TIMEOUT: Final[int] = int(os.getenv("OLLAMA_WARMUP_TIMEOUT", "300"))
OLLAMA_KEEP_ALIVE: Final[str] = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
OLLAMA_MAX_COLUMNS_PER_BATCH: Final[int] = int(os.getenv("OLLAMA_MAX_COLUMNS_PER_BATCH", "2"))
OLLAMA_MAX_ENTITIES_PER_BATCH: Final[int] = int(os.getenv("OLLAMA_MAX_ENTITIES_PER_BATCH", "8"))

# LLM providers
PROVIDER_OLLAMA: Final[str] = "ollama"
PROVIDER_OPENAI: Final[str] = "openai"
PROVIDER_ANTHROPIC: Final[str] = "anthropic"
LLM_PROVIDERS: Final[list[str]] = [PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC]
DEFAULT_LLM_PROVIDER: Final[str] = os.getenv("LLM_PROVIDER", PROVIDER_OLLAMA)

OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: Final[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_DEFAULT_MODEL: Final[str] = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS: Final[int] = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

ANTHROPIC_API_KEY: Final[str] = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: Final[str] = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
ANTHROPIC_DEFAULT_MODEL: Final[str] = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-haiku-latest")
ANTHROPIC_TIMEOUT_SECONDS: Final[int] = int(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "60"))

CLOUD_MAX_COLUMNS_PER_BATCH: Final[int] = int(os.getenv("CLOUD_MAX_COLUMNS_PER_BATCH", "10"))
CLOUD_MAX_ENTITIES_PER_BATCH: Final[int] = int(os.getenv("CLOUD_MAX_ENTITIES_PER_BATCH", "20"))

DEFAULT_MODEL_BY_PROVIDER: Final[dict[str, str]] = {
    PROVIDER_OLLAMA: OLLAMA_DEFAULT_MODEL,
    PROVIDER_OPENAI: OPENAI_DEFAULT_MODEL,
    PROVIDER_ANTHROPIC: ANTHROPIC_DEFAULT_MODEL,
}

# Analysis defaults
MAX_SAMPLE_VALUES: Final[int] = 3
CONTEXT_WINDOW_CHARS: Final[int] = 80
MAX_UNSTRUCTURED_BATCH: Final[int] = 8
MAX_TEXT_PREVIEW_CHARS: Final[int] = 1500
MAX_LLM_ENTITIES: Final[int] = 12
HEURISTIC_SKIP_LLM_CONFIDENCE: Final[float] = 75.0

# LLM performance
OLLAMA_NUM_PREDICT: Final[int] = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
OLLAMA_NUM_CTX: Final[int] = int(os.getenv("OLLAMA_NUM_CTX", "1024"))

# Speed modes: fast | balanced | full
SPEED_FAST: Final[str] = "fast"
SPEED_BALANCED: Final[str] = "balanced"
SPEED_FULL: Final[str] = "full"
DEFAULT_SPEED_MODE: Final[str] = SPEED_BALANCED

# Risk scoring thresholds
RISK_THRESHOLDS: Final[dict[str, int]] = {
    RISK_LOW: 25,
    RISK_MEDIUM: 50,
    RISK_HIGH: 75,
    RISK_CRITICAL: 90,
}

SENSITIVE_ENTITY_TYPES: Final[set[str]] = {
    "IN_AADHAAR",
    "IN_PAN",
    "CREDIT_CARD",
    "US_PASSPORT",
    "US_BANK_NUMBER",
    "IBAN_CODE",
}

NPPI_ENTITY_TYPES: Final[set[str]] = {
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",
}

NON_NPPI_ENTITY_TYPES: Final[set[str]] = {
    "URL",
    "IP_ADDRESS",
    "DATE_TIME",
    "NRP",
}

# Heuristic column name keywords
HEURISTIC_NPPI_KEYWORDS: Final[list[str]] = [
    "name",
    "customer",
    "client",
    "phone",
    "mobile",
    "contact",
    "email",
    "address",
    "city",
    "employee_name",
]

HEURISTIC_SENSITIVE_KEYWORDS: Final[list[str]] = [
    "aadhaar",
    "aadhar",
    "pan",
    "passport",
    "credit",
    "card",
    "bank",
    "ssn",
    "salary",
]

HEURISTIC_NON_NPPI_KEYWORDS: Final[list[str]] = [
    "product",
    "sku",
    "inventory",
    "item",
    "order",
    "warehouse",
    "supplier",
    "quantity",
    "stock",
    "employee_id",
    "emp_id",
    "system_id",
]

APP_TITLE: Final[str] = "ADC – AI Powered Automated Data Classification"
APP_SUBTITLE: Final[str] = "Context-aware NPPI detection across structured and unstructured data"

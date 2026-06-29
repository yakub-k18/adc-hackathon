# ADC – AI Powered Automated Data Classification

Context-aware **NPPI** detection prototype for structured and unstructured enterprise data. Combines **Microsoft Presidio** with **Ollama (Llama 3.1)** to reduce false positives using column and document context.

## Overview

ADC analyzes uploaded files and classifies detected data as:

- **NPPI** — names, phone numbers, emails, locations
- **SENSITIVE_NPPI** — Aadhaar, PAN, passport, credit card, bank-like numbers
- **NON_NPPI** — product IDs, SKUs, order IDs, internal business identifiers

The app assigns confidence scores, AI explanations, and file-level risk (**LOW / MEDIUM / HIGH / CRITICAL**).

## Architecture

```
Upload → File Processor → Presidio (entities) → Ollama (context validation) → Risk Scoring → Streamlit Dashboard
```

See [docs/architecture.md](docs/architecture.md) for details.

## Features

- Structured: CSV, Excel (`.xlsx`)
- Unstructured: TXT, PDF, DOCX
- Custom India recognizers (Aadhaar, PAN)
- Heuristic fallback when Ollama is unavailable
- Summary metrics, findings table, Plotly chart, raw preview

## Folder structure

```
adc-prototype/
├── app.py
├── requirements.txt
├── core/
├── services/
├── recognizers/
├── sample_data/
└── docs/
```

## Installation

### 1. Python environment

Requires **Python 3.11+**.

```powershell
cd C:\Users\hp\adc-prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install Ollama (optional but recommended)

1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Install and start Ollama
3. Pull the model:

```powershell
ollama pull llama3.1:8b
```

Verify:

```powershell
ollama list
```

### 3. Generate sample HR document

```powershell
python sample_data\generate_sample_hr.py
```

## Run the app

```powershell
cd C:\Users\hp\adc-prototype
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Sample files to test

| File | Expected result |
|------|-----------------|
| `sample_data/sample_customer.csv` | NPPI columns (name, phone, email) |
| `sample_data/sample_inventory.csv` | Mostly NON_NPPI |
| `sample_data/sample_customer.txt` | Entity-level NPPI + SENSITIVE_NPPI |
| `sample_data/sample_hr.docx` | NPPI + SENSITIVE_NPPI, HIGH/CRITICAL risk |

## Configuration

Copy `.env.example` to `.env` (optional):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=120
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Ollama unavailable | App falls back to Presidio + heuristics; start Ollama for AI validation |
| `en_core_web_sm` missing | Run `python -m spacy download en_core_web_sm` |
| PDF empty text | PDF may be scanned/image-based; try TXT or DOCX |
| Slow first run | Presidio/spaCy models load on first analysis |
| Windows path errors | Run commands from project root with venv activated |

## Hackathon demo flow

1. Upload `sample_inventory.csv` → show NON_NPPI product columns
2. Upload `sample_customer.csv` → show NPPI phone/name/email with context reasoning
3. Upload `sample_hr.docx` → show SENSITIVE_NPPI (PAN) + HIGH risk

## License

Hackathon prototype — synthetic data only.

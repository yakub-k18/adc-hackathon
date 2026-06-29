"""ADC – AI Powered Automated Data Classification (Streamlit app)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_MODEL_BY_PROVIDER,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    SPEED_BALANCED,
    SPEED_FAST,
    SPEED_FULL,
)
from core.utils import findings_to_dataframe_rows
from services.classifier_service import ClassifierService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def configure_page() -> None:
    st.set_page_config(
        page_title="ADC – Automated Data Classification",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown(
        "Combine **Microsoft Presidio** entity detection with **AI context validation** "
        "(Ollama, OpenAI, or Claude) to classify data as **NPPI**, **SENSITIVE_NPPI**, or **NON_NPPI**."
    )


def render_sidebar() -> dict:
    st.sidebar.header("Analysis Controls")

    uploaded_file = st.sidebar.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "txt", "pdf", "docx"],
        help="Supported: CSV, Excel, TXT, PDF, DOCX",
    )

    mode = st.sidebar.selectbox(
        "Analysis mode",
        options=["Auto detect file type", "Structured data", "Unstructured data"],
        index=0,
    )

    use_ai = st.sidebar.toggle("Use AI validation", value=True)

    provider_labels = {
        "Ollama (local, free)": PROVIDER_OLLAMA,
        "OpenAI (cloud, fast)": PROVIDER_OPENAI,
        "Claude (cloud, fast)": PROVIDER_ANTHROPIC,
    }
    provider_label = st.sidebar.selectbox(
        "LLM provider",
        options=list(provider_labels.keys()),
        index=0,
        help="OpenAI/Claude are usually much faster than local Ollama on CPU. Requires API key.",
    )
    llm_provider = provider_labels[provider_label]

    default_model = DEFAULT_MODEL_BY_PROVIDER[llm_provider]
    model_name = st.sidebar.text_input(
        "Model name",
        value=default_model,
        help=(
            "Ollama: llama3.2:3b | OpenAI: gpt-4o-mini | Claude: claude-3-5-haiku-latest"
        ),
    )

    speed_mode = st.sidebar.selectbox(
        "Analysis speed",
        options=["Balanced (recommended)", "Fast (no AI)", "Full (AI for all)"],
        index=0,
        help="Balanced skips AI for obvious columns. Fast uses Presidio only. Full sends more to Ollama.",
    )

    show_raw = st.sidebar.toggle("Show raw extracted text", value=True)
    show_nppi_only = st.sidebar.toggle("Show only NPPI", value=False)

    speed_map = {
        "Balanced (recommended)": SPEED_BALANCED,
        "Fast (no AI)": SPEED_FAST,
        "Full (AI for all)": SPEED_FULL,
    }

    analyze_clicked = st.sidebar.button("Analyze File", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("Sample files")
    if SAMPLE_DIR.exists():
        for sample in sorted(SAMPLE_DIR.glob("*")):
            if sample.is_file() and not sample.name.startswith("."):
                st.sidebar.text(f"• {sample.name}")

    mode_map = {
        "Auto detect file type": "auto",
        "Structured data": "structured",
        "Unstructured data": "unstructured",
    }

    return {
        "uploaded_file": uploaded_file,
        "mode": mode_map[mode],
        "use_ai": use_ai,
        "show_raw": show_raw,
        "show_nppi_only": show_nppi_only,
        "model_name": model_name.strip() or default_model,
        "llm_provider": llm_provider,
        "speed_mode": speed_map[speed_mode],
        "analyze_clicked": analyze_clicked,
    }


def filter_findings(result, show_nppi_only: bool):
    findings = result.findings
    if show_nppi_only:
        findings = [f for f in findings if f.classification in {"NPPI", "SENSITIVE_NPPI"}]
    return findings


def render_summary_cards(result) -> None:
    cols = st.columns(5)
    cols[0].metric("Total Findings", result.summary.total_findings)
    cols[1].metric("NPPI", result.summary.nppi_count)
    cols[2].metric("Sensitive NPPI", result.summary.sensitive_nppi_count)
    cols[3].metric("Non-NPPI", result.summary.non_nppi_count)
    cols[4].metric("Risk Level", result.risk_level, delta=f"Score {result.risk_score}")


def render_risk_chart(findings) -> None:
    if not findings:
        st.info("No findings to chart.")
        return

    counts = {"NPPI": 0, "SENSITIVE_NPPI": 0, "NON_NPPI": 0}
    for finding in findings:
        if finding.classification in counts:
            counts[finding.classification] += 1

    chart_df = pd.DataFrame(
        {"Classification": list(counts.keys()), "Count": list(counts.values())}
    )
    fig = px.pie(
        chart_df,
        names="Classification",
        values="Count",
        title="Findings by Classification",
        color="Classification",
        color_discrete_map={
            "NPPI": "#2563eb",
            "SENSITIVE_NPPI": "#dc2626",
            "NON_NPPI": "#16a34a",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def render_findings_table(findings) -> None:
    if not findings:
        st.warning("No findings matched the current filters.")
        return

    rows = findings_to_dataframe_rows(findings)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_preview(result, show_raw: bool) -> None:
    if not show_raw:
        return

    st.subheader("Extracted Text / Data Preview")
    if result.file_type == "structured":
        if isinstance(result.raw_preview, pd.DataFrame):
            st.dataframe(result.raw_preview, use_container_width=True)
        else:
            st.write(result.raw_preview)
    else:
        st.text_area(
            "Extracted text",
            value=str(result.raw_preview or result.extracted_text or ""),
            height=260,
        )


def render_ai_panel(result, findings) -> None:
    st.subheader("AI Reasoning Panel")
    payload = {
        "file_name": result.file_name,
        "file_type": result.file_type,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "risk_explanation": result.risk_explanation,
        "summary": result.summary.to_dict(),
        "findings": [f.to_dict() for f in findings],
        "ai_warnings": result.ai_warnings,
    }
    with st.expander("View structured analysis JSON", expanded=False):
        st.json(payload)


def render_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def get_classifier(model_name: str, llm_provider: str, use_ai: bool) -> ClassifierService:
    cache_key = f"classifier_{llm_provider}_{model_name}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = ClassifierService(
            model_name=model_name,
            provider=llm_provider,
        )
    classifier = st.session_state[cache_key]

    if use_ai and llm_provider == PROVIDER_OLLAMA:
        warm_key = f"warmed_{llm_provider}_{model_name}"
        if not st.session_state.get(warm_key):
            with st.spinner("Loading Ollama model (first run may take 1–2 min)…"):
                classifier.llm.warmup()
            st.session_state[warm_key] = True

    return classifier


def run_analysis(uploaded_file, controls: dict):
    file_bytes = uploaded_file.getvalue()
    buffer = io.BytesIO(file_bytes)

    with st.spinner("Analyzing file… this may take a few seconds in Balanced/Fast mode"):
        classifier = get_classifier(
            controls["model_name"],
            controls["llm_provider"],
            controls["use_ai"],
        )
        result = classifier.analyze_uploaded_file(
            uploaded_file=buffer,
            file_name=uploaded_file.name,
            use_ai_validation=controls["use_ai"],
            mode=controls["mode"],
            speed_mode=controls["speed_mode"],
        )
    return result


def main() -> None:
    configure_page()
    render_header()
    controls = render_sidebar()

    if controls["analyze_clicked"]:
        if not controls["uploaded_file"]:
            st.error("Please upload a file before analyzing.")
            return

        try:
            result = run_analysis(controls["uploaded_file"], controls)
        except Exception as exc:
            logger.exception("Analysis failed")
            st.error(f"Analysis failed: {exc}")
            return

        findings = filter_findings(result, controls["show_nppi_only"])

        if result.ai_warnings:
            render_warnings(result.ai_warnings)

        st.success(
            f"Analysis complete for **{result.file_name}** "
            f"({result.file_type}) — Risk: **{result.risk_level}**"
        )

        render_summary_cards(result)
        st.divider()

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Findings")
            render_findings_table(findings)
        with right:
            render_risk_chart(findings)

        render_preview(result, controls["show_raw"])
        render_ai_panel(result, findings)

        if result.risk_explanation:
            st.info(f"**Risk rationale:** {result.risk_explanation}")

    else:
        st.markdown(
            """
            ### Getting started
            1. Upload a CSV, Excel, TXT, PDF, or DOCX file
            2. For **fastest AI**, choose **OpenAI** or **Claude** and add API keys to `.env`
            3. Set **Analysis speed** to **Balanced** or **Fast** for quicker results
            4. Click **Analyze File**

            Try the sample files in `sample_data/` such as:
            - `sample_customer.csv` → expect NPPI columns
            - `sample_inventory.csv` → expect mostly NON_NPPI
            - `sample_customer.txt` → entity-level NPPI
            - `sample_hr.docx` → NPPI + SENSITIVE_NPPI
            """
        )


if __name__ == "__main__":
    main()

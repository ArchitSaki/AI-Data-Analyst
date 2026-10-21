import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import requests

from app.services.data_loader import CSVValidationError, load_csv

import io
import os
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = (
    st.secrets.get("API_BASE_URL", "")
    or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
).rstrip("/")

API_URL = f"{API_BASE_URL}/api/ask"

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

# Keep conversation data across Streamlit reruns.
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "business_insights" not in st.session_state:
    st.session_state.business_insights = None

def render_chart(chart_data):
    """Render a Plotly chart returned by the API, if available."""
    if not chart_data:
        return

    try:
        figure = go.Figure(chart_data)
        st.plotly_chart(
            figure,
            use_container_width=True,
        )
    except (ValueError, TypeError, KeyError, RuntimeError):
        st.warning("The chart could not be displayed.")


def render_data_quality(details):
    """Display the data-quality profile returned by the API."""
    st.markdown("#### Data quality report")

    report = details.get("quality_report")

    if not report:
        st.info("No data-quality report is available.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", report.get("row_count", 0))

    with col2:
        st.metric("Columns", report.get("column_count", 0))

    with col3:
        st.metric(
            "Completeness",
            f"{report.get('completeness_percentage', 0):.2f}%",
        )

    with col4:
        st.metric(
            "Duplicate rows",
            report.get("duplicate_rows", 0),
        )

    st.metric(
        "Missing cells",
        report.get("missing_cells", 0),
    )

    # Show statistics for every column.
    column_details = report.get("columns", [])

    if column_details:
        st.markdown("**Column-level profile**")
        st.dataframe(
            pd.DataFrame(column_details),
            use_container_width=True,
        )

    # Highlight only columns that contain missing values.
    missing_by_column = report.get("missing_by_column", {})

    missing_columns = {
        name: count
        for name, count in missing_by_column.items()
        if count > 0
    }

    if missing_columns:
        st.markdown("**Columns with missing values**")

        missing_df = pd.DataFrame(
            [
                {
                    "Column": name,
                    "Missing values": count,
                }
                for name, count in missing_columns.items()
            ]
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
        )
    else:
        st.success("No missing values were detected.")

    st.caption(
        "These checks identify potential data-quality issues. "
        "Missing values and duplicate rows should be reviewed in context."
    )



def render_business_insights(details: dict) -> None:
    insights = details.get("insights", [])

    st.subheader("Business Insights")
    st.caption(
        f"Analyzed {details.get('row_count', 0):,} rows "
        f"across {details.get('column_count', 0)} columns."
    )

    business_summary = details.get("business_summary")

    if business_summary:
        st.markdown("### AI Business Analysis")
        st.write(business_summary)

    if not insights:
        st.info("No meaningful insights could be generated for this dataset.")
        return

    for insight in insights:
        insight_type = insight.get("type", "")
        evidence = insight.get("evidence", {})

        st.markdown(f"### {insight.get('title', 'Finding')}")
        st.write(insight.get("explanation", ""))

        # Compare categories, such as quantity by customer.
        if insight_type == "category_performance":
            groups = evidence.get("all_groups", [])

            if groups:
                chart_df = pd.DataFrame(groups)

                if {"group", "value"}.issubset(chart_df.columns):
                    chart_df = chart_df.dropna(subset=["value"])
                    chart_df["value"] = pd.to_numeric(
                        chart_df["value"], errors="coerce"
                    )
                    chart_df = chart_df.dropna(subset=["value"])

                    chart_df = chart_df.rename(
                        columns={"group": "Category", "value": "Total"}
                    )

                    if not chart_df.empty:
                        st.bar_chart(
                            chart_df.set_index("Category")["Total"]
                        )

            highest = evidence.get("highest", {})
            lowest = evidence.get("lowest", {})

            if highest and lowest:
                difference = highest["value"] - lowest["value"]
                st.write(
                    f"**What stands out:** "
                    f"{highest['group']} is ahead of "
                    f"{lowest['group']} by {difference:,.4g}."
                )

        # Show changes over time when period-level values are available.
        elif insight_type == "time_trend":
            monthly_values = evidence.get("monthly_values", [])

            if monthly_values:
                trend_df = pd.DataFrame(monthly_values)

                trend_df = trend_df.rename(
                    columns={
                        "period": "Month",
                        "value": "Total",
                    }
                )

                st.line_chart(
                    trend_df.set_index("Month")["Total"]
                )

        elif insight_type == "numeric_distribution":
            histogram = evidence.get("histogram", [])

            if histogram:
                histogram_df = pd.DataFrame(histogram)
                histogram_df = histogram_df.rename(
                    columns={"range": "Value range", "count": "Records"}
                )
                st.bar_chart(
                    histogram_df.set_index("Value range")["Records"]
                )

        # Keep raw technical data out of the main report.
        with st.expander("View supporting data"):
            st.json(evidence)

        st.divider()



def render_anomaly_details(details):
    """Display anomaly metrics and the original flagged records."""
    st.markdown("#### Anomaly detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Anomalies detected",
            details.get("anomaly_count", 0),
        )

    with col2:
        st.metric(
            "Column analyzed",
            details.get("column") or "Not selected",
        )

    with col3:
        st.metric(
            "Detection method",
            details.get("method", "IQR"),
        )

    lower_bound = details.get("lower_bound")
    upper_bound = details.get("upper_bound")

    if lower_bound is not None and upper_bound is not None:
        bound1, bound2 = st.columns(2)

        with bound1:
            st.metric("Lower bound", f"{lower_bound:.4g}")

        with bound2:
            st.metric("Upper bound", f"{upper_bound:.4g}")

    anomalies = details.get("anomalies", [])

    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)

        # Expand original record fields into separate columns.
        if "record" in anomaly_df.columns:
            records_df = pd.json_normalize(
                anomaly_df["record"]
            )

            anomaly_df = pd.concat(
                [
                    anomaly_df.drop(columns=["record"]).reset_index(
                        drop=True
                    ),
                    records_df.reset_index(drop=True),
                ],
                axis=1,
            )

        st.markdown("**Detected outlier records**")

        st.dataframe(
            anomaly_df,
            use_container_width=True,
        )

    elif details.get("column") is None:
        st.info(
            "Specify a numeric column, for example: "
            "'Find anomalies in revenue'."
        )
    else:
        st.info("No anomaly records were returned.")

    st.caption(
        "The IQR method flags statistically unusual values. "
        "An outlier is not necessarily an error."
    )


def render_sql_details(details):
    """Display generated SQL, query results, and an optional chart."""
    with st.expander("View generated SQL"):
        st.code(
            details.get("sql", ""),
            language="sql",
        )

    with st.expander("View query results"):
        columns = details.get("columns", [])
        results = details.get("results", [])

        if columns:
            results_df = pd.DataFrame(
                results,
                columns=columns,
            )
        else:
            results_df = pd.DataFrame(results)

        st.dataframe(
            results_df,
            use_container_width=True,
        )

    render_chart(details.get("chart"))



def render_result_details(details):
    """Route result rendering according to the analysis type."""
    analysis_type = details.get("analysis_type", "sql")

    if analysis_type == "data_quality":
        render_data_quality(details)
        return

    if analysis_type == "anomaly_detection":
        render_anomaly_details(details)
        return

    if analysis_type == "business_insights":
        render_business_insights(details)
        return

    render_sql_details(details)




def build_result_details(result):
    """Preserve all analysis-specific fields in conversation history."""
    return {
        "analysis_type": result.get("analysis_type", "sql"),

        # SQL analysis
        "sql": result.get("sql", ""),
        "columns": result.get("columns", []),
        "results": result.get("results", []),
        "chart": result.get("chart"),

        # Anomaly detection
        "column": result.get("column"),
        "method": result.get("method"),
        "anomaly_count": result.get("anomaly_count", 0),
        "anomalies": result.get("anomalies", []),
        "lower_bound": result.get("lower_bound"),
        "upper_bound": result.get("upper_bound"),
        "valid_values": result.get("valid_values"),

        # Data quality
        "quality_report": result.get("quality_report"),

        # Business insights
        "insights": result.get("insights", []),
        "insight_count": result.get("insight_count", 0),
        "row_count": result.get("row_count", 0),
        "column_count": result.get("column_count", 0),
        "business_summary": result.get("business_summary"),
    }


# ---------------------------------------------------------------------
# Application header
# ---------------------------------------------------------------------

st.title("📊 AI Data Analyst")

st.caption(
    "Upload a CSV, ask questions in natural language, "
    "and inspect SQL, results, visualizations, and data quality."
)


# ---------------------------------------------------------------------
# Sidebar: dataset upload and metadata
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Dataset")

    
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files to analyze.",
    )

    selected_file = None
    preview_df = None

    
    selected_file = None
    preview_df = None
    valid_files = {}

    if uploaded_files:
        for uploaded in uploaded_files:
            try:
                file_bytes = uploaded.getvalue()
                dataframe = load_csv(file_bytes)

                if dataframe.empty:
                    st.warning(
                        f"{uploaded.name}: the CSV contains no data rows."
                    )
                    continue

                valid_files[uploaded.name] = {
                    "bytes": file_bytes,
                    "dataframe": dataframe,
                }

            except CSVValidationError as exc:
                st.warning(f"{uploaded.name}: {exc}")

            except Exception as exc:
                st.warning(
                    f"Could not load {uploaded.name}: {exc}"
                )

    if valid_files:
        analysis_mode = st.radio(
            "Analysis mode",
            options=["Single dataset", "All datasets"],
            index=0,
            help=(
                "Single dataset analyzes one CSV. All datasets lets "
                "the AI query across uploaded CSVs."
            ),
        )

        selected_name = st.selectbox(
            "Choose a dataset to preview",
            options=list(valid_files.keys()),
        )

        selected_file = valid_files[selected_name]
        preview_df = selected_file["dataframe"]

        st.session_state.active_analysis_mode = analysis_mode
        st.session_state.active_datasets = valid_files

        # A changed dataset selection or file set starts a fresh conversation.
        current_signature = tuple(
            (name, item["bytes"])
            for name, item in valid_files.items()
        )

        if (
            st.session_state.get("dataset_signature")
            != current_signature
            or st.session_state.get("last_analysis_mode")
            != analysis_mode
        ):
            st.session_state.dataset_signature = current_signature
            st.session_state.last_analysis_mode = analysis_mode
            st.session_state.uploaded_file_name = selected_name
            st.session_state.uploaded_file_bytes = selected_file["bytes"]
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.business_insights = None

        else:
            st.session_state.uploaded_file_name = selected_name
            st.session_state.uploaded_file_bytes = selected_file["bytes"]

        st.success(f"Previewing: {selected_name}")
        st.metric("Rows", f"{len(preview_df):,}")
        st.metric("Columns", len(preview_df.columns))

        if analysis_mode == "All datasets":
            st.info(
                f"{len(valid_files)} datasets available for cross-file analysis."
            )


    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by FastAPI, LangGraph, and Groq.")


# Generate and display business insights once per uploaded file.
if preview_df is not None and st.session_state.business_insights is None:
    try:
        insight_files = {
            "file": (
                st.session_state.uploaded_file_name,
                st.session_state.uploaded_file_bytes,
                "text/csv",
            )
        }

        insight_form = {}

        if st.session_state.conversation_id:
            insight_form["conversation_id"] = (
                st.session_state.conversation_id
            )

        with st.spinner("Generating business insights..."):
            response = requests.post(
                f"{API_BASE_URL}/api/insights",
                files=insight_files,
                data=insight_form,
                timeout=120,
            )

        if response.ok:
            report = response.json()

            st.session_state.conversation_id = (
                report["conversation_id"]
            )
            st.session_state.business_insights = report
        else:
            st.warning(
                "Could not generate business insights: "
                f"{response.text}"
            )

    except requests.RequestException as exc:
        st.warning(f"Could not load business insights: {exc}")


if st.session_state.business_insights:
    with st.expander("Business insights for your dataset", expanded=True):
        render_business_insights(
            st.session_state.business_insights
        )


# ---------------------------------------------------------------------
# Dataset preview and column information
# ---------------------------------------------------------------------

if preview_df is not None:
    tab_preview, tab_schema = st.tabs(
        ["Data preview", "Column information"]
    )

    with tab_preview:
        st.dataframe(
            preview_df.head(100),
            use_container_width=True,
        )

    with tab_schema:
        schema_df = pd.DataFrame(
            {
                "Column": preview_df.columns,
                "Data type": [
                    str(dtype) for dtype in preview_df.dtypes
                ],
                "Missing values": [
                    int(count)
                    for count in preview_df.isna().sum()
                ],
            }
        )

        st.dataframe(
            schema_df,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------

st.divider()
st.subheader("Ask your data")

if not st.session_state.get("active_datasets"):
    st.info(
        "Upload a CSV from the sidebar to start analyzing your data."
    )


# Render previous messages and their saved analysis details.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("details")
        ):
            render_result_details(message["details"])


question = st.chat_input(
    "Ask a question about your dataset...",
    disabled=not bool(st.session_state.get("active_datasets")),
)


# ---------------------------------------------------------------------
# Submit a question to the FastAPI backend
# ---------------------------------------------------------------------

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            try:
                active_mode = st.session_state.get(
                    "active_analysis_mode",
                    "Single dataset",
                )

                if active_mode == "All datasets":
                    files = [
                        (
                            "files",
                            (name, item["bytes"], "text/csv"),
                        )
                        for name, item in st.session_state.active_datasets.items()
                    ]
                else:
                    files = {
                        "file": (
                            st.session_state.uploaded_file_name,
                            st.session_state.uploaded_file_bytes,
                            "text/csv",
                        )
                    }
                
                response = requests.post(
                API_URL,
                files=files,
                data={
                    "question": question,
                    "conversation_id": st.session_state.conversation_id,
                    },
                timeout=120,
                )

                if not response.ok:
                    st.error(
                        f"API error ({response.status_code}): {response.text}"
                    )
                    st.stop()

                result = response.json()

                st.session_state.conversation_id = result.get(
                    "conversation_id",
                    st.session_state.conversation_id,
                )

                answer = result.get("answer", "No answer was returned.")
                details = build_result_details(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "details": details,
                    }
                )

                st.markdown(answer)
                render_result_details(details)

            except requests.RequestException as exc:
                st.error(
                    "Could not connect to the API. "
                    "Make sure FastAPI is running. "
                    f"Details: {exc}"
                )

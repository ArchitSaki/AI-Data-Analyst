
import json

import pandas as pd
import uuid
from app.services.llm_service import ask_groq
from app.services.sql_engine import execute_sql, normalize_table_name
from app.services.charting import create_chart
from app.services.business_insights import generate_business_insights
import re

from app.services.dataset_store import (
    save_dataset,
    save_datasets,
    get_business_insights,
)


from app.services.analytics import (
    detect_anomalies,
    profile_data_quality,
    AnalyticsError,
)


SQL_SYSTEM_PROMPT = """
You are the SQL planning component of an AI data analyst.

Convert the user's natural-language question into one DuckDB SQL query.

Rules:
1. Use only the tables explicitly listed in the provided schema.
2. Use only columns that exist in those tables.
3. Return exactly one SQL query, with no Markdown or explanation.
4. Generate read-only SELECT queries only.
5. Never use INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER.
6. Never read external files, access other databases, or use network resources.
7. Use DuckDB-compatible SQL.
8. Use clear output column aliases.
9. Handle missing values sensibly and avoid unsupported assumptions.
10. If the question is ambiguous, make a reasonable assumption only
    when the schema supports it. Otherwise, ask for clarification
    instead of inventing a calculation.
11. Do not assume that a column named revenue, amount, price, or date
    has a meaning beyond what the schema indicates.
12. Keep result sets reasonably small.
13. When combining tables, use JOINs only when the question requires them.
14. Join tables using compatible columns supported by the schema.
    Do not invent relationships or join keys.
15. Use table aliases when they improve readability.
16. Never reference a table that is not listed in the supplied schema.

Output only the SQL query.
"""



EXPLANATION_SYSTEM_PROMPT = """
You are the explanation component of an AI data analyst.

Explain the result of the executed SQL in clear, concise language.
Use only the supplied question, query, and actual query result.
Do not invent numbers, causes, trends, or business explanations.
If the result is empty, say so.
If the result is limited, mention that when relevant.
Briefly explain the calculation when it helps the user.
"""



BUSINESS_SUMMARY_SYSTEM_PROMPT = """
You are a business analyst explaining data findings to a non-technical user.

You will receive findings calculated by a deterministic analytics system.

Rules:
1. Use only the supplied findings and evidence.
2. Explain the most important patterns in simple, clear language.
3. Mention specific metrics and values when useful.
4. Explain why a finding may matter to a business, but distinguish
   possible implications from proven facts.
5. Never invent causes, trends, profits, or recommendations unsupported
   by the evidence.
6. Do not repeat every finding. Prioritize the most useful observations.
7. If evidence is insufficient for a conclusion, say so.
8. Suggest at most two useful follow-up questions.
9. Keep the response concise and readable.
10. Return plain text, not JSON or Markdown tables.
"""



def get_schema(df: pd.DataFrame) -> list[dict]:
    """Describe columns without sending the full dataset to the LLM."""

    return [
        {
            "column": str(column),
            "data_type": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
        }
        for column in df.columns
    ]


def generate_sql(
    df: pd.DataFrame,
    question: str,
    previous_error: str | None = None,
) -> str:
    schema = get_schema(df)

    prompt = {
        "question": question,
        "table_name": "uploaded_data",
        "schema": schema,
    }

    if previous_error:
        prompt["previous_attempt_error"] = previous_error
        prompt["instruction"] = (
            "Correct the previous query based on the error. "
            "Return a new SQL query, not an explanation."
        )

    return ask_groq(
        system_prompt=SQL_SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt, ensure_ascii=False),
    )



def get_multi_schema(
    datasets: dict[str, pd.DataFrame],
) -> list[dict]:
    """Describe each uploaded table without sending its full data."""

    tables = []
    used_names = set()

    for file_name, df in datasets.items():
        table_name = normalize_table_name(file_name)

        if table_name in used_names:
            raise ValueError(
                f"Multiple files resolve to the table name '{table_name}'. "
                "Rename the files to make their names unique."
            )

        used_names.add(table_name)

        tables.append(
            {
                "table_name": table_name,
                "source_file": file_name,
                "columns": get_schema(df),
            }
        )

    return tables


def generate_sql_multi(
    datasets: dict[str, pd.DataFrame],
    question: str,
    previous_error: str | None = None,
) -> str:
    """Generate SQL using the schemas of all uploaded CSV files."""

    prompt = {
        "question": question,
        "tables": get_multi_schema(datasets),
    }

    if previous_error:
        prompt["previous_attempt_error"] = previous_error
        prompt["instruction"] = (
            "Correct the previous query based on the error. "
            "Use only the provided tables and columns. "
            "Return a new SQL query, not an explanation."
        )

    return ask_groq(
        system_prompt=SQL_SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt, ensure_ascii=False, default=str),
    )



def explain_result(
    question: str,
    execution_result: dict,
) -> str:
    prompt = {
        "question": question,
        "executed_sql": execution_result["sql"],
        "result_columns": execution_result["columns"],
        "result_rows": execution_result["rows"][:100],
        "returned_row_count": execution_result["row_count"],
        "results_truncated": execution_result["truncated"],
    }

    return ask_groq(
        system_prompt=EXPLANATION_SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt, ensure_ascii=False, default=str),
    )


def generate_business_summary(report: dict) -> str:
    """Explain calculated business insights using the Groq LLM."""

    prompt = {
        "row_count": report["row_count"],
        "column_count": report["column_count"],
        "findings": [
            {
                "type": insight["type"],
                "title": insight["title"],
                "explanation": insight["explanation"],
                "evidence": insight["evidence"],
            }
            for insight in report["insights"][:20]
        ],
    }

    return ask_groq(
        system_prompt=BUSINESS_SUMMARY_SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt, ensure_ascii=False, default=str),
    )



def _find_anomaly_column(df: pd.DataFrame, question: str) -> str | None:
    """Identify a numeric column explicitly mentioned in an anomaly request."""
    question_lower = question.lower()

    candidates = [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
        and re.search(
            rf"(?<!\w){re.escape(str(column).lower())}(?!\w)",
            question_lower,
        )
    ]

    # Avoid silently choosing a column if the user mentioned several.
    if len(candidates) != 1:
        return None

    return candidates[0]


def _analyze_anomalies(
    df: pd.DataFrame,
    question: str,
    conversation_id: str,
) -> dict:
    """Run deterministic anomaly detection without generating SQL."""
    column = _find_anomaly_column(df, question)

    if column is None:
        numeric_columns = [
            str(name)
            for name in df.columns
            if pd.api.types.is_numeric_dtype(df[name])
            and not pd.api.types.is_bool_dtype(df[name])
        ]

        if not numeric_columns:
            answer = (
                "This dataset has no numeric columns suitable for "
                "IQR-based anomaly detection."
            )
        else:
            answer = (
                "Which numeric column should I check for anomalies? "
                f"Available columns: {', '.join(numeric_columns)}."
            )

        return {
            "conversation_id": conversation_id,
            "question": question,
            "analysis_type": "anomaly_detection",
            "answer": answer,
            "column": None,
            "method": "IQR",
            "anomaly_count": 0,
            "anomalies": [],
            "lower_bound": None,
            "upper_bound": None,
            "chart": None,
        }

    try:
        result = detect_anomalies(df, column)
    except AnalyticsError as exc:
        return {
            "conversation_id": conversation_id,
            "question": question,
            "analysis_type": "anomaly_detection",
            "answer": str(exc),
            "column": column,
            "method": "IQR",
            "anomaly_count": 0,
            "anomalies": [],
            "lower_bound": None,
            "upper_bound": None,
            "chart": None,
        }

    count = result["anomaly_count"]

    answer = (
        f"Found {count} statistical outlier(s) in '{column}' using "
        f"the IQR method. The lower bound is "
        f"{result['lower_bound']:.4g} and the upper bound is "
        f"{result['upper_bound']:.4g}. "
        "These values are statistically unusual, but are not "
        "necessarily errors."
    )

    return {
        "conversation_id": conversation_id,
        "question": question,
        "analysis_type": "anomaly_detection",
        "answer": answer,
        "column": column,
        "method": result["method"],
        "anomaly_count": count,
        "anomalies": result["anomalies"],
        "lower_bound": result["lower_bound"],
        "upper_bound": result["upper_bound"],
        "valid_values": result["valid_values"],
        "chart": None,
    }

def _analyze_business_insights(
    df: pd.DataFrame,
    question: str,
    conversation_id: str,
) -> dict:
    """Generate deterministic, evidence-backed business insights."""

    try:
        report = generate_business_insights(df)
    except AnalyticsError as exc:
        return {
            "conversation_id": conversation_id,
            "question": question,
            "analysis_type": "business_insights",
            "answer": str(exc),
            "insight_count": 0,
            "insights": [],
            "quality_report": None,
        }

    insights = report["insights"]

    if not insights:
        answer = (
            "The dataset was analyzed, but no notable findings "
            "were identified by the current insight rules."
        )
    else:
        answer = (
            f"Analysis complete. Generated {len(insights)} findings "
            f"from {report['row_count']} rows and "
            f"{report['column_count']} columns. "
            "Review the evidence accompanying each finding."
        )

    return {
        "conversation_id": conversation_id,
        "question": question,
        "analysis_type": "business_insights",
        "answer": answer,
        "row_count": report["row_count"],
        "column_count": report["column_count"],
        "insight_count": report["insight_count"],
        "insights": insights,
        "quality_report": report["quality_report"],
    }

    
def _analyze_data_quality(
    df: pd.DataFrame,
    question: str,
    conversation_id: str,
) -> dict:
    """Profile data quality without generating SQL or calling the LLM."""

    try:
        profile = profile_data_quality(df)
    except AnalyticsError as exc:
        return {
            "conversation_id": conversation_id,
            "question": question,
            "analysis_type": "data_quality",
            "answer": str(exc),
            "quality_report": None,
        }

    missing_columns = [
        name
        for name, count in profile["missing_by_column"].items()
        if count > 0
    ]

    findings = []

    if profile["missing_cells"]:
        findings.append(
            f"{profile['missing_cells']} missing cell(s) across "
            f"{len(missing_columns)} column(s)"
        )

    if profile["duplicate_rows"]:
        findings.append(
            f"{profile['duplicate_rows']} duplicate row(s)"
        )

    if not findings:
        summary = (
            "No missing cells or duplicate rows were detected "
            "by these checks."
        )
    else:
        summary = "Data quality observations: " + "; ".join(findings) + "."

    answer = (
        f"Profiled {profile['row_count']} rows and "
        f"{profile['column_count']} columns. "
        f"Overall cell completeness is "
        f"{profile['completeness_percentage']:.2f}%. "
        f"{summary} These checks identify potential issues; "
        "they do not automatically mean the data is incorrect."
    )

    return {
        "conversation_id": conversation_id,
        "question": question,
        "analysis_type": "data_quality",
        "answer": answer,
        "quality_report": profile,
    }



def analyze_question(
    df: pd.DataFrame,
    question: str,
    conversation_id: str | None = None,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Validate input and run the checkpointed LangGraph workflow."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter an analysis question.")

    if len(question) > 2000:
        raise ValueError("The question must be 2,000 characters or fewer.")

    # Create an ID for a new conversation when one isn't provided.
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    multi_file = datasets is not None and len(datasets) > 1

    if multi_file:
        save_datasets(conversation_id, datasets)
    else:
        save_dataset(conversation_id, df)

    # Route explicit anomaly requests to deterministic statistical analysis.
    question_lower = question.lower()

    anomaly_request = any(
        phrase in question_lower
        for phrase in (
            "anomal",
            "outlier",
            "outliers",
            "unusual value",
            "unusual values",
            "unusual data",
            "detect outlier",
            "detect anomaly",
        )
    )

    if not multi_file and anomaly_request:
        return _analyze_anomalies(df, question, conversation_id)

    
    quality_request = any(
        phrase in question_lower
        for phrase in (
            "data quality",
            "quality check",
            "quality report",
            "profile this dataset",
            "profile the dataset",
            "missing values",
            "missing data",
            "duplicate rows",
            "duplicates",
        )
    )

    if not multi_file and quality_request:
        return _analyze_data_quality(
            df,
            question,
            conversation_id,
        )
        
    
    business_insights_request = any(
    phrase in question_lower
    for phrase in (
        "business insights",
        "generate insights",
        "show insights",
        "key findings",
        "analyze this dataset",
        "analyse this dataset",
        "analyze the dataset",
        "analyse the dataset",
        "overall analysis",
        "executive summary",
        "business trends",
        "trends in",
        "underperforming",
        "best performing",
        "highest performing",
        "lowest performing",
        "business opportunities",
        "business risks",
    )
)

    if not multi_file and business_insights_request:
        cached_report = get_business_insights(conversation_id)

        if cached_report is None:
            return _analyze_business_insights(
                df,
                question,
                conversation_id,
            )

        return {
            **cached_report,
            "conversation_id": conversation_id,
            "question": question,
            "analysis_type": "business_insights",
            "answer": cached_report.get(
                "business_summary",
                (
                    f"Found {cached_report['insight_count']} findings "
                    f"from {cached_report['row_count']} rows and "
                    f"{cached_report['column_count']} columns."
                ),
            ),
        }

    # Normal questions continue through the existing LangGraph workflow.
    from app.services.analysis_graph import get_analysis_graph
    

    graph = get_analysis_graph()

    result = graph.invoke(
        {
            "dataset_id": conversation_id,
            "question": question,
            "sql": "",
            "execution_result": None,
            "answer": "",
            "last_error": None,
            "attempts": 0,
            "max_attempts": 2,
            "multi_file": multi_file,
        },
        config={
            "configurable": {
                "thread_id": conversation_id,
            }
        },
    )

    execution_result = result["execution_result"]

    if execution_result is None:
        raise RuntimeError(result["answer"])
    
        # Build a chart from the actual SQL results.
    chart_figure = create_chart(
        question=question,
        columns=execution_result["columns"],
        results=execution_result["rows"],
    )

    # Convert the Plotly figure to JSON-safe data for the API.
    chart = (
        json.loads(chart_figure.to_json())
        if chart_figure is not None
        else None
    )

    return {
        "conversation_id": conversation_id,
        "question": question,
        "answer": result["answer"],
        "sql": execution_result["sql"],
        "columns": execution_result["columns"],
        "results": execution_result["rows"],
        "row_count": execution_result["row_count"],
        "truncated": execution_result["truncated"],
        "chart": chart,
        "analysis_type": "sql",
    }
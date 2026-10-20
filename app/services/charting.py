
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MAX_CHART_ROWS = 100


def _to_dataframe(columns: list, results: list) -> pd.DataFrame:
    """Convert SQL results into a DataFrame safely."""
    if not results or not columns:
        return pd.DataFrame()

    if isinstance(results[0], dict):
        return pd.DataFrame(results).reindex(columns=columns)

    return pd.DataFrame(results, columns=columns)


def create_chart(
    question: str,
    columns: list,
    results: list,
) -> go.Figure | None:
    """
    Select a suitable chart using the question and SQL result schema.

    Returns a Plotly figure, or None when the result is unsuitable
    for visualization.
    """
    df = _to_dataframe(columns, results)

    if df.empty or len(df.columns) < 2:
        return None

    question_lower = question.lower()

    # Identify numeric columns.
    numeric_columns = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]

    # Identify date-like columns.
    date_columns = []

    for col in df.columns:
        if col in numeric_columns:
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")

        if len(df[col]) > 0 and parsed.notna().mean() >= 0.8:
            date_columns.append(col)
            df[col] = parsed

    # Identify categorical columns.
    category_columns = [
        col
        for col in df.columns
        if col not in numeric_columns and col not in date_columns
    ]

    # Prefer the first numeric measure.
    y_column = numeric_columns[0] if numeric_columns else None

    # Scatter plot: compare two numeric measures.
    if (
        "scatter" in question_lower
        and len(numeric_columns) >= 2
    ):
        return px.scatter(
            df.head(MAX_CHART_ROWS),
            x=numeric_columns[0],
            y=numeric_columns[1],
            title=f"{numeric_columns[1]} vs {numeric_columns[0]}",
        )

    # Pie chart: show the composition of a measure.
    if (
        "pie" in question_lower
        and category_columns
        and y_column
    ):
        return px.pie(
            df.head(MAX_CHART_ROWS),
            names=category_columns[0],
            values=y_column,
            title=f"{y_column} by {category_columns[0]}",
        )

    # Time-series chart.
    if (
        date_columns
        and y_column
        and any(
            word in question_lower
            for word in ("trend", "over time", "monthly", "daily", "yearly", "line")
        )
    ):
        date_column = date_columns[0]
        chart_df = df.sort_values(date_column).head(MAX_CHART_ROWS)

        return px.line(
            chart_df,
            x=date_column,
            y=y_column,
            markers=True,
            title=f"{y_column} over time",
        )

    # Scatter chart requested explicitly, if two numeric fields exist.
    if "scatter" in question_lower and len(numeric_columns) >= 2:
        return px.scatter(
            df.head(MAX_CHART_ROWS),
            x=numeric_columns[0],
            y=numeric_columns[1],
        )

    # Bar chart for category comparisons.
    if category_columns and y_column:
        category = category_columns[0]

        chart_df = df.head(MAX_CHART_ROWS)

        if len(chart_df) > 20:
            chart_df = chart_df.nlargest(20, y_column)

        return px.bar(
            chart_df,
            x=category,
            y=y_column,
            title=f"{y_column} by {category}",
        )

    # A date and measure can still be visualized without an explicit
    # trend keyword.
    if date_columns and y_column:
        return px.line(
            df.sort_values(date_columns[0]).head(MAX_CHART_ROWS),
            x=date_columns[0],
            y=y_column,
            markers=True,
            title=f"{y_column} over time",
        )

    # No sensible chart could be selected.
    return None

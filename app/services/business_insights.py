import numpy as np
import pandas as pd

from app.services.analytics import (
    AnalyticsError,
    aggregate_by_group,
    detect_anomalies,
    profile_data_quality,
)


DATE_HINTS = ("date", "time", "timestamp", "created_at", "order_at")


def _json_value(value):
    """Convert common Pandas/NumPy values into JSON-friendly values."""
    if value is None or pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _numeric_columns(df):
    """Return numeric columns, excluding boolean columns."""
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(df[column])
    ]


def _category_columns(df, max_unique=30):
    """Find categorical columns with a manageable number of groups."""
    candidates = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    return [
        column
        for column in candidates
        if 2 <= df[column].nunique(dropna=True) <= max_unique
    ]


def _numeric_insights(df):
    """Summarize numeric columns and detect potential outliers."""
    insights = []

    for column in _numeric_columns(df):
        values = df[column].dropna()

        if values.empty:
            continue

        mean_value = float(values.mean())
        median_value = float(values.median())
        minimum = float(values.min())
        maximum = float(values.max())
        std_value = float(values.std()) if len(values) > 1 else 0.0
        hist_counts, bin_edges = (
            np.histogram(values, bins=min(10, values.nunique()))
            if values.nunique() > 1
            else ([len(values)], [minimum, maximum + 1])
)

        insights.append({
            "type": "numeric_distribution",
            "title": f"Distribution of {column}",
            "column": str(column),
            "evidence": {
                "valid_values": int(len(values)),
                "missing_values": int(df[column].isna().sum()),
                "mean": mean_value,
                "median": median_value,
                "minimum": minimum,
                "maximum": maximum,
                "range": maximum - minimum,
                "standard_deviation": std_value,
                "histogram": [
                    {
                        "range": f"{bin_edges[i]:.4g}–{bin_edges[i + 1]:.4g}",
                        "count": int(hist_counts[i]),
                    }
                    for i in range(len(hist_counts))
                ],
                
            },
            "explanation": (
                f"{column} has {len(values)} valid values, "
                f"with a mean of {mean_value:.4g}, median of "
                f"{median_value:.4g}, and range from "
                f"{minimum:.4g} to {maximum:.4g}."
            ),
        })

        try:
            anomaly_report = detect_anomalies(df, column)

            if anomaly_report["anomaly_count"] > 0:
                insights.append({
                    "type": "potential_outliers",
                    "title": f"Potential outliers in {column}",
                    "column": str(column),
                    "evidence": {
                        "anomaly_count": anomaly_report["anomaly_count"],
                        "valid_values": anomaly_report["valid_values"],
                        "lower_bound": anomaly_report["lower_bound"],
                        "upper_bound": anomaly_report["upper_bound"],
                        "examples": [
                            {
                                "row_index": item["row_index"],
                                "value": item["value"],
                                "type": item["type"],
                            }
                            for item in anomaly_report["anomalies"][:5]
                        ],
                    },
                    "explanation": anomaly_report["explanation"],
                })

        except AnalyticsError:
            # A column without usable values cannot be analyzed for outliers.
            continue

    return insights


def _category_insights(df):
    """Find the highest- and lowest-valued groups for numeric metrics."""
    insights = []

    numeric_columns = _numeric_columns(df)
    category_columns = _category_columns(df)

    for category_column in category_columns:
        for value_column in numeric_columns:
            try:
                report = aggregate_by_group(
                    df,
                    group_column=category_column,
                    value_column=value_column,
                    operation="sum",
                )
            except AnalyticsError:
                continue

            groups = [
                item
                for item in report["results"]
                if item["group"] is not None
            ]

            if len(groups) < 2:
                continue

            # Sorting makes the ranking deterministic.
            groups.sort(key=lambda item: item["value"])

            lowest = groups[0]
            highest = groups[-1]

            insights.append({
                "type": "category_performance",
                "title": f"{value_column} by {category_column}",
                "category_column": str(category_column),
                "value_column": str(value_column),
                "operation": "sum",
                "evidence": {
                    "highest": highest,
                    "lowest": lowest,
                    "category_count": len(groups),
                    "all_groups": groups,
                },
                "explanation": (
                    f"{highest['group']} has the highest total "
                    f"{value_column} ({highest['value']:.4g}), while "
                    f"{lowest['group']} has the lowest total "
                    f"({lowest['value']:.4g}), grouped by "
                    f"{category_column}."
                ),
            })

    return insights


def _find_date_columns(df):
    """Identify columns that can reasonably be interpreted as dates."""
    date_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            name_suggests_date = any(
                hint in str(column).lower()
                for hint in DATE_HINTS
            )

            if not name_suggests_date:
                continue

            parsed = pd.to_datetime(series, errors="coerce")
        else:
            continue

        valid_count = int(parsed.notna().sum())

        # Avoid interpreting a few accidental date-like strings as a
        # meaningful time dimension.
        if valid_count >= 3 and parsed.nunique(dropna=True) >= 2:
            date_columns.append((column, parsed))

    return date_columns


def _time_trend_insights(df):
    """Aggregate numeric metrics over time when a date column is usable."""
    insights = []
    numeric_columns = _numeric_columns(df)

    if not numeric_columns:
        return insights

    for date_column, parsed_dates in _find_date_columns(df):
        working = pd.DataFrame({
            "_date": parsed_dates,
        })

        # Use a small number of metrics to keep reports manageable.
        for value_column in numeric_columns[:5]:
            working["_value"] = df[value_column].to_numpy()

            valid = working.dropna(subset=["_date", "_value"])

            if len(valid) < 3:
                continue

            monthly = (
                valid.assign(period=valid["_date"].dt.to_period("M"))
                .groupby("period")["_value"]
                .sum()
                .sort_index()
            )

            if len(monthly) < 2:
                continue

            first_value = float(monthly.iloc[0])
            last_value = float(monthly.iloc[-1])
            difference = last_value - first_value

            if first_value != 0:
                percentage_change = (
                    difference / abs(first_value)
                ) * 100
            else:
                percentage_change = None

            direction = (
                "increased" if difference > 0
                else "decreased" if difference < 0
                else "remained unchanged"
            )

            insights.append({
                "type": "time_trend",
                "title": f"{value_column} over time",
                "date_column": str(date_column),
                "value_column": str(value_column),
                "evidence": {
                    "period": "month",
                    "period_count": int(len(monthly)),
                    "first_period": str(monthly.index[0]),
                    "first_value": first_value,
                    "last_period": str(monthly.index[-1]),
                    "last_value": last_value,
                    "absolute_change": difference,
                    "percentage_change": percentage_change,
                    "peak_period": str(monthly.idxmax()),
                    "peak_value": float(monthly.max()),
                    "monthly_values": [
                        {
                            "period": str(period),
                            "value": float(value),
                        }
                        for period, value in monthly.items()
],
                },
                "explanation": (
                    f"Monthly {value_column} {direction} from "
                    f"{first_value:.4g} in {monthly.index[0]} "
                    f"to {last_value:.4g} in {monthly.index[-1]}."
                ),
            })

    return insights


def generate_business_insights(df: pd.DataFrame) -> dict:
    """
    Generate deterministic, evidence-backed findings from a dataset.

    No LLM is used to calculate the reported values.
    """
    if len(df.columns) == 0:
        raise AnalyticsError(
            "Cannot generate insights for a dataset with no columns."
        )

    quality_report = profile_data_quality(df)

    insights = []

    # Data-quality findings.
    if quality_report["missing_cells"] > 0:
        insights.append({
            "type": "missing_data",
            "title": "Missing values detected",
            "evidence": {
                "missing_cells": quality_report["missing_cells"],
                "row_count": quality_report["row_count"],
                "column_count": quality_report["column_count"],
                "missing_by_column": {
                    column: count
                    for column, count
                    in quality_report["missing_by_column"].items()
                    if count > 0
                },
            },
            "explanation": (
                f"The dataset contains {quality_report['missing_cells']} "
                "missing cells. Review the affected columns before "
                "drawing conclusions from those fields."
            ),
        })

    if quality_report["duplicate_rows"] > 0:
        insights.append({
            "type": "duplicate_rows",
            "title": "Duplicate rows detected",
            "evidence": {
                "duplicate_rows": quality_report["duplicate_rows"],
                "row_count": quality_report["row_count"],
            },
            "explanation": (
                f"{quality_report['duplicate_rows']} rows duplicate "
                "earlier rows across all columns. Check whether these "
                "are repeated records or legitimate repeated events."
            ),
        })

    insights.extend(_numeric_insights(df))
    insights.extend(_category_insights(df))
    insights.extend(_time_trend_insights(df))

    return {
        "operation": "generate_business_insights",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "insight_count": len(insights),
        "insights": insights,
        "quality_report": quality_report,
    }

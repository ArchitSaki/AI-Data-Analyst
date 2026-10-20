import pandas as pd

class AnalyticsError(ValueError):
    """Raised when an analytical request cannot be completed."""


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise AnalyticsError(f"Column '{column}' does not exist.")


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    _require_column(df, column)

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise AnalyticsError(
            f"Column '{column}' must contain numeric data."
        )


def calculate_total(df: pd.DataFrame, column: str) -> dict:
    """Calculate the sum of a numeric column."""

    _require_numeric(df, column)

    return {
        "operation": "sum",
        "column": column,
        "value": float(df[column].sum()),
        "valid_values": int(df[column].count()),
    }


def calculate_average(df: pd.DataFrame, column: str) -> dict:
    """Calculate the mean of a numeric column."""

    _require_numeric(df, column)

    value = df[column].mean()

    if pd.isna(value):
        raise AnalyticsError(
            f"Column '{column}' has no valid numeric values."
        )

    return {
        "operation": "mean",
        "column": column,
        "value": float(value),
        "valid_values": int(df[column].count()),
    }


def aggregate_by_group(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    operation: str = "sum",
) -> dict:
    """Aggregate a numeric column by categories."""

    _require_column(df, group_column)
    _require_numeric(df, value_column)

    if operation not in {"sum", "mean", "count", "min", "max"}:
        raise AnalyticsError(
            "Unsupported operation. Use sum, mean, count, min, or max."
        )

    grouped = df.groupby(
        group_column,
        dropna=False,
    )[value_column].agg(operation)

    results = [
        {
            "group": None if pd.isna(group) else str(group),
            "value": float(value),
        }
        for group, value in grouped.items()
        if pd.notna(value)
    ]

    return {
        "operation": operation,
        "group_column": group_column,
        "value_column": value_column,
        "results": results,
    }


def get_top_records(
    df: pd.DataFrame,
    column: str,
    limit: int = 5,
) -> dict:
    """Return the highest-ranking records by a numeric column."""

    _require_numeric(df, column)

    if not 1 <= limit <= 100:
        raise AnalyticsError("Limit must be between 1 and 100.")

    records = (
        df.nlargest(limit, column)
        .astype(object)
        .where(pd.notna(df.nlargest(limit, column)), None)
        .to_dict(orient="records")
    )

    return {
        "operation": "top_records",
        "sort_column": column,
        "limit": limit,
        "results": records,
    }


def detect_anomalies(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> dict:
    """Detect statistical outliers using the IQR method."""

    _require_numeric(df, column)

    if not pd.api.types.is_number(multiplier):
        raise AnalyticsError("The IQR multiplier must be a positive number.")

    if not pd.notna(multiplier) or multiplier <= 0:
        raise AnalyticsError("The IQR multiplier must be a positive number.")

    if df.empty:
        raise AnalyticsError("Cannot detect anomalies in an empty dataset.")

    values = df[column]
    valid_values = values.dropna()

    if valid_values.empty:
        raise AnalyticsError(
            f"Column '{column}' has no valid numeric values."
        )

    q1 = float(valid_values.quantile(0.25))
    q3 = float(valid_values.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    mask = values.notna() & (
        (values < lower_bound) | (values > upper_bound)
    )

    anomalies = []

    for row_index, row in df.loc[mask].iterrows():
        record = {
            str(key): (
                None if pd.isna(value) else
                value.item() if hasattr(value, "item") else value
            )
            for key, value in row.items()
        }

        anomalies.append({
            "row_index": (
                int(row_index)
                if isinstance(row_index, int)
                else str(row_index)
            ),
            "value": float(row[column]),
            "type": (
                "high" if row[column] > upper_bound else "low"
            ),
            "record": record,
        })

    return {
        "operation": "detect_anomalies",
        "column": column,
        "method": "IQR",
        "multiplier": float(multiplier),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "valid_values": int(valid_values.count()),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "explanation": (
            f"Flagged {len(anomalies)} outlier(s) in '{column}' "
            f"using the IQR method. Values below {lower_bound:.4g} "
            f"or above {upper_bound:.4g} were flagged. "
            "Outliers are unusual values, not necessarily errors."
        ),
    }


def profile_data_quality(df: pd.DataFrame) -> dict:
    """Summarize dataset completeness, duplicates, and column statistics."""

    if df.empty and len(df.columns) == 0:
        raise AnalyticsError("Cannot profile a dataset with no columns.")

    missing_by_column = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    duplicate_rows = int(df.duplicated().sum())

    columns = []

    for column in df.columns:
        series = df[column]

        column_info = {
            "column": str(column),
            "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": (
                round(float(series.isna().mean() * 100), 2)
                if len(df) > 0 else 0.0
            ),
            "unique_values": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            valid = series.dropna()

            if not valid.empty:
                column_info["minimum"] = float(valid.min())
                column_info["maximum"] = float(valid.max())
                column_info["mean"] = float(valid.mean())
                column_info["median"] = float(valid.median())
            else:
                column_info["minimum"] = None
                column_info["maximum"] = None
                column_info["mean"] = None
                column_info["median"] = None

        columns.append(column_info)

    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())

    return {
        "operation": "profile_data_quality",
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "completeness_percentage": (
            round((1 - missing_cells / total_cells) * 100, 2)
            if total_cells else 100.0
        ),
        "missing_by_column": missing_by_column,
        "columns": columns,
    }

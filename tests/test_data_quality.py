
import pandas as pd
import pytest

from app.services.analytics import (
    AnalyticsError,
    profile_data_quality,
)


def test_profile_reports_missing_values_and_duplicates():
    df = pd.DataFrame({
        "product": ["A", "B", "B", None],
        "revenue": [10.0, 20.0, 20.0, None],
    })

    result = profile_data_quality(df)

    assert result["row_count"] == 4
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 1
    assert result["missing_cells"] == 2
    assert result["completeness_percentage"] == 75.0
    assert result["missing_by_column"]["product"] == 1
    assert result["missing_by_column"]["revenue"] == 1


def test_profile_reports_numeric_statistics():
    df = pd.DataFrame({
        "revenue": [10.0, 20.0, 30.0, None],
    })

    result = profile_data_quality(df)
    revenue = result["columns"][0]

    assert revenue["data_type"] == "float64"
    assert revenue["unique_values"] == 3
    assert revenue["minimum"] == 10.0
    assert revenue["maximum"] == 30.0
    assert revenue["mean"] == 20.0
    assert revenue["median"] == 20.0


def test_profile_handles_empty_rows_with_columns():
    df = pd.DataFrame({
        "revenue": pd.Series(dtype="float64"),
    })

    result = profile_data_quality(df)

    assert result["row_count"] == 0
    assert result["column_count"] == 1
    assert result["completeness_percentage"] == 100.0


def test_profile_rejects_dataset_without_columns():
    df = pd.DataFrame()

    with pytest.raises(AnalyticsError):
        profile_data_quality(df)

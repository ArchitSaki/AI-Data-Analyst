
import pandas as pd
import pytest

from app.services.analytics import AnalyticsError
from app.services.business_insights import generate_business_insights


def test_discovers_category_performance():
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C"],
        "revenue": [100, 50, 200, 150, 20, 10],
    })

    result = generate_business_insights(df)

    category_insights = [
        item for item in result["insights"]
        if item["type"] == "category_performance"
        and item["category_column"] == "category"
        and item["value_column"] == "revenue"
    ]

    assert category_insights

    evidence = category_insights[0]["evidence"]
    assert evidence["highest"] == {"group": "B", "value": 350.0}
    assert evidence["lowest"] == {"group": "C", "value": 30.0}


def test_reports_missing_and_duplicate_rows():
    df = pd.DataFrame({
        "category": ["A", "A", "B"],
        "revenue": [100.0, 100.0, None],
    })

    result = generate_business_insights(df)
    types = {item["type"] for item in result["insights"]}

    assert "missing_data" in types
    assert "duplicate_rows" in types
    assert result["quality_report"]["missing_cells"] == 1
    assert result["quality_report"]["duplicate_rows"] == 1


def test_reports_numeric_distribution():
    df = pd.DataFrame({
        "revenue": [10, 20, 30, 40, 50],
    })

    result = generate_business_insights(df)

    distribution = next(
        item for item in result["insights"]
        if item["type"] == "numeric_distribution"
    )

    assert distribution["evidence"]["mean"] == 30.0
    assert distribution["evidence"]["median"] == 30.0
    assert distribution["evidence"]["minimum"] == 10.0
    assert distribution["evidence"]["maximum"] == 50.0


def test_detects_monthly_trend():
    df = pd.DataFrame({
        "order_date": [
            "2026-01-05",
            "2026-01-20",
            "2026-02-05",
            "2026-02-20",
            "2026-03-05",
            "2026-03-20",
        ],
        "revenue": [10, 20, 30, 40, 50, 60],
    })

    result = generate_business_insights(df)

    trends = [
        item for item in result["insights"]
        if item["type"] == "time_trend"
        and item["value_column"] == "revenue"
    ]

    assert trends
    assert trends[0]["evidence"]["first_value"] == 30.0
    assert trends[0]["evidence"]["last_value"] == 110.0
    assert trends[0]["evidence"]["absolute_change"] == 80.0


def test_rejects_dataset_without_columns():
    with pytest.raises(AnalyticsError):
        generate_business_insights(pd.DataFrame())


def test_handles_empty_rows_with_columns():
    df = pd.DataFrame({
        "category": pd.Series(dtype="object"),
        "revenue": pd.Series(dtype="float64"),
    })

    result = generate_business_insights(df)

    assert result["row_count"] == 0
    assert result["insight_count"] == 0

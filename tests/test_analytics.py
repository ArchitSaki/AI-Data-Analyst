import pandas as pd
import pytest

from app.services.analytics import (
    AnalyticsError,
    aggregate_by_group,
    calculate_average,
    calculate_total,
    get_top_records,
)
from app.services.analytics import detect_anomalies



@pytest.fixture
def sales_data():
    return pd.DataFrame({
        "region": ["West", "East", "West", "North", "East"],
        "product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone"],
        "revenue": [50000, 30000, 15000, 60000, None],
        "units": [2, 3, 5, 2, 4],
    })


def test_calculate_total(sales_data):
    result = calculate_total(sales_data, "revenue")

    assert result["value"] == 155000.0
    assert result["valid_values"] == 4


def test_calculate_average(sales_data):
    result = calculate_average(sales_data, "revenue")

    assert result["value"] == 38750.0


def test_aggregate_by_region(sales_data):
    result = aggregate_by_group(
        sales_data,
        "region",
        "revenue",
        "sum",
    )

    values = {
        item["group"]: item["value"]
        for item in result["results"]
    }

    assert values["West"] == 65000.0
    assert values["East"] == 30000.0
    assert values["North"] == 60000.0


def test_top_records(sales_data):
    result = get_top_records(sales_data, "revenue", 2)

    assert len(result["results"]) == 2
    assert result["results"][0]["revenue"] == 60000.0


def test_missing_column(sales_data):
    with pytest.raises(AnalyticsError):
        calculate_total(sales_data, "profit")


def test_non_numeric_column(sales_data):
    with pytest.raises(AnalyticsError):
        calculate_total(sales_data, "product")


def test_invalid_aggregation(sales_data):
    with pytest.raises(AnalyticsError):
        aggregate_by_group(
            sales_data,
            "region",
            "revenue",
            "median",
        )


def test_detects_high_and_low_anomalies():
    df = pd.DataFrame({
        "value": [10, 11, 12, 13, 14, 100, -100]
    })

    result = detect_anomalies(df, "value")

    anomalies = result["anomalies"]
    flagged_values = {item["value"] for item in anomalies}

    assert flagged_values == {100.0, -100.0}
    assert result["column"] == "value"
    assert result["method"] == "IQR"


def test_detect_anomalies_ignores_missing_values():
    df = pd.DataFrame({
        "value": [10, 11, 12, 13, 14, 100, None]
    })

    result = detect_anomalies(df, "value")

    assert len(result["anomalies"]) == 1
    assert result["anomalies"][0]["value"] == 100.0
    assert result["valid_values"] == 6


def test_detect_anomalies_returns_empty_for_constant_column():
    df = pd.DataFrame({"value": [5, 5, 5, 5, 5]})

    result = detect_anomalies(df, "value")

    assert result["anomalies"] == []
    assert result["lower_bound"] == 5.0
    assert result["upper_bound"] == 5.0


def test_detect_anomalies_returns_empty_when_no_outliers():
    df = pd.DataFrame({
        "value": [10, 11, 12, 13, 14, 15]
    })

    result = detect_anomalies(df, "value")

    assert result["anomalies"] == []


def test_detect_anomalies_rejects_missing_column(sales_data):
    with pytest.raises(AnalyticsError, match="does not exist"):
        detect_anomalies(sales_data, "profit")


def test_detect_anomalies_rejects_non_numeric_column(sales_data):
    with pytest.raises(AnalyticsError, match="numeric"):
        detect_anomalies(sales_data, "product")


def test_detect_anomalies_rejects_invalid_multiplier(sales_data):
    with pytest.raises(AnalyticsError, match="multiplier"):
        detect_anomalies(sales_data, "units", multiplier=0)


import pandas as pd
import pytest

from app.services import ai_analyst


def test_analyze_question_rejects_empty_question():
    df = pd.DataFrame({"sales": [10, 20]})

    with pytest.raises(ValueError, match="analysis question"):
        ai_analyst.analyze_question(df, "   ")


def test_analyze_question_rejects_question_over_2000_characters():
    df = pd.DataFrame({"sales": [10, 20]})

    with pytest.raises(ValueError, match="2,000 characters"):
        ai_analyst.analyze_question(df, "x" * 2001)


def test_anomaly_response_has_expected_contract(monkeypatch):
    df = pd.DataFrame({"sales": [10, 11, 12, 13, 1000]})

    monkeypatch.setattr(
        ai_analyst,
        "detect_anomalies",
        lambda df, column: {
            "anomaly_count": 1,
            "anomalies": [{"sales": 1000}],
            "lower_bound": 5.0,
            "upper_bound": 20.0,
            "valid_values": 5,
            "method": "IQR",
        },
    )

    result = ai_analyst.analyze_question(df, "detect anomalies in sales")

    expected_keys = {
        "conversation_id",
        "question",
        "analysis_type",
        "answer",
        "column",
        "method",
        "anomaly_count",
        "anomalies",
        "lower_bound",
        "upper_bound",
        "valid_values",
        "chart",
    }

    assert expected_keys.issubset(result.keys())
    assert result["analysis_type"] == "anomaly_detection"
    assert result["column"] == "sales"
    assert result["anomaly_count"] == 1


def test_data_quality_response_has_expected_contract(monkeypatch):
    df = pd.DataFrame({"sales": [10, None, 20]})

    monkeypatch.setattr(
        ai_analyst,
        "profile_data_quality",
        lambda df: {
            "row_count": 3,
            "column_count": 1,
            "missing_cells": 1,
            "missing_by_column": {"sales": 1},
            "duplicate_rows": 0,
            "completeness_percentage": 66.67,
        },
    )

    result = ai_analyst.analyze_question(df, "run a data quality check")

    assert result["analysis_type"] == "data_quality"
    assert result["question"] == "run a data quality check"
    assert isinstance(result["quality_report"], dict)
    assert "answer" in result
    assert "conversation_id" in result


def test_business_insights_response_has_expected_contract(monkeypatch):
    df = pd.DataFrame({"sales": [10, 20, 30]})

    monkeypatch.setattr(
        ai_analyst,
        "generate_business_insights",
        lambda df: {
            "row_count": 3,
            "column_count": 1,
            "insight_count": 1,
            "insights": [
                {
                    "type": "summary",
                    "title": "Sales summary",
                    "explanation": "Sales values are available.",
                    "evidence": {"count": 3},
                }
            ],
            "quality_report": {},
        },
    )

    result = ai_analyst.analyze_question(df, "generate business insights")

    assert result["analysis_type"] == "business_insights"
    assert result["row_count"] == 3
    assert result["column_count"] == 1
    assert result["insight_count"] == 1
    assert len(result["insights"]) == 1
    assert "answer" in result
    assert "conversation_id" in result

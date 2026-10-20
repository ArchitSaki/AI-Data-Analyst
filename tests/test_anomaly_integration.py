
import pandas as pd
import pytest

from app.services import ai_analyst
from app.services.analytics import AnalyticsError


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "product": ["A", "B", "C", "D", "E", "F", "G"],
        "revenue": [10, 11, 12, 13, 14, 15, 100],
        "units": [2, 3, 4, 5, 6, 7, 8],
    })


def test_anomaly_request_uses_detector_without_running_graph(
    sample_data, monkeypatch
):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Normal analysis graph should not run for anomaly requests.")

    monkeypatch.setattr(
        ai_analyst,
        "get_schema",
        lambda df: [
            {"column": str(col), "data_type": str(df[col].dtype)}
            for col in df.columns
        ],
    )

    monkeypatch.setattr(
        ai_analyst,
        "get_analysis_graph",
        fail_if_called,
        raising=False,
    )

    result = ai_analyst.analyze_question(
        sample_data,
        "Find anomalies in revenue",
        conversation_id="anomaly-test-1",
    )

    assert result["analysis_type"] == "anomaly_detection"
    assert result["column"] == "revenue"
    assert result["anomaly_count"] >= 1
    assert result["method"] == "IQR"


def test_anomaly_request_requires_a_column_when_ambiguous(
    sample_data,
):
    result = ai_analyst.analyze_question(
        sample_data,
        "Find anomalies",
        conversation_id="anomaly-test-2",
    )

    assert result["analysis_type"] == "anomaly_detection"
    assert result["anomaly_count"] == 0
    assert "column" in result["answer"].lower()


def test_normal_question_still_uses_analysis_graph(
    sample_data, monkeypatch
):
    class FakeGraph:
        def invoke(self, state, config):
            return {
                "execution_result": {
                    "sql": "SELECT SUM(revenue) AS total_revenue FROM uploaded_data",
                    "columns": ["total_revenue"],
                    "rows": [[175]],
                    "row_count": 1,
                    "truncated": False,
                },
                "answer": "Total revenue is 175.",
            }

    from app.services.analysis_graph import get_analysis_graph

    monkeypatch.setattr(
        "app.services.analysis_graph.get_analysis_graph",
        lambda: FakeGraph(),
    )

    result = ai_analyst.analyze_question(
        sample_data,
        "What is the total revenue?",
        conversation_id="normal-test-1",
    )

    assert result["answer"] == "Total revenue is 175."
    assert result["sql"].startswith("SELECT SUM")
    assert "analysis_type" in result
    assert result["analysis_type"] == "sql"

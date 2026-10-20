
import pandas as pd

from app.services import ai_analyst


def test_data_quality_request_uses_profiler_without_graph(
    monkeypatch,
):
    df = pd.DataFrame({
        "product": ["A", "B", None],
        "revenue": [10.0, 20.0, 20.0],
    })

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Data-quality requests should not invoke the SQL graph."
        )

    monkeypatch.setattr(
        ai_analyst,
        "get_schema",
        lambda df: [],
    )
    monkeypatch.setattr(
        ai_analyst,
        "get_analysis_graph",
        fail_if_called,
        raising=False,
    )

    result = ai_analyst.analyze_question(
        df,
        "Check the data quality",
        conversation_id="quality-test-1",
    )

    assert result["analysis_type"] == "data_quality"
    assert result["quality_report"]["missing_cells"] == 1
    assert result["quality_report"]["duplicate_rows"] == 0
    assert "completeness" in result["answer"].lower()


def test_data_quality_request_detects_duplicate_rows():
    df = pd.DataFrame({
        "product": ["A", "A", "B"],
        "revenue": [10, 10, 20],
    })

    result = ai_analyst.analyze_question(
        df,
        "Find duplicate rows",
        conversation_id="quality-test-2",
    )

    assert result["analysis_type"] == "data_quality"
    assert result["quality_report"]["duplicate_rows"] == 1
    assert "duplicate row" in result["answer"].lower()


def test_normal_question_still_uses_sql_graph(monkeypatch):
    class FakeGraph:
        def invoke(self, state, config):
            return {
                "execution_result": {
                    "sql": "SELECT COUNT(*) AS total FROM uploaded_data",
                    "columns": ["total"],
                    "rows": [[3]],
                    "row_count": 1,
                    "truncated": False,
                },
                "answer": "There are 3 rows.",
            }

    from app.services.analysis_graph import get_analysis_graph

    monkeypatch.setattr(
        "app.services.analysis_graph.get_analysis_graph",
        lambda: FakeGraph(),
    )

    result = ai_analyst.analyze_question(
        pd.DataFrame({"revenue": [10, 20, 30]}),
        "How many rows are there?",
        conversation_id="quality-test-3",
    )

    assert result["analysis_type"] == "sql"
    assert result["answer"] == "There are 3 rows."

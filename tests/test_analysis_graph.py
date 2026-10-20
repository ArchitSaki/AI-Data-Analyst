
import pandas as pd

from app.services import analysis_graph
from app.services.dataset_store import save_dataset

def make_state():
    df = pd.DataFrame({
        "product": ["A", "B", "A"],
        "sales": [10, 20, 15],
    })

    dataset_id = "test-dataset"
    save_dataset(dataset_id, df)

    return {
        "dataset_id": dataset_id,
        "question": "What is the total sales?",
        "sql": "",
        "execution_result": None,
        "answer": "",
        "last_error": None,
        "attempts": 0,
        "max_attempts": 2,
    }

def test_successful_analysis(monkeypatch):
    expected_result = {
        "sql": "SELECT SUM(sales) FROM uploaded_data",
        "columns": ["total_sales"],
        "rows": [{"total_sales": 45}],
        "row_count": 1,
        "truncated": False,
    }

    monkeypatch.setattr(
        analysis_graph,
        "generate_sql",
        lambda **kwargs: "SELECT SUM(sales) FROM uploaded_data",
    )
    monkeypatch.setattr(
        analysis_graph,
        "execute_sql",
        lambda df, sql: expected_result,
    )
    monkeypatch.setattr(
        analysis_graph,
        "explain_result",
        lambda question, result: "Total sales are 45.",
    )

    result = analysis_graph.build_analysis_graph().invoke(
    make_state(),
    config={
        "configurable": {
            "thread_id": "test-thread",
        }
    },
)

    assert result["attempts"] == 1
    assert result["execution_result"] == expected_result
    assert result["answer"] == "Total sales are 45."


def test_retries_after_execution_failure(monkeypatch):
    attempts = []
    execution_calls = []

    def fake_generate_sql(**kwargs):
        attempts.append(kwargs["previous_error"])
        return "SELECT SUM(sales) FROM uploaded_data"

    def fake_execute_sql(df, sql):
        execution_calls.append(sql)

        if len(execution_calls) == 1:
            raise ValueError("Simulated SQL validation error")

        return {
            "sql": sql,
            "columns": ["total_sales"],
            "rows": [{"total_sales": 45}],
            "row_count": 1,
            "truncated": False,
        }

    monkeypatch.setattr(
        analysis_graph, "generate_sql", fake_generate_sql
    )
    monkeypatch.setattr(
        analysis_graph, "execute_sql", fake_execute_sql
    )
    monkeypatch.setattr(
        analysis_graph,
        "explain_result",
        lambda question, result: "Total sales are 45.",
    )

    result = analysis_graph.build_analysis_graph().invoke(
    make_state(),
    config={
        "configurable": {
            "thread_id": "test-thread",
        }
    },
)

    assert len(execution_calls) == 2
    assert attempts[0] is None
    assert "Simulated SQL validation error" in attempts[1]
    assert result["attempts"] == 2
    assert result["answer"] == "Total sales are 45."


def test_stops_after_retry_limit(monkeypatch):
    execution_calls = []

    monkeypatch.setattr(
        analysis_graph,
        "generate_sql",
        lambda **kwargs: "INVALID SQL",
    )

    def fake_execute_sql(df, sql):
        execution_calls.append(sql)
        raise ValueError("Simulated SQL validation error")

    monkeypatch.setattr(
        analysis_graph, "execute_sql", fake_execute_sql
    )

    result = analysis_graph.build_analysis_graph().invoke(
    make_state(),
    config={
        "configurable": {
            "thread_id": "test-thread",
        }
    },
)
    assert len(execution_calls) == 2
    assert result["attempts"] == 2
    assert "couldn't safely execute" in result["answer"]
    assert result["execution_result"] is None

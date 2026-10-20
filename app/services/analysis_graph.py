
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.services.ai_analyst import (
    generate_sql,
    generate_sql_multi,
    explain_result,
)
from app.services.sql_engine import execute_sql, execute_sql_multi
from app.services.dataset_store import get_dataset, get_datasets
from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
import operator
from typing import Annotated

class AnalysisState(TypedDict):
    dataset_id: str
    question: str
    sql: str
    execution_result: dict[str, Any] | None
    answer: str
    last_error: str | None
    attempts: int
    max_attempts: int
    multi_file: bool
    history: Annotated[list[dict[str, str]], operator.add]




def generate_query(state: AnalysisState) -> dict:
    """Generate SQL using the current session's dataset schema."""

    history = state.get("history", [])[-5:]
    contextual_question = state["question"]

    if history:
        history_text = "\n".join(
            f"Previous question: {item['question']}\n"
            f"Previous answer: {item['answer']}"
            for item in history
        )

        contextual_question = (
            "Use the conversation history below only to resolve references "
            "in the current question. Generate SQL using the current dataset "
            "schema and data, not by assuming previous results are still valid.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Current question: {state['question']}"
        )

    if state.get("multi_file", False):
        datasets = get_datasets(state["dataset_id"])
        sql = generate_sql_multi(
            datasets=datasets,
            question=contextual_question,
            previous_error=state["last_error"],
        )
    else:
        df = get_dataset(state["dataset_id"])
        sql = generate_sql(
            df=df,
            question=contextual_question,
            previous_error=state["last_error"],
        )

    return {
        "sql": sql,
        "attempts": state["attempts"] + 1,
        "last_error": None,
    }




def execute_query(state: AnalysisState) -> dict:
    """Execute SQL against the uploaded single or multiple datasets."""

    try:
        if state.get("multi_file", False):
            datasets = get_datasets(state["dataset_id"])
            result = execute_sql_multi(datasets, state["sql"])
        else:
            df = get_dataset(state["dataset_id"])
            result = execute_sql(df, state["sql"])

        return {
            "execution_result": result,
            "last_error": None,
        }

    except Exception as exc:
        return {
            "execution_result": None,
            "last_error": str(exc)[:500],
        }

def route_after_execution(state: AnalysisState) -> str:
    """Retry once after an execution error, otherwise stop or explain."""

    if state["execution_result"] is not None:
        return "explain"

    if state["attempts"] < state["max_attempts"]:
        return "retry"

    return "failed"



def explain_successful_result(state: AnalysisState) -> dict:
    """Explain the result and append the completed turn to history."""

    result = state["execution_result"]

    if result is None:
        raise RuntimeError("Cannot explain a missing execution result.")

    answer = explain_result(state["question"], result)

    return {
        "answer": answer,
        "history": [
            {
                "question": state["question"],
                "answer": answer,
            }
        ],
    }

def finish_failed_analysis(state: AnalysisState) -> dict:
    """Produce a controlled failure message after retries are exhausted."""

    return {
        "answer": (
            "I couldn't safely execute a valid query for this question. "
            "Try rephrasing it or checking the dataset columns."
        )
    }


def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("generate_query", generate_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("explain_result", explain_successful_result)
    graph.add_node("failed", finish_failed_analysis)

    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "execute_query")

    graph.add_conditional_edges(
        "execute_query",
        route_after_execution,
        {
            "retry": "generate_query",
            "explain": "explain_result",
            "failed": "failed",
        },
    )

    graph.add_edge("explain_result", END)
    graph.add_edge("failed", END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_analysis_graph():
    """Reuse one compiled graph and its in-memory checkpointer."""
    return build_analysis_graph()

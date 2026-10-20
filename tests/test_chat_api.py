
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chat as chat_module


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_module.router)
    return TestClient(app)


def test_ask_accepts_two_csv_files(client, monkeypatch):
    captured = {}

    def fake_analyze_question(df, question, conversation_id=None, datasets=None):
        captured["question"] = question
        captured["datasets"] = datasets

        return {
            "conversation_id": conversation_id or "test-conversation",
            "question": question,
            "answer": "Joined successfully",
            "sql": "SELECT * FROM customers",
            "columns": ["customer_id"],
            "results": [{"customer_id": "C001"}],
        }

    monkeypatch.setattr(
        chat_module,
        "analyze_question",
        fake_analyze_question,
    )

    response = client.post(
        "/api/ask",
        data={"question": "Join customers and orders"},
        files=[
            ("files", ("customers.csv", b"customer_id,name\nC001,Asha\n", "text/csv")),
            ("files", ("orders.csv", b"order_id,customer_id\nO001,C001\n", "text/csv")),
        ],
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Joined successfully"
    assert set(captured["datasets"]) == {"customers.csv", "orders.csv"}
    assert captured["question"] == "Join customers and orders"


def test_ask_rejects_missing_files(client):
    response = client.post(
        "/api/ask",
        data={"question": "Show all rows"},
    )

    assert response.status_code == 400
    assert "at least one CSV" in response.json()["detail"]


def test_ask_rejects_non_csv_file(client):
    response = client.post(
        "/api/ask",
        data={"question": "Analyze this file"},
        files={
            "file": ("notes.txt", b"not a CSV", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert "must be CSV" in response.json()["detail"]


def test_ask_rejects_duplicate_filenames(client):
    response = client.post(
        "/api/ask",
        data={"question": "Join these datasets"},
        files=[
            ("files", ("customers.csv", b"id\n1\n", "text/csv")),
            ("files", ("customers.csv", b"id\n2\n", "text/csv")),
        ],
    )

    assert response.status_code == 400
    assert "Duplicate filename" in response.json()["detail"]


def test_ask_preserves_single_file_compatibility(client, monkeypatch):
    captured = {}

    def fake_analyze_question(df, question, conversation_id=None, datasets=None):
        captured["df"] = df
        captured["datasets"] = datasets

        return {
            "conversation_id": conversation_id or "test-conversation",
            "answer": "Single dataset works",
        }

    monkeypatch.setattr(
        chat_module,
        "analyze_question",
        fake_analyze_question,
    )

    response = client.post(
        "/api/ask",
        data={"question": "Show all rows"},
        files={
            "file": ("customers.csv", b"id,name\n1,Asha\n", "text/csv"),
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Single dataset works"
    assert isinstance(captured["df"], pd.DataFrame)
    assert captured["datasets"] is None

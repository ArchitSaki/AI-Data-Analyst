
import pandas as pd
import pytest

from app.services.sql_engine import (
    SQLValidationError,
    execute_sql,
    execute_sql_multi,
    normalize_table_name,
    validate_sql,
)


@pytest.fixture
def sample_datasets():
    customers = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "customer_name": ["Asha", "Ravi", "Mira"],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": ["O001", "O002", "O003"],
            "customer_id": ["C001", "C001", "C002"],
            "quantity": [2, 3, 4],
            "unit_price": [100.0, 50.0, 25.0],
        }
    )

    return {
        "customers.csv": customers,
        "orders.csv": orders,
    }


def test_multi_file_join_calculates_revenue(sample_datasets):
    sql = """
        SELECT
            c.customer_name,
            SUM(o.quantity * o.unit_price) AS total_revenue
        FROM customers AS c
        JOIN orders AS o
            ON c.customer_id = o.customer_id
        GROUP BY c.customer_name
        ORDER BY c.customer_name
    """

    result = execute_sql_multi(sample_datasets, sql)

    assert result["columns"] == [
        "customer_name",
        "total_revenue",
    ]
    assert result["rows"] == [
        {"customer_name": "Asha", "total_revenue": 350.0},
        {"customer_name": "Ravi", "total_revenue": 100.0},
    ]
    assert result["row_count"] == 2


def test_multi_file_query_rejects_unuploaded_table(sample_datasets):
    sql = "SELECT * FROM payments"

    with pytest.raises(SQLValidationError, match="not permitted"):
        execute_sql_multi(sample_datasets, sql)


def test_single_dataset_query_still_works():
    df = pd.DataFrame({"sales": [10, 20, 15]})

    result = execute_sql(
        df,
        "SELECT SUM(sales) AS total_sales FROM uploaded_data",
    )

    assert result["rows"] == [{"total_sales": 45}]
    assert result["columns"] == ["total_sales"]


def test_rejects_multiple_sql_statements():
    sql = "SELECT 1; SELECT 2"

    with pytest.raises(SQLValidationError, match="one SQL statement"):
        validate_sql(sql)


def test_rejects_non_select_statement():
    with pytest.raises(SQLValidationError, match="read-only SELECT"):
        validate_sql("DELETE FROM uploaded_data")


def test_rejects_blocked_function():
    with pytest.raises(SQLValidationError, match="not permitted"):
        validate_sql("SELECT shell('echo unsafe')")


def test_normalizes_filename_to_table_name():
    assert normalize_table_name("Customer Data.csv") == "customer_data"
    assert normalize_table_name("2026-orders.csv") == "table_2026_orders"


def test_rejects_duplicate_normalized_table_names():
    datasets = {
        "customer-data.csv": pd.DataFrame({"id": [1]}),
        "customer data.csv": pd.DataFrame({"id": [2]}),
    }

    with pytest.raises(ValueError, match="same SQL table name"):
        execute_sql_multi(datasets, "SELECT * FROM customer_data")


def test_rejects_empty_dataset_collection():
    with pytest.raises(ValueError, match="At least one dataset"):
        execute_sql_multi({}, "SELECT 1")

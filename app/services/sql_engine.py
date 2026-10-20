import json
import re

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


TABLE_NAME = "uploaded_data"
MAX_RESULT_ROWS = 1000

# These functions can access external resources or execute nested queries.
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "parquet_scan",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "glob",
    "query",
    "query_table",
    "sqlite_scan",
    "postgres_scan",
    "shell",
    "system",
}


class SQLValidationError(ValueError):
    """Raised when generated SQL violates our query rules."""


def clean_generated_sql(sql: str) -> str:
    """Remove common Markdown code fences from LLM output."""

    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";")


def validate_sql(sql: str) -> str:
    """Allow one read-only query against the uploaded dataset."""

    cleaned_sql = clean_generated_sql(sql)

    if not cleaned_sql:
        raise SQLValidationError("The generated SQL query is empty.")

    try:
        statements = sqlglot.parse(cleaned_sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError("The generated SQL is invalid.") from exc

    statements = [statement for statement in statements if statement]

    if len(statements) != 1:
        raise SQLValidationError("Only one SQL statement is allowed.")

    tree = statements[0]

    # Accept SELECT, UNION, and other read-only query expressions.
    if not isinstance(tree, exp.Query):
        raise SQLValidationError(
            "Only read-only SELECT queries are permitted."
        )

    # Allow the uploaded table and names introduced by CTEs.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
    }
    allowed_tables = {TABLE_NAME.lower()} | cte_names

    for table in tree.find_all(exp.Table):
        table_name = table.name.lower()

        # Reject database.schema.table and other qualified references.
        if table.db or table.catalog:
            raise SQLValidationError(
                "Qualified table references are not permitted."
            )

        if table_name not in allowed_tables:
            raise SQLValidationError(
                f"Access to table '{table.name}' is not permitted. "
                f"Use '{TABLE_NAME}'."
            )

    for function in tree.find_all(exp.Anonymous):
        function_name = function.name.lower()

        if function_name in BLOCKED_FUNCTIONS:
            raise SQLValidationError(
                f"The SQL function '{function_name}' is not permitted."
            )

    return cleaned_sql


def execute_sql(df: pd.DataFrame, sql: str) -> dict:
    """Execute validated SQL against the uploaded DataFrame."""

    validated_sql = validate_sql(sql)

    # Disable external access and limit the resources available to DuckDB.
    connection = duckdb.connect(
        database=":memory:",
        config={
            "enable_external_access": "false",
            "threads": "2",
            "memory_limit": "512MB",
        },
    )

    try:
        connection.register(TABLE_NAME, df)

        # Limit returned rows even when the model forgets to add LIMIT.
        bounded_sql = (
            f"SELECT * FROM ({validated_sql}) AS ai_result "
            f"LIMIT {MAX_RESULT_ROWS}"
        )

        result_df = connection.execute(bounded_sql).fetchdf()

        # JSON conversion handles missing values and common date types.
        rows = json.loads(
            result_df.to_json(orient="records", date_format="iso")
        )

        return {
            "sql": validated_sql,
            "columns": [str(column) for column in result_df.columns],
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(rows) == MAX_RESULT_ROWS,
        }

    finally:
        connection.close()

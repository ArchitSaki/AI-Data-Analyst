
import json
import re

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


TABLE_NAME = "uploaded_data"
MAX_RESULT_ROWS = 1000

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


def normalize_table_name(file_name: str) -> str:
    """Convert an uploaded filename into a safe SQL table identifier."""
    name = re.sub(r"\.csv$", "", file_name.strip(), flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()

    if not name:
        raise ValueError(
            f"Cannot derive a table name from filename '{file_name}'."
        )

    if name[0].isdigit():
        name = f"table_{name}"

    return name


def validate_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
) -> str:
    """Validate one read-only query against permitted tables."""
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

    if not isinstance(tree, exp.Query):
        raise SQLValidationError(
            "Only read-only SELECT queries are permitted."
        )

    permitted_tables = {
        name.lower() for name in (allowed_tables or {TABLE_NAME})
    }

    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
    }
    permitted_tables.update(cte_names)

    for table in tree.find_all(exp.Table):
        table_name = table.name.lower()

        if table.db or table.catalog:
            raise SQLValidationError(
                "Qualified table references are not permitted."
            )

        if table_name not in permitted_tables:
            allowed = ", ".join(sorted(allowed_tables or {TABLE_NAME}))
            raise SQLValidationError(
                f"Access to table '{table.name}' is not permitted. "
                f"Available tables: {allowed}."
            )

    for function in tree.find_all(exp.Anonymous):
        function_name = function.name.lower()

        if function_name in BLOCKED_FUNCTIONS:
            raise SQLValidationError(
                f"The SQL function '{function_name}' is not permitted."
            )

    return cleaned_sql


def _execute(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    allowed_tables: set[str],
) -> dict:
    """Validate and execute SQL with a bounded result set."""
    validated_sql = validate_sql(sql, allowed_tables)

    bounded_sql = (
        f"SELECT * FROM ({validated_sql}) AS ai_result "
        f"LIMIT {MAX_RESULT_ROWS}"
    )

    result_df = connection.execute(bounded_sql).fetchdf()

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


def _create_connection() -> duckdb.DuckDBPyConnection:
    """Create an isolated DuckDB connection with external access disabled."""
    return duckdb.connect(
        database=":memory:",
        config={
            "enable_external_access": "false",
            "threads": "2",
            "memory_limit": "512MB",
        },
    )


def execute_sql(df: pd.DataFrame, sql: str) -> dict:
    """Execute validated SQL against one uploaded DataFrame."""
    connection = _create_connection()

    try:
        connection.register(TABLE_NAME, df)
        return _execute(connection, sql, {TABLE_NAME})
    finally:
        connection.close()


def execute_sql_multi(
    datasets: dict[str, pd.DataFrame],
    sql: str,
) -> dict:
    """Execute SQL across multiple uploaded DataFrames."""
    if not datasets:
        raise ValueError("At least one dataset is required.")

    table_mapping: dict[str, pd.DataFrame] = {}

    for file_name, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Dataset '{file_name}' is not a valid DataFrame."
            )

        table_name = normalize_table_name(file_name)

        if table_name in table_mapping:
            raise ValueError(
                "Multiple filenames resolve to the same SQL table name "
                f"'{table_name}'. Rename the files to make them unique."
            )

        table_mapping[table_name] = df

    connection = _create_connection()

    try:
        for table_name, df in table_mapping.items():
            connection.register(table_name, df)

        return _execute(connection, sql, set(table_mapping))
    finally:
        connection.close()

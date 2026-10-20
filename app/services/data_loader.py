from io import BytesIO

import pandas as pd


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class CSVValidationError(ValueError):
    """Raised when an uploaded CSV is invalid."""


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load and validate CSV bytes into a pandas DataFrame."""

    if not file_bytes:
        raise CSVValidationError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise CSVValidationError("The file exceeds the 20 MB limit.")

    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CSVValidationError(
            "Unable to parse the CSV. Check its format and encoding."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError(
            "The uploaded file does not contain CSV data."
        ) from exc

    if df.empty and len(df.columns) == 0:
        raise CSVValidationError("The CSV has no columns.")

    if len(df.columns) == 0:
        raise CSVValidationError("The CSV has no columns.")

    if len(df.columns) > 200:
        raise CSVValidationError("The CSV exceeds the 200-column limit.")

    if not df.columns.is_unique:
        raise CSVValidationError(
            "Duplicate column names detected. Rename them before uploading."
        )

    if any(not isinstance(column, str) or not column.strip()
           for column in df.columns):
        raise CSVValidationError("Column names must not be empty.")

    if df.shape[0] > 500_000:
        raise CSVValidationError("The CSV exceeds the 500,000-row limit.")

    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a JSON-compatible overview of a DataFrame."""

    column_profiles = []

    for column in df.columns:
        series = df[column]

        column_profiles.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_values": int(series.nunique()),
        })

    numeric_summary = (
        df.describe()
        .replace({float("inf"): None, float("-inf"): None})
        .round(2)
        .to_dict()
    )

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": column_profiles,
        "numeric_summary": numeric_summary,
        "preview": df.head(10).astype(object).where(
            pd.notna(df.head(10)), None
        ).to_dict(orient="records"),
    }

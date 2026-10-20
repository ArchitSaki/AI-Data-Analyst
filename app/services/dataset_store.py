
import pandas as pd


_DATASETS: dict[str, pd.DataFrame] = {}
_BUSINESS_INSIGHTS: dict[str, dict] = {}


def save_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Keep a dataset available without putting it in graph state."""
    _DATASETS[dataset_id] = df


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve a dataset by its ID."""
    if dataset_id not in _DATASETS:
        raise ValueError(
            "Dataset not found. Please upload the CSV again."
        )

    return _DATASETS[dataset_id]


def save_business_insights(
    dataset_id: str,
    report: dict,
) -> None:
    """Cache generated business insights for a dataset."""
    _BUSINESS_INSIGHTS[dataset_id] = report


def get_business_insights(dataset_id: str) -> dict | None:
    """Retrieve cached business insights, if available."""
    return _BUSINESS_INSIGHTS.get(dataset_id)

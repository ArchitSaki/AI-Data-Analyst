
import pandas as pd


DatasetCollection = dict[str, pd.DataFrame]
StoredDataset = pd.DataFrame | DatasetCollection

_DATASETS: dict[str, StoredDataset] = {}
_BUSINESS_INSIGHTS: dict[str, dict] = {}


def save_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Store a single DataFrame for backward compatibility."""
    _DATASETS[dataset_id] = df


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve a single DataFrame by its ID."""
    if dataset_id not in _DATASETS:
        raise ValueError(
            "Dataset not found. Please upload the CSV again."
        )

    dataset = _DATASETS[dataset_id]

    if not isinstance(dataset, pd.DataFrame):
        raise ValueError(
            "This session contains multiple datasets. "
            "Use get_datasets() to retrieve them."
        )

    return dataset


def save_datasets(
    dataset_id: str,
    datasets: DatasetCollection,
) -> None:
    """Store multiple named DataFrames for cross-file analysis."""
    if not datasets:
        raise ValueError("At least one dataset is required.")

    if not all(
        isinstance(name, str)
        and name.strip()
        and isinstance(df, pd.DataFrame)
        for name, df in datasets.items()
    ):
        raise ValueError(
            "Datasets must have non-empty names and DataFrame values."
        )

    _DATASETS[dataset_id] = dict(datasets)


def get_datasets(dataset_id: str) -> DatasetCollection:
    """Retrieve all DataFrames, supporting older single-file sessions."""
    if dataset_id not in _DATASETS:
        raise ValueError(
            "Dataset not found. Please upload the CSV again."
        )

    dataset = _DATASETS[dataset_id]

    if isinstance(dataset, pd.DataFrame):
        return {"uploaded_data": dataset}

    return dataset


def save_business_insights(
    dataset_id: str,
    report: dict,
) -> None:
    """Cache generated business insights for a dataset."""
    _BUSINESS_INSIGHTS[dataset_id] = report


def get_business_insights(dataset_id: str) -> dict | None:
    """Retrieve cached business insights, if available."""
    return _BUSINESS_INSIGHTS.get(dataset_id)


import pandas as pd

from app.services.charting import create_chart


def test_category_comparison_creates_bar_chart():
    columns = ["region", "revenue"]
    results = [
        ["West", 1200],
        ["East", 900],
        ["North", 1500],
    ]

    fig = create_chart(
        "Compare revenue by region",
        columns,
        results,
    )

    assert fig is not None
    assert fig.data[0].type == "bar"


def test_monthly_trend_creates_line_chart():
    columns = ["month", "sales"]
    results = [
        ["2026-01-01", 100],
        ["2026-02-01", 150],
        ["2026-03-01", 125],
    ]

    fig = create_chart(
        "Show monthly sales trends",
        columns,
        results,
    )

    assert fig is not None
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines+markers"


def test_insufficient_results_returns_none():
    fig = create_chart(
        "Show sales",
        ["sales"],
        [[100], [200]],
    )

    assert fig is None

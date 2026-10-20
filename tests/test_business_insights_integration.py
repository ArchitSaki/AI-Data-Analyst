
import pandas as pd

from app.services.ai_analyst import analyze_question


def test_business_insights_request_returns_structured_findings():
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C"],
        "revenue": [100, 50, 200, 150, 20, 10],
    })

    result = analyze_question(
        df,
        "Generate business insights",
    )

    assert result["analysis_type"] == "business_insights"
    assert result["insight_count"] > 0
    assert isinstance(result["insights"], list)

    category_insights = [
        item for item in result["insights"]
        if item["type"] == "category_performance"
        and item["category_column"] == "category"
        and item["value_column"] == "revenue"
    ]

    assert category_insights
    assert category_insights[0]["evidence"]["highest"] == {
        "group": "B",
        "value": 350.0,
    }


def test_business_insights_do_not_require_groq():
    # The insights route should use deterministic calculations,
    # not SQL generation or an LLM request.
    df = pd.DataFrame({
        "category": ["A", "B", "C"],
        "revenue": [10, 20, 30],
    })

    result = analyze_question(
        df,
        "Show business insights",
    )

    assert result["analysis_type"] == "business_insights"
    assert result["row_count"] == 3

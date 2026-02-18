"""Tests for fundamentals scoring."""

import pandas as pd
import pytest

from src.scoring.fundamentals import compute_fundamentals_metrics, score_fundamentals


@pytest.fixture
def sample_financials() -> pd.DataFrame:
    """Minimal financial data for testing."""
    return pd.DataFrame({
        "NetSales": [1000, 2000, 1500],
        "OperatingProfit": [100, 200, 150],
        "ForecastOperatingProfit": [120, 220, 160],
        "NextYearForecastNetSales": [1100, 2200, 1600],
        "NextYearForecastOperatingProfit": [140, 250, 180],
        "Profit": [80, 160, 120],
        "Equity": [500, 1000, 750],
        "EquityToAssetRatio": [0.4, 0.5, 0.45],
    })


def test_compute_fundamentals_metrics(sample_financials: pd.DataFrame) -> None:
    """Metric computation produces expected columns."""
    result = compute_fundamentals_metrics(sample_financials)
    expected_cols = [
        "f2_f0_sales_growth", "f2_f0_ebit_growth",
        "f1_f0_op_growth", "f2_f1_op_growth",
        "opm", "roe", "equity_ratio",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_score_fundamentals_returns_series(sample_financials: pd.DataFrame) -> None:
    """Scoring returns a Series with same index as input."""
    metrics = compute_fundamentals_metrics(sample_financials)
    scores = score_fundamentals(metrics)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(sample_financials)

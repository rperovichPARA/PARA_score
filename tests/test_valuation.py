"""Tests for valuation scoring."""

import pandas as pd
import pytest

from src.scoring.valuation import score_valuation


@pytest.fixture
def sample_valuation_data() -> pd.DataFrame:
    """Minimal valuation data for testing."""
    return pd.DataFrame({
        "adv_liquidity": [1e8, 5e7, 2e8],
        "pbr_vs_10yr": [0.8, 1.2, 0.9],
        "pen_vs_10yr": [0.7, 1.1, 0.95],
        "pegn": [1.0, 1.5, 0.8],
    })


def test_score_valuation_returns_series(sample_valuation_data: pd.DataFrame) -> None:
    """Scoring returns a Series with same index as input."""
    scores = score_valuation(sample_valuation_data)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(sample_valuation_data)

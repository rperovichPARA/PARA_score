"""Tests for composite scoring."""

import pandas as pd
import pytest

from src.scoring.composite import compute_composite_scores


@pytest.fixture
def sample_category_scores() -> dict[str, pd.Series]:
    """Mock category scores for three companies."""
    idx = pd.Index([0, 1, 2])
    return {
        "fundamentals": pd.Series([0.5, -0.3, 0.1], index=idx),
        "valuation": pd.Series([0.2, 0.4, -0.5], index=idx),
        "sector": pd.Series([0.0, 0.1, 0.3], index=idx),
        "factors": pd.Series([0.0, 0.0, 0.0], index=idx),
        "kozo": pd.Series([0.8, -0.2, 0.3], index=idx),
    }


def test_composite_produces_vi_and_sp(sample_category_scores: dict) -> None:
    """Composite output has VI and SP score and rank columns."""
    result = compute_composite_scores(sample_category_scores)
    assert "VI_score" in result.columns
    assert "SP_score" in result.columns
    assert "VI_rank" in result.columns
    assert "SP_rank" in result.columns


def test_composite_ranks_are_valid(sample_category_scores: dict) -> None:
    """Ranks should be 1-indexed integers covering all companies."""
    result = compute_composite_scores(sample_category_scores)
    for col in ("VI_rank", "SP_rank"):
        ranks = sorted(result[col].tolist())
        assert ranks == [1.0, 2.0, 3.0]

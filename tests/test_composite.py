"""Tests for composite scoring."""

import pandas as pd
import pytest

from src.scoring.composite import CATEGORY_NAMES, compute_composite_scores


@pytest.fixture
def sample_category_dfs() -> dict[str, pd.DataFrame]:
    """Mock category DataFrames for three companies.

    Each DataFrame contains a ``Code`` column and a single metric column
    with realistic-ish values.  The metric names match the YAML config
    so ``score_category_from_config`` can process them.
    """
    idx = pd.Index([0, 1, 2])
    base = {"Code": pd.Series(["10010", "20020", "30030"], index=idx)}

    return {
        "fundamentals": pd.DataFrame(
            {**base, "roe": [0.12, 0.05, 0.08], "opm": [0.15, 0.10, 0.20]},
            index=idx,
        ),
        "valuation": pd.DataFrame(
            {**base, "adv_liquidity": [1e9, 5e8, 2e9]},
            index=idx,
        ),
        "sector": pd.DataFrame(
            {**base},  # all NaN — stub category
            index=idx,
        ),
        "factors": pd.DataFrame(
            {**base},  # all NaN — stub category
            index=idx,
        ),
        "kozo": pd.DataFrame(
            {**base, "payout_ratio": [30.0, 50.0, 40.0]},
            index=idx,
        ),
    }


def test_composite_produces_expected_columns(
    sample_category_dfs: dict[str, pd.DataFrame],
) -> None:
    """Composite output has Code, category scores, VI/SP scores and ranks."""
    result = compute_composite_scores(sample_category_dfs)

    assert "Code" in result.columns
    for cat_name in CATEGORY_NAMES:
        assert f"{cat_name}_score" in result.columns
    assert "VI_score" in result.columns
    assert "SP_score" in result.columns
    assert "VI_rank" in result.columns
    assert "SP_rank" in result.columns


def test_composite_ranks_are_valid(
    sample_category_dfs: dict[str, pd.DataFrame],
) -> None:
    """Ranks should be 1-indexed integers covering all companies."""
    result = compute_composite_scores(sample_category_dfs)
    for col in ("VI_rank", "SP_rank"):
        ranks = sorted(result[col].tolist())
        assert ranks == [1.0, 2.0, 3.0]


def test_composite_row_count_matches_input(
    sample_category_dfs: dict[str, pd.DataFrame],
) -> None:
    """Output should have the same number of rows as the input DataFrames."""
    result = compute_composite_scores(sample_category_dfs)
    assert len(result) == 3

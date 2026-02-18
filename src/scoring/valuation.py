"""Category 2: Valuation scoring.

Covers relative valuation vs history, liquidity, price momentum,
and target price upside metrics.
"""

import logging

import pandas as pd
import yaml

from src.scoring.utils import score_category

logger = logging.getLogger(__name__)


def compute_valuation_metrics(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Derive valuation metrics from financials and price data.

    Args:
        financials: DataFrame with financial statement fields.
        prices: DataFrame with daily price/volume data.

    Returns:
        DataFrame with computed valuation metric columns.
    """
    df = financials.copy()

    # PBR (Price-to-Book)
    if "BookValuePerShare" in df.columns and "AdjustmentClose" in prices.columns:
        latest_price = prices.groupby("Code")["AdjustmentClose"].last()
        df["pbr"] = latest_price / df["BookValuePerShare"]

    # ADV liquidity (average daily value traded)
    if not prices.empty:
        prices["daily_value"] = prices["AdjustmentClose"] * prices["Volume"]
        adv = prices.groupby("Code")["daily_value"].mean()
        df["adv_liquidity"] = df.index.map(adv)

    return df


def score_valuation(
    df: pd.DataFrame,
    config_path: str = "config/scoring_weights.yaml",
) -> pd.Series:
    """Score the Valuation category.

    Args:
        df: DataFrame with computed valuation metric columns.
        config_path: Path to scoring weights YAML.

    Returns:
        Series of valuation category scores.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metric_defs = {
        name: (spec[0], spec[1])
        for name, spec in config["valuation"].items()
    }
    min_coverage = config.get("scoring", {}).get("min_coverage", 0.05)

    return score_category(df, metric_defs, min_coverage=min_coverage)

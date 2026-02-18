"""Category 5: Value Impact (Kozo) scoring.

Core of alliance investing — measures structural improvement potential.
Covers underperforming segments, cost structure, capital allocation,
governance, and engagement opportunity indicators.
"""

import logging

import pandas as pd
import yaml

from src.scoring.utils import score_category

logger = logging.getLogger(__name__)


def compute_kozo_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive Value Impact (Kozo) metrics.

    Args:
        financials: DataFrame with financial data including peer context.

    Returns:
        DataFrame with kozo metric columns.
    """
    df = financials.copy()

    # ROE vs peer average — computable from J-Quants data
    if "roe" in df.columns and "Sector33Code" in df.columns:
        peer_avg_roe = df.groupby("Sector33Code")["roe"].transform("mean")
        df["roe_vs_peer"] = df["roe"] - peer_avg_roe

    # Payout ratio — directly available
    if "ResultPayoutRatioAnnual" in df.columns:
        df["payout_ratio"] = df["ResultPayoutRatioAnnual"]

    return df


def score_kozo(
    df: pd.DataFrame,
    config_path: str = "config/scoring_weights.yaml",
) -> pd.Series:
    """Score the Value Impact (Kozo) category.

    Args:
        df: DataFrame with computed kozo metric columns.
        config_path: Path to scoring weights YAML.

    Returns:
        Series of kozo category scores.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metric_defs = {
        name: (spec[0], spec[1])
        for name, spec in config["kozo"].items()
    }
    min_coverage = config.get("scoring", {}).get("min_coverage", 0.05)

    return score_category(df, metric_defs, min_coverage=min_coverage)

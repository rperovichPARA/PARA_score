"""Category 3: Sector Attractiveness scoring.

Used only in the SP composite (0% weight in VI).
Covers cyclical momentum, sector fundamental/valuation trends,
and competitive strength.
"""

import logging

import pandas as pd
import yaml

from src.scoring.utils import score_category

logger = logging.getLogger(__name__)


def compute_sector_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive sector-level attractiveness metrics.

    Args:
        financials: DataFrame with financial data including sector codes.

    Returns:
        DataFrame with sector metric columns mapped back to each company.
    """
    df = financials.copy()
    # Sector trend - fundamentals: aggregated OP growth / ROE by sector
    # Sector trend - valuation: sector-level PE/PBR aggregates
    # These require sector-level aggregation — stub for now
    logger.info("Sector metrics computation is partially implemented")
    return df


def score_sector(
    df: pd.DataFrame,
    config_path: str = "config/scoring_weights.yaml",
) -> pd.Series:
    """Score the Sector Attractiveness category.

    Args:
        df: DataFrame with computed sector metric columns.
        config_path: Path to scoring weights YAML.

    Returns:
        Series of sector category scores.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metric_defs = {
        name: (spec[0], spec[1])
        for name, spec in config["sector"].items()
    }
    min_coverage = config.get("scoring", {}).get("min_coverage", 0.05)

    return score_category(df, metric_defs, min_coverage=min_coverage)

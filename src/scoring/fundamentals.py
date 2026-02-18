"""Category 1: Fundamentals scoring.

Covers earnings growth, profitability, balance sheet strength,
and financial health metrics.
"""

import logging

import pandas as pd
import yaml

from src.scoring.utils import score_category

logger = logging.getLogger(__name__)


def compute_fundamentals_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive fundamental metrics from raw financial statement data.

    Args:
        financials: DataFrame with J-Quants financial statement fields.

    Returns:
        DataFrame with computed metric columns ready for scoring.
    """
    df = financials.copy()

    # F2/F0 Sales growth
    df["f2_f0_sales_growth"] = df["NextYearForecastNetSales"] / df["NetSales"]

    # F2/F0 EBIT growth (OP ≈ EBIT for Japan)
    df["f2_f0_ebit_growth"] = (
        df["NextYearForecastOperatingProfit"] / df["OperatingProfit"]
    )

    # F1/F0 OP growth
    df["f1_f0_op_growth"] = df["ForecastOperatingProfit"] / df["OperatingProfit"]

    # F2/F1 OP growth
    df["f2_f1_op_growth"] = (
        df["NextYearForecastOperatingProfit"] / df["ForecastOperatingProfit"]
    )

    # Operating profit margin
    df["opm"] = df["OperatingProfit"] / df["NetSales"]

    # ROE
    df["roe"] = df["Profit"] / df["Equity"]

    # Equity ratio (provided directly by J-Quants)
    df["equity_ratio"] = df.get("EquityToAssetRatio", pd.Series(dtype=float))

    return df


def score_fundamentals(
    df: pd.DataFrame,
    config_path: str = "config/scoring_weights.yaml",
) -> pd.Series:
    """Score the Fundamentals category.

    Args:
        df: DataFrame with computed fundamental metric columns.
        config_path: Path to scoring weights YAML.

    Returns:
        Series of fundamentals category scores.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metric_defs = {
        name: (spec[0], spec[1])
        for name, spec in config["fundamentals"].items()
    }
    min_coverage = config.get("scoring", {}).get("min_coverage", 0.05)

    return score_category(df, metric_defs, min_coverage=min_coverage)

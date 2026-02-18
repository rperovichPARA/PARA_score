"""Category 4: Factors / Themes scoring.

Covers factor fit (size, value/growth, dividend yield exposures)
and theme fit (AI, ESG, takeover candidates, etc.).

Note: Factor scoring is currently a placeholder (0.0). Needs a
factor model built from returns data.
"""

import logging

import pandas as pd
import yaml

from src.scoring.utils import score_category

logger = logging.getLogger(__name__)


def compute_factor_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive factor and theme fit metrics.

    Args:
        financials: DataFrame with financial data.

    Returns:
        DataFrame with factor/theme metric columns.
    """
    df = financials.copy()
    # Placeholder — factor_fit and theme_fit are not yet computable
    df["factor_fit"] = 0.0
    df["theme_fit"] = 0.0
    logger.warning("Factor/theme metrics are placeholders (0.0)")
    return df


def score_factors(
    df: pd.DataFrame,
    config_path: str = "config/scoring_weights.yaml",
) -> pd.Series:
    """Score the Factors / Themes category.

    Args:
        df: DataFrame with computed factor metric columns.
        config_path: Path to scoring weights YAML.

    Returns:
        Series of factor/theme category scores.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metric_defs = {
        name: (spec[0], spec[1])
        for name, spec in config["factors"].items()
    }
    min_coverage = config.get("scoring", {}).get("min_coverage", 0.05)

    return score_category(df, metric_defs, min_coverage=min_coverage)

"""Scoring utility functions.

Provides winsorized z-scoring, normalization, and common helpers
used across all scoring category modules.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def winsorized_zscore(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.Series:
    """Compute a winsorized z-score for a series.

    1. Clip values at the lower_pct and upper_pct percentiles.
    2. Subtract the mean and divide by standard deviation.

    Args:
        series: Raw metric values (may contain NaN).
        lower_pct: Lower percentile for winsorization (default 1%).
        upper_pct: Upper percentile for winsorization (default 99%).

    Returns:
        Z-scored series with NaN preserved.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index)

    lower_bound = valid.quantile(lower_pct)
    upper_bound = valid.quantile(upper_pct)
    clipped = series.clip(lower=lower_bound, upper=upper_bound)

    mean = clipped.mean()
    std = clipped.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)

    return (clipped - mean) / std


def score_category(
    df: pd.DataFrame,
    metric_defs: dict[str, tuple[float, bool]],
    min_coverage: float = 0.05,
) -> pd.Series:
    """Score a category by weighted combination of winsorized z-scored metrics.

    Args:
        df: DataFrame with one row per company, columns for each metric.
        metric_defs: Dict mapping metric_name -> (weight, higher_is_better).
        min_coverage: Minimum fraction of non-null values to include a metric.

    Returns:
        Series of category scores indexed by company.
    """
    total = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for metric_name, (weight, higher_is_better) in metric_defs.items():
        if metric_name not in df.columns:
            continue
        values = df[metric_name]
        if values.notna().mean() < min_coverage:
            logger.debug("Skipping %s — coverage %.1f%% < %.1f%% threshold",
                         metric_name, values.notna().mean() * 100, min_coverage * 100)
            continue

        z = winsorized_zscore(values)
        if not higher_is_better:
            z = -z

        total += z.fillna(0.0) * weight
        total_weight += weight

    if 0 < total_weight < 1:
        total /= total_weight

    return total

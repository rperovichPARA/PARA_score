"""Scoring utility functions.

Provides winsorized z-scoring, normalization, and common helpers
used across all scoring category modules. Loads metric definitions
and scoring parameters from ``config/scoring_weights.yaml``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scoring_weights.yaml"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the scoring weights YAML configuration.

    Args:
        config_path: Path to the YAML file.  Defaults to
            ``<repo_root>/config/scoring_weights.yaml``.

    Returns:
        Parsed YAML as a dictionary.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    return config


def load_metric_defs(
    category: str,
    config_path: str | Path | None = None,
) -> dict[str, tuple[float, bool]]:
    """Return metric definitions for a scoring category.

    Reads the ``<category>`` key from the YAML config and converts each
    entry from ``[weight, higher_is_better]`` list form into the
    ``{metric_name: (weight, higher_is_better)}`` dict expected by
    :func:`score_category`.

    Args:
        category: One of ``fundamentals``, ``valuation``, ``sector``,
            ``factors``, or ``kozo``.
        config_path: Optional override for the YAML file path.

    Returns:
        Dict mapping metric name to ``(weight, higher_is_better)`` tuple.

    Raises:
        KeyError: If the requested category is not present in the config.
    """
    config = load_config(config_path)
    if category not in config:
        raise KeyError(
            f"Category '{category}' not found in config. "
            f"Available: {[k for k in config if k not in ('category_weights', 'scoring')]}"
        )
    return {
        name: (float(spec[0]), bool(spec[1]))
        for name, spec in config[category].items()
    }


def load_category_weights(
    composite: str,
    config_path: str | Path | None = None,
) -> dict[str, float]:
    """Return category-level weights for a composite score (VI or SP).

    Args:
        composite: ``"VI"`` or ``"SP"``.
        config_path: Optional override for the YAML file path.

    Returns:
        Dict mapping category name to its weight in the composite.
    """
    config = load_config(config_path)
    return {k: float(v) for k, v in config["category_weights"][composite].items()}


def load_scoring_params(
    config_path: str | Path | None = None,
) -> dict[str, float]:
    """Return global scoring parameters from the config.

    Returns a dict with keys ``winsorize_lower``, ``winsorize_upper``,
    and ``min_coverage``.
    """
    config = load_config(config_path)
    defaults = {"winsorize_lower": 0.01, "winsorize_upper": 0.99, "min_coverage": 0.05}
    return {k: float(config.get("scoring", {}).get(k, v)) for k, v in defaults.items()}


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


def score_category_from_config(
    df: pd.DataFrame,
    category: str,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score a category using metric definitions loaded from the YAML config.

    Convenience wrapper that combines :func:`load_metric_defs`,
    :func:`load_scoring_params`, and :func:`score_category`.

    Args:
        df: DataFrame with one row per company, columns for each metric.
        category: Category key in the config (e.g. ``"fundamentals"``).
        config_path: Optional override for the YAML file path.

    Returns:
        Series of category scores indexed by company.
    """
    metric_defs = load_metric_defs(category, config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

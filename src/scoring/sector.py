"""Category 3: Sector Attractiveness scoring.

Used only in the SP composite (0% weight in VI).
Covers cyclical momentum, sector fundamental/valuation trends,
and competitive strength.

Data sources
------------
**Unavailable (NaN):**
    All four sector metrics require external data that is not yet
    integrated:

    * ``cyclical_momentum`` -- Needs external macro / business-cycle data.
    * ``sector_trend_fund`` -- Aggregated OP growth / ROE / margins by
      TOPIX sector.  Partially computable from J-Quants once the sector
      rotation framework is wired in.
    * ``sector_trend_val`` -- Sector-level PE / PBR aggregates.  Same
      dependency as ``sector_trend_fund``.
    * ``competitive_strength`` -- Qualitative assessment; no automated
      data source.

The sector rotation framework (multi-window z-scored relative strength
against TOPIX with TOPIX-17 sectors) should eventually feed into this
module once the integration is built.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

logger = logging.getLogger(__name__)

# All sector YAML metric keys (for NaN initialisation and coverage reporting).
_ALL_METRICS: list[str] = [
    "cyclical_momentum",
    "sector_trend_fund",
    "sector_trend_val",
    "competitive_strength",
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_sector_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive sector attractiveness metrics.

    Currently a **stub**: every metric column is set to ``NaN`` because the
    required data sources (macro data, sector rotation framework, qualitative
    assessments) are not yet integrated.  This ensures the composite scorer
    can run end-to-end — ``score_category`` will skip metrics that fall below
    the coverage threshold.

    Args:
        financials: DataFrame with financial data including sector codes.

    Returns:
        Copy of *financials* with NaN columns for each sector metric.
    """
    df = financials.copy()

    for metric in _ALL_METRICS:
        df[metric] = np.nan

    logger.info(
        "Sector metrics are stubs (NaN) — awaiting data source integration"
    )

    # Coverage summary (all will be 0.0% for now).
    for col in _ALL_METRICS:
        coverage = df[col].notna().mean()
        logger.info(
            "Sector  %-25s coverage: %5.1f%%", col, coverage * 100,
        )

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_sector(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Sector Attractiveness category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    With all metrics at NaN, the returned scores will be zero — no metrics
    pass the coverage gate.

    Args:
        df: DataFrame with computed sector metric columns (output of
            :func:`compute_sector_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of sector category scores indexed by company.
    """
    metric_defs = load_metric_defs("sector", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

"""Category 4: Factors / Themes scoring.

Covers factor fit (size, value/growth, dividend yield exposures)
and theme fit (AI chain, ESG, takeover candidates, etc.).

Data sources
------------
**Unavailable (NaN):**
    Both metrics require data or models that are not yet integrated:

    * ``factor_fit`` -- Needs a factor model built from returns data.
      Size, value/growth, and dividend yield factor exposures are
      partially derivable from J-Quants prices and financials, but
      the factor model infrastructure has not been built yet.
    * ``theme_fit`` -- Qualitative theme tagging (AI chain, ESG,
      takeover candidates).  Requires proprietary classification
      that is maintained outside this pipeline.

Once the factor model from returns data is available, ``factor_fit``
should be computed here.  ``theme_fit`` will need a separate data
ingestion path (e.g. a Google Sheet or Pinecone lookup).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

logger = logging.getLogger(__name__)

# All factors YAML metric keys (for NaN initialisation and coverage reporting).
_ALL_METRICS: list[str] = [
    "factor_fit",
    "theme_fit",
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_factor_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive factor and theme fit metrics.

    Currently a **stub**: every metric column is set to ``NaN`` because the
    required data sources (factor model from returns, proprietary theme tags)
    are not yet available.  This ensures the composite scorer can run
    end-to-end — ``score_category`` will skip metrics that fall below the
    coverage threshold.

    Args:
        financials: DataFrame with financial data.

    Returns:
        Copy of *financials* with NaN columns for each factor/theme metric.
    """
    df = financials.copy()

    for metric in _ALL_METRICS:
        df[metric] = np.nan

    logger.info(
        "Factor/theme metrics are stubs (NaN) — awaiting data source integration"
    )

    # Coverage summary (all will be 0.0% for now).
    for col in _ALL_METRICS:
        coverage = df[col].notna().mean()
        logger.info(
            "Factors %-25s coverage: %5.1f%%", col, coverage * 100,
        )

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_factors(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Factors / Themes category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    With all metrics at NaN, the returned scores will be zero — no metrics
    pass the coverage gate.

    Args:
        df: DataFrame with computed factor metric columns (output of
            :func:`compute_factor_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of factor/theme category scores indexed by company.
    """
    metric_defs = load_metric_defs("factors", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

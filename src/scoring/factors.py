"""Category 4: Factors / Themes scoring.

Computes quantitative factor exposures from daily price data and
financial statements:

* **Price momentum 6M** — 6-month total return minus TOPIX benchmark
  return over the same window.
* **Price momentum 12M** — 12-month total return minus TOPIX return.
  Set to ``NaN`` when fewer than ~12 months of daily data are available.
* **Earnings momentum** — Year-over-year change in operating profit,
  approximated as ``(ForecastOperatingProfit − OperatingProfit) /
  |OperatingProfit|`` from the latest financial statements.
* **Volatility** — Annualised standard deviation of daily returns.

Data sources
------------
* Daily quotes (``prices`` DataFrame) — already fetched by the pipeline
  for the 6-month window.  Provides price momentum and volatility.
* Financial statements (``financials`` DataFrame) — deduplicated one-row-
  per-company.  Provides earnings momentum via actual vs forecast OP.
* TOPIX index (``topix`` DataFrame) — used as benchmark for excess
  returns in momentum metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

logger = logging.getLogger(__name__)

# All factors YAML metric keys.
_ALL_METRICS: list[str] = [
    "price_momentum_6m",
    "price_momentum_12m",
    "earnings_momentum",
    "volatility",
]

# Approximate number of trading days per calendar year (used for
# annualising volatility).
_TRADING_DAYS_PER_YEAR: int = 252

# Minimum number of trading days required per stock to compute
# meaningful 6-month returns and volatility.
_MIN_DAYS_6M: int = 20

# Minimum calendar-day span between the first and last trade date for a
# stock to qualify for the 12-month momentum calculation.
_MIN_CALENDAR_DAYS_12M: int = 360


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_topix_return(topix: Optional[pd.DataFrame]) -> float:
    """Compute the TOPIX total return over the available date window.

    Args:
        topix: TOPIX index DataFrame with ``Date`` and ``Close`` columns.

    Returns:
        Total return as a decimal (e.g. 0.05 for +5%).  Returns ``0.0``
        if the data is unavailable or insufficient.
    """
    if topix is None or topix.empty:
        return 0.0
    if not {"Close", "Date"}.issubset(topix.columns):
        return 0.0

    topix_sorted = topix.sort_values("Date")
    if len(topix_sorted) < 2:
        return 0.0

    first_val = float(topix_sorted["Close"].iloc[0])
    last_val = float(topix_sorted["Close"].iloc[-1])
    if first_val <= 0:
        return 0.0
    return (last_val / first_val) - 1.0


def _compute_stock_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock return and volatility statistics from daily quotes.

    Args:
        prices: Daily quotes with ``Code``, ``Date``, ``AdjustmentClose``.

    Returns:
        DataFrame indexed by ``Code`` with columns:

        * ``total_return`` — Price return from first to last observation.
        * ``ann_volatility`` — Annualised standard deviation of daily
          returns.
        * ``n_days`` — Number of non-null price observations.
        * ``calendar_span`` — Calendar days between first and last trade.
    """
    required = {"Code", "Date", "AdjustmentClose"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(
            columns=["total_return", "ann_volatility", "n_days", "calendar_span"],
        )

    px = prices[["Code", "Date", "AdjustmentClose"]].copy()
    px["AdjustmentClose"] = pd.to_numeric(px["AdjustmentClose"], errors="coerce")
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px = px.dropna(subset=["AdjustmentClose", "Date"])
    px = px.sort_values(["Code", "Date"])

    # Daily returns per stock.
    px["daily_return"] = px.groupby("Code")["AdjustmentClose"].pct_change()

    # Aggregate per stock.
    first = px.groupby("Code").first()
    last = px.groupby("Code").last()

    first_price = first["AdjustmentClose"]
    last_price = last["AdjustmentClose"]
    total_return = (last_price / first_price.replace(0, np.nan)) - 1.0

    first_date = first["Date"]
    last_date = last["Date"]
    calendar_span = (last_date - first_date).dt.days

    n_days = px.groupby("Code")["AdjustmentClose"].count()
    ann_vol = px.groupby("Code")["daily_return"].std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    stats = pd.DataFrame({
        "total_return": total_return,
        "ann_volatility": ann_vol,
        "n_days": n_days,
        "calendar_span": calendar_span,
    })
    return stats


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_factor_metrics(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    topix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive factor metrics from daily price data and financial statements.

    Args:
        financials: Deduplicated financials DataFrame (one row per
            company).  Expected to contain ``Code``,
            ``OperatingProfit``, and ``ForecastOperatingProfit``.
        prices: Daily quotes DataFrame with ``Code``, ``Date``,
            ``AdjustmentClose``.
        topix: Optional TOPIX index DataFrame with ``Date`` and
            ``Close`` columns.  Used as benchmark for excess return
            in momentum metrics.

    Returns:
        Copy of *financials* with the 4 factor metric columns added.
    """
    df = financials.copy()

    code_col: Optional[pd.Series] = None
    if "Code" in df.columns:
        code_col = df["Code"].astype(str).str.strip()

    # Initialise all metric columns to NaN.
    for metric in _ALL_METRICS:
        df[metric] = np.nan

    # ==================================================================
    # 1. Price-based metrics (momentum + volatility)
    # ==================================================================
    if prices is not None and not prices.empty and code_col is not None:
        stock_stats = _compute_stock_stats(prices)

        if not stock_stats.empty:
            topix_return = _compute_topix_return(topix)
            logger.info(
                "TOPIX return over price window: %.2f%%", topix_return * 100,
            )

            # ── 6-month momentum (excess return vs TOPIX) ────────────
            # Require at least _MIN_DAYS_6M trading days per stock.
            n_days = code_col.map(stock_stats["n_days"])
            has_enough = n_days >= _MIN_DAYS_6M

            raw_return = code_col.map(stock_stats["total_return"])
            momentum_6m = raw_return - topix_return
            # Mask stocks with insufficient data.
            momentum_6m[~has_enough.values] = np.nan
            df["price_momentum_6m"] = momentum_6m.values

            logger.info(
                "6M momentum: %d / %d stocks with sufficient data (>= %d days)",
                has_enough.sum(), len(df), _MIN_DAYS_6M,
            )

            # ── 12-month momentum ────────────────────────────────────
            # Only valid when the price window spans >= _MIN_CALENDAR_DAYS_12M.
            calendar_span = code_col.map(stock_stats["calendar_span"])
            has_12m = calendar_span >= _MIN_CALENDAR_DAYS_12M

            if has_12m.any():
                momentum_12m = raw_return - topix_return
                momentum_12m[~has_12m.values] = np.nan
                df["price_momentum_12m"] = momentum_12m.values
                logger.info(
                    "12M momentum: %d stocks with sufficient history (>= %d calendar days)",
                    has_12m.sum(), _MIN_CALENDAR_DAYS_12M,
                )
            else:
                logger.info(
                    "12M momentum: no stocks have >= %d calendar days of data; "
                    "price_momentum_12m will be NaN for all.",
                    _MIN_CALENDAR_DAYS_12M,
                )

            # ── Volatility ───────────────────────────────────────────
            ann_vol = code_col.map(stock_stats["ann_volatility"])
            ann_vol[~has_enough.values] = np.nan
            df["volatility"] = ann_vol.values

    else:
        logger.warning(
            "Price data unavailable; price_momentum_6m, price_momentum_12m, "
            "and volatility will be NaN."
        )

    # ==================================================================
    # 2. Earnings momentum (YoY operating profit change)
    # ==================================================================
    if {"ForecastOperatingProfit", "OperatingProfit"}.issubset(df.columns):
        actual_op = pd.to_numeric(df["OperatingProfit"], errors="coerce")
        forecast_op = pd.to_numeric(df["ForecastOperatingProfit"], errors="coerce")

        # Use absolute value of actual OP as the denominator so that
        # improvements from a negative base still produce a positive signal.
        abs_actual = actual_op.abs()
        abs_actual[abs_actual == 0] = np.nan

        df["earnings_momentum"] = (forecast_op - actual_op) / abs_actual
    else:
        logger.warning(
            "Missing ForecastOperatingProfit or OperatingProfit; "
            "earnings_momentum will be NaN."
        )

    # ==================================================================
    # Coverage summary
    # ==================================================================
    for col in _ALL_METRICS:
        coverage = df[col].notna().mean()
        logger.info(
            "Factors  %-25s coverage: %5.1f%%", col, coverage * 100,
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

"""Category 1: Fundamentals scoring.

Computes growth, profitability, leverage, and financial health metrics.
Combines J-Quants financial statement data with supplementary data from
the PARA.FS.data Google Sheet for metrics not fully available via the
J-Quants Standard plan (Altman Z-Score, net cash / market cap, and
growth metric overrides).

Google Sheet reference (PARA.FS.data):
    https://docs.google.com/spreadsheets/d/10mjEbmtJC6y5tCqnQ_SrUQAfheDhafAtpX0DjJJO5fk/edit?gid=0#gid=0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PARA.FS.data Google Sheet — supplementary data source
# ---------------------------------------------------------------------------
# This sheet provides metrics that cannot be fully computed from J-Quants
# Standard-plan data alone.  It is expected to contain a ``Code`` column
# (security code) as the join key, plus any of the metric columns listed
# in ``_SUPPLEMENT_METRIC_COLS``.
#
# The sheet is populated by separate pipeline steps and may be empty
# during initial setup — this is handled gracefully.

GSHEET_ID = "10mjEbmtJC6y5tCqnQ_SrUQAfheDhafAtpX0DjJJO5fk"
GSHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}"
    "/export?format=csv&gid=0"
)

# Supplement columns that can override or fill J-Quants-derived values.
_SUPPLEMENT_METRIC_COLS: list[str] = [
    "f2_f0_sales_growth",
    "f2_f0_ebit_growth",
    "f1_f0_op_growth",
    "f2_f1_op_growth",
    "opm",
    "roe",
    "equity_ratio",
    "net_cash_mktcap",
    "altman_z",
]


# ---------------------------------------------------------------------------
# Google Sheet loader
# ---------------------------------------------------------------------------

def load_gsheet_supplement(url: str = GSHEET_CSV_URL) -> pd.DataFrame:
    """Load supplementary fundamental metrics from the PARA.FS.data sheet.

    The sheet is read via its public CSV export URL.  A ``Code`` column
    is required as the join key; any columns matching known metric names
    are coerced to numeric and returned.

    Args:
        url: CSV export URL for the Google Sheet.

    Returns:
        DataFrame indexed by ``Code`` with available supplement columns.
        Returns an empty DataFrame if the sheet is unreachable, empty,
        or missing a ``Code`` column.
    """
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        logger.warning(
            "Could not load PARA.FS.data supplement sheet: %s. "
            "Supplement metrics will be unavailable.",
            exc,
        )
        return pd.DataFrame()

    if df.empty or "Code" not in df.columns:
        logger.info(
            "PARA.FS.data sheet is empty or missing 'Code' column; "
            "supplement metrics unavailable."
        )
        return pd.DataFrame()

    df["Code"] = df["Code"].astype(str).str.strip()
    df = df.set_index("Code")

    # Keep only recognised metric columns and coerce to numeric.
    keep = [c for c in df.columns if c in _SUPPLEMENT_METRIC_COLS]
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Loaded PARA.FS.data supplement: %d rows, columns: %s",
        len(df),
        keep,
    )
    return df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute *numerator / denominator* with guards for bad bases.

    Returns ``NaN`` where the denominator is zero or negative (growth
    ratios from a negative base are misleading).
    """
    denom = denominator.copy()
    denom[denom <= 0] = np.nan
    return numerator / denom


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_fundamentals_metrics(
    financials: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive all 9 fundamentals metrics.

    **J-Quants-derived metrics** (computed directly when source columns
    are present):

    * ``f2_f0_sales_growth`` — ``NextYearForecastNetSales / NetSales``
    * ``f2_f0_ebit_growth`` — ``NextYearForecastOperatingProfit / OperatingProfit``
    * ``f1_f0_op_growth`` — ``ForecastOperatingProfit / OperatingProfit``
    * ``f2_f1_op_growth`` — ``NextYearForecastOperatingProfit / ForecastOperatingProfit``
    * ``opm`` — ``OperatingProfit / NetSales``
    * ``roe`` — ``Profit / Equity``
    * ``equity_ratio`` — ``EquityToAssetRatio`` (provided directly)

    **Supplement-dependent metrics** (loaded from the PARA.FS.data Google
    Sheet when available; these cannot be fully computed from J-Quants
    Standard-plan data):

    * ``net_cash_mktcap`` — Net cash position / market capitalisation.
      Requires Premium ``/fins/fs_details`` for a proper net-debt
      breakdown; the supplement sheet provides pre-computed values.
    * ``altman_z`` — Altman Z-Score.  Not derivable from J-Quants
      summary data; sourced externally via the supplement sheet.

    The supplement sheet also acts as a fallback for the growth and
    profitability metrics above: if a company's J-Quants data is
    missing a value but the sheet has one, the sheet value fills the
    gap.

    Args:
        financials: Prepared financials DataFrame with J-Quants fields.
            Expected columns include ``NetSales``, ``OperatingProfit``,
            ``ForecastOperatingProfit``, ``NextYearForecastNetSales``,
            ``NextYearForecastOperatingProfit``, ``Profit``, ``Equity``,
            ``EquityToAssetRatio``.  A ``Code`` column is required for
            joining supplement data.
        prices: Optional daily price DataFrame (``Code``,
            ``AdjustmentClose``, ``Volume``, ``Date``).  Used to derive
            market-cap for ``net_cash_mktcap`` if supplement data is
            not available.

    Returns:
        Copy of *financials* with the 9 computed metric columns added.
    """
    df = financials.copy()

    # ==================================================================
    # 1. J-Quants-derived metrics
    # ==================================================================

    # --- Growth ratios ---------------------------------------------------

    # F2/F0 Sales Growth
    if {"NextYearForecastNetSales", "NetSales"}.issubset(df.columns):
        df["f2_f0_sales_growth"] = _safe_ratio(
            df["NextYearForecastNetSales"], df["NetSales"],
        )
    else:
        logger.debug("Missing columns for f2_f0_sales_growth")

    # F2/F0 EBIT (≈ OP) Growth
    if {"NextYearForecastOperatingProfit", "OperatingProfit"}.issubset(df.columns):
        df["f2_f0_ebit_growth"] = _safe_ratio(
            df["NextYearForecastOperatingProfit"], df["OperatingProfit"],
        )
    else:
        logger.debug("Missing columns for f2_f0_ebit_growth")

    # F1/F0 OP Growth
    if {"ForecastOperatingProfit", "OperatingProfit"}.issubset(df.columns):
        df["f1_f0_op_growth"] = _safe_ratio(
            df["ForecastOperatingProfit"], df["OperatingProfit"],
        )
    else:
        logger.debug("Missing columns for f1_f0_op_growth")

    # F2/F1 OP Growth
    if {"NextYearForecastOperatingProfit", "ForecastOperatingProfit"}.issubset(df.columns):
        df["f2_f1_op_growth"] = _safe_ratio(
            df["NextYearForecastOperatingProfit"],
            df["ForecastOperatingProfit"],
        )
    else:
        logger.debug("Missing columns for f2_f1_op_growth")

    # --- Profitability ---------------------------------------------------

    # Operating Profit Margin
    if {"OperatingProfit", "NetSales"}.issubset(df.columns):
        denom = df["NetSales"].replace(0, np.nan)
        df["opm"] = df["OperatingProfit"] / denom
    else:
        logger.debug("Missing columns for opm")

    # Return on Equity
    if {"Profit", "Equity"}.issubset(df.columns):
        denom = df["Equity"].replace(0, np.nan)
        df["roe"] = df["Profit"] / denom
    else:
        logger.debug("Missing columns for roe")

    # --- Balance sheet ---------------------------------------------------

    # Equity Ratio (directly available from J-Quants)
    if "EquityToAssetRatio" in df.columns:
        df["equity_ratio"] = df["EquityToAssetRatio"]
    else:
        logger.debug("Missing EquityToAssetRatio for equity_ratio")

    # ==================================================================
    # 2. Market-cap proxy for net_cash_mktcap (best-effort)
    # ==================================================================
    # A proper net-cash figure needs Premium /fins/fs_details for the
    # full balance-sheet breakdown.  When price data is available we
    # store market cap so downstream supplement logic can divide any
    # gross-cash proxy that might exist.  The authoritative values
    # come from the PARA.FS.data supplement sheet.

    if prices is not None and not prices.empty and "Code" in df.columns:
        if {"AdjustmentClose", "Code", "Date"}.issubset(prices.columns):
            latest_prices = (
                prices.sort_values("Date")
                .groupby("Code")["AdjustmentClose"]
                .last()
            )
            shares_col = (
                "NumberOfIssuedAndOutstandingSharesAtTheEndOf"
                "FiscalYearIncludingTreasuryStock"
            )
            if shares_col in df.columns:
                mkt_cap = (
                    df["Code"].astype(str).map(latest_prices.astype(float))
                    * df[shares_col]
                )
                df["_market_cap"] = mkt_cap
                logger.debug(
                    "Market-cap proxy computed for %d companies",
                    mkt_cap.notna().sum(),
                )

    # ==================================================================
    # 3. Merge supplement data from PARA.FS.data Google Sheet
    # ==================================================================
    # The sheet is the authoritative source for altman_z and
    # net_cash_mktcap, and provides fallback values for the growth
    # and profitability metrics where J-Quants data is incomplete.

    supplement = load_gsheet_supplement()

    if not supplement.empty and "Code" in df.columns:
        join_key = df["Code"].astype(str).str.strip()

        for metric_col in _SUPPLEMENT_METRIC_COLS:
            if metric_col not in supplement.columns:
                continue

            supp_values = join_key.map(supplement[metric_col])

            if metric_col in df.columns:
                # Fill only where the J-Quants-derived value is missing.
                before_nulls = df[metric_col].isna().sum()
                df[metric_col] = df[metric_col].fillna(
                    pd.Series(supp_values.values, index=df.index),
                )
                filled = before_nulls - df[metric_col].isna().sum()
            else:
                df[metric_col] = supp_values.values
                filled = pd.Series(supp_values.values).notna().sum()

            if filled > 0:
                logger.info(
                    "Supplement filled %d values for %s", filled, metric_col,
                )

    # ==================================================================
    # 4. Coverage summary
    # ==================================================================
    _ALL_METRICS = [
        "f2_f0_sales_growth", "f2_f0_ebit_growth",
        "f1_f0_op_growth", "f2_f1_op_growth",
        "opm", "roe", "equity_ratio",
        "net_cash_mktcap", "altman_z",
    ]
    for col in _ALL_METRICS:
        if col in df.columns:
            coverage = df[col].notna().mean()
            logger.info(
                "Fundamentals %-25s coverage: %5.1f%%", col, coverage * 100,
            )
        else:
            logger.warning("Fundamentals metric %s was not computed", col)

    # Clean up internal helper columns.
    df.drop(columns=["_market_cap"], errors="ignore", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_fundamentals(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Fundamentals category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    Args:
        df: DataFrame with computed fundamentals metric columns (output
            of :func:`compute_fundamentals_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of fundamentals category scores indexed by company.
    """
    metric_defs = load_metric_defs("fundamentals", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

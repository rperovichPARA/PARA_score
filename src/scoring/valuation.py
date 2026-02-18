"""Category 2: Valuation scoring.

Covers relative valuation vs history, liquidity, price momentum,
and target price upside metrics.

Combines J-Quants financial statements and daily bars with supplementary
data from the PARA.FS.data Google Sheet for metrics not available via
J-Quants alone (broker target upside, PARA target upside, historical
PBR/PE averages).

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
# Same sheet used by fundamentals.py.  Valuation-specific columns include
# broker/PARA target upside, PBR/PE vs 10yr averages, and PEGn overrides.
# The sheet is populated by separate pipeline steps and may be empty.

GSHEET_ID = "10mjEbmtJC6y5tCqnQ_SrUQAfheDhafAtpX0DjJJO5fk"
GSHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}"
    "/export?format=csv&gid=0"
)

# Supplement columns that can override or fill J-Quants-derived values.
_SUPPLEMENT_METRIC_COLS: list[str] = [
    "pbr_vs_10yr",
    "pen_vs_10yr",
    "pegn",
    "adv_liquidity",
    "broker_target_upside",
    "peer_target_upside",
    "price_6mo_vs_tpx",
    "para_target_upside",
]


# ---------------------------------------------------------------------------
# Google Sheet loader
# ---------------------------------------------------------------------------

def load_gsheet_supplement(url: str = GSHEET_CSV_URL) -> pd.DataFrame:
    """Load supplementary valuation metrics from the PARA.FS.data sheet.

    The sheet is read via its public CSV export URL.  A ``Code`` column
    is required as the join key; any columns matching known valuation
    metric names are coerced to numeric and returned.

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
            "Could not load valuation supplement sheet: %s. "
            "Supplement metrics will be unavailable.",
            exc,
        )
        return pd.DataFrame()

    if df.empty or "Code" not in df.columns:
        logger.info(
            "Valuation supplement sheet is empty or missing 'Code' column; "
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
        "Loaded valuation supplement: %d rows, columns: %s",
        len(df),
        keep,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute *numerator / denominator* with guards for bad bases.

    Returns ``NaN`` where the denominator is zero or negative (growth
    ratios from a negative base are misleading).
    """
    denom = denominator.copy()
    denom[denom <= 0] = np.nan
    return numerator / denom


def _get_latest_prices(prices: pd.DataFrame) -> pd.Series:
    """Return the most recent adjusted close price per security code.

    Args:
        prices: Daily quotes with ``Code``, ``Date``, ``AdjustmentClose``.

    Returns:
        Series indexed by ``Code`` with the latest adjusted close price.
    """
    return (
        prices.sort_values("Date")
        .groupby("Code")["AdjustmentClose"]
        .last()
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_valuation_metrics(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    topix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Derive all 8 valuation metrics.

    **J-Quants-derived metrics** (computed when source columns are present):

    * ``pbr_vs_10yr`` — Current PBR (Price / BookValuePerShare).  Best-effort
      proxy; the Google Sheet provides the true ratio vs 10yr average when
      available.
    * ``pen_vs_10yr`` — Current PE (Price / EarningsPerShare).  Same caveat
      as PBR; sheet overrides with the 10yr-relative value.
    * ``pegn`` — PE / annualised forecast earnings growth (%).  Uses F1 EPS
      growth, falling back to annualised F2 growth.
    * ``adv_liquidity`` — Trailing average daily traded value
      (AdjustmentClose * Volume).
    * ``peer_target_upside`` — Sector median PE * company F2 EPS / current
      price − 1.  Peer-implied upside.
    * ``price_6mo_vs_tpx`` — Stock 6-month return minus TOPIX 6-month
      return.  Requires daily bars and a TOPIX index DataFrame.

    **Supplement-only metrics** (loaded from the PARA.FS.data Google Sheet):

    * ``broker_target_upside`` — Consensus target upside.  Requires
      Bloomberg BEST / IBES; sourced via the supplement sheet.
    * ``para_target_upside`` — Proprietary PARA model target upside.
      Populated externally.

    The supplement sheet also acts as a fallback for all J-Quants-derived
    metrics: if a company's computed value is missing but the sheet has
    one, the sheet value fills the gap.

    Args:
        financials: Prepared financials DataFrame with J-Quants fields.
            Expected columns include ``BookValuePerShare``,
            ``EarningsPerShare``, ``ForecastEarningsPerShare``,
            ``NextYearForecastEarningsPerShare``, ``Sector33Code``.
            A ``Code`` column is required for joining supplement data
            and mapping price data.
        prices: Daily price DataFrame with ``Code``, ``Date``,
            ``AdjustmentClose``, and ``Volume``.  Used for ADV, PBR/PE
            computation, peer upside, and relative price performance.
        topix: Optional TOPIX index DataFrame with ``Date`` and ``Close``
            columns.  Required for ``price_6mo_vs_tpx``.

    Returns:
        Copy of *financials* with valuation metric columns added.
    """
    df = financials.copy()

    code_col: Optional[pd.Series] = None
    if "Code" in df.columns:
        code_col = df["Code"].astype(str).str.strip()

    # Map the latest adjusted close price onto the financials frame.
    has_prices = (
        not prices.empty
        and {"AdjustmentClose", "Code", "Date"}.issubset(prices.columns)
    )
    if has_prices and code_col is not None:
        latest_prices = _get_latest_prices(prices)
        df["_latest_price"] = code_col.map(latest_prices.astype(float))
    else:
        logger.warning(
            "Price data unavailable or missing required columns; "
            "price-dependent valuation metrics will be NaN."
        )

    # ==================================================================
    # 1. PBR vs 10yr average
    # ==================================================================
    # Full metric is current PBR / 10yr-avg PBR.  With only ~6 months of
    # daily bars we compute current PBR as a cross-sectional proxy.  The
    # Google Sheet overrides with the true historical ratio when available.

    if "_latest_price" in df.columns and "BookValuePerShare" in df.columns:
        bvps = df["BookValuePerShare"].replace(0, np.nan)
        df["pbr_vs_10yr"] = df["_latest_price"] / bvps
    else:
        logger.debug("Missing columns for pbr_vs_10yr")

    # ==================================================================
    # 2. PEn vs 10yr average
    # ==================================================================
    # Same approach as PBR: current PE as proxy, sheet provides the full
    # 10yr-relative value.

    if "_latest_price" in df.columns and "EarningsPerShare" in df.columns:
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan  # PE undefined for negative earnings
        df["pen_vs_10yr"] = df["_latest_price"] / eps
    else:
        logger.debug("Missing columns for pen_vs_10yr")

    # ==================================================================
    # 3. PEGn — PE / forecast earnings growth
    # ==================================================================

    if "_latest_price" in df.columns and "EarningsPerShare" in df.columns:
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan
        pe = df["_latest_price"] / eps

        # Prefer 1-year forward growth (F1/F0); fall back to annualised
        # 2-year forward growth ((F2/F0)^0.5 − 1).
        growth_pct = pd.Series(np.nan, index=df.index)

        if "ForecastEarningsPerShare" in df.columns:
            f1_eps = df["ForecastEarningsPerShare"]
            growth_pct = _safe_ratio(f1_eps - df["EarningsPerShare"], eps) * 100
        elif "NextYearForecastEarningsPerShare" in df.columns:
            f2_eps = df["NextYearForecastEarningsPerShare"]
            ratio = _safe_ratio(f2_eps, eps)
            # Annualise 2-year growth: (ratio^0.5 − 1) × 100
            growth_pct = (np.power(ratio.clip(lower=0), 0.5) - 1) * 100

        # PEG meaningful only with positive growth above 1%.
        growth_pct[growth_pct <= 1.0] = np.nan
        df["pegn"] = pe / growth_pct
    else:
        logger.debug("Missing columns for pegn")

    # ==================================================================
    # 4. ADV liquidity — trailing average daily value traded
    # ==================================================================

    if has_prices and {"Volume"}.issubset(prices.columns) and code_col is not None:
        px = prices.copy()
        px["_daily_value"] = px["AdjustmentClose"] * px["Volume"]
        adv = px.groupby("Code")["_daily_value"].mean()
        df["adv_liquidity"] = code_col.map(adv)
    else:
        logger.debug("Missing data for adv_liquidity")

    # ==================================================================
    # 5. Broker target upside — external data only
    # ==================================================================
    # Cannot be derived from J-Quants.  Loaded from supplement sheet below.

    # ==================================================================
    # 6. Peer target upside
    # ==================================================================
    # Peer-implied price = sector median PE × company F2 EPS.
    # Upside = implied_price / current_price − 1.

    if (
        "_latest_price" in df.columns
        and "EarningsPerShare" in df.columns
        and "Sector33Code" in df.columns
    ):
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan
        pe = df["_latest_price"] / eps
        # Cap extreme PEs to prevent sector median distortion.
        pe_clipped = pe.clip(lower=0, upper=200)
        df["_pe_clipped"] = pe_clipped

        peer_median_pe = df.groupby("Sector33Code")["_pe_clipped"].transform("median")

        # Use F2 EPS (NextYearForecast) preferred; fall back to F1.
        f_eps: Optional[pd.Series] = None
        if "NextYearForecastEarningsPerShare" in df.columns:
            f_eps = df["NextYearForecastEarningsPerShare"]
        elif "ForecastEarningsPerShare" in df.columns:
            f_eps = df["ForecastEarningsPerShare"]

        if f_eps is not None:
            implied_price = peer_median_pe * f_eps
            df["peer_target_upside"] = _safe_ratio(
                implied_price - df["_latest_price"],
                df["_latest_price"],
            )

        df.drop(columns=["_pe_clipped"], errors="ignore", inplace=True)
    else:
        logger.debug("Missing columns for peer_target_upside")

    # ==================================================================
    # 7. Price 6mo vs TOPIX
    # ==================================================================
    # Stock 6-month return minus TOPIX 6-month return.

    if (
        topix is not None
        and not topix.empty
        and has_prices
        and code_col is not None
    ):
        px = prices.sort_values(["Code", "Date"])
        first_price = px.groupby("Code")["AdjustmentClose"].first()
        last_price = px.groupby("Code")["AdjustmentClose"].last()
        stock_return = _safe_ratio(last_price - first_price, first_price)

        # TOPIX return over the same window.
        topix_sorted = topix.sort_values("Date")
        tpx_return = np.nan
        if "Close" in topix_sorted.columns and len(topix_sorted) >= 2:
            tpx_first = topix_sorted["Close"].iloc[0]
            tpx_last = topix_sorted["Close"].iloc[-1]
            if tpx_first > 0:
                tpx_return = (tpx_last - tpx_first) / tpx_first

        if not np.isnan(tpx_return):
            df["price_6mo_vs_tpx"] = code_col.map(stock_return) - tpx_return
            logger.debug(
                "TOPIX 6mo return: %.2f%%; computed relative returns for %d stocks",
                tpx_return * 100,
                code_col.map(stock_return).notna().sum(),
            )
        else:
            logger.warning(
                "Could not compute TOPIX 6mo return; price_6mo_vs_tpx unavailable."
            )
    elif topix is None or (topix is not None and topix.empty):
        logger.debug("TOPIX index data not provided; skipping price_6mo_vs_tpx")

    # ==================================================================
    # 8. PARA target upside — proprietary
    # ==================================================================
    # Not computed here.  Loaded from supplement sheet below.

    # ==================================================================
    # Merge supplement data from PARA.FS.data Google Sheet
    # ==================================================================
    # The sheet is the authoritative source for broker_target_upside,
    # para_target_upside, and the true PBR/PE vs 10yr ratios.  It also
    # provides fallback values for the other metrics.

    supplement = load_gsheet_supplement()

    if not supplement.empty and code_col is not None:
        for metric_col in _SUPPLEMENT_METRIC_COLS:
            if metric_col not in supplement.columns:
                continue

            supp_values = code_col.map(supplement[metric_col])

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
    # Coverage summary
    # ==================================================================
    for col in _SUPPLEMENT_METRIC_COLS:
        if col in df.columns:
            coverage = df[col].notna().mean()
            logger.info(
                "Valuation  %-25s coverage: %5.1f%%", col, coverage * 100,
            )
        else:
            logger.warning("Valuation metric %s was not computed", col)

    # Clean up internal helper columns.
    df.drop(columns=["_latest_price"], errors="ignore", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_valuation(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Valuation category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    Args:
        df: DataFrame with computed valuation metric columns (output
            of :func:`compute_valuation_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of valuation category scores indexed by company.
    """
    metric_defs = load_metric_defs("valuation", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

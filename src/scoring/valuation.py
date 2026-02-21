"""Category 2: Valuation scoring.

Covers relative valuation vs history, liquidity, price momentum,
and target price upside metrics.

Combines J-Quants financial statements and daily bars with supplementary
data from the PARA.FS.data Google Sheet (via :class:`~src.data.gsheets.GoogleSheetsClient`)
for metrics not available via J-Quants alone (broker target upside,
PARA target upside, historical PBR/PE averages).

Google Sheet reference (PARA.FS.data):
    https://docs.google.com/spreadsheets/d/10mjEbmtJC6y5tCqnQ_SrUQAfheDhafAtpX0DjJJO5fk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.gsheets import GoogleSheetsClient
from src.scoring.utils import load_metric_defs, load_scoring_params, normalise_code, score_category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Sheets configuration
# ---------------------------------------------------------------------------
# GID for the valuation supplement tab in the PARA.FS.data spreadsheet.
_VALUATION_SHEET_GID: int = 0

# Column map: sheet column names -> internal metric names.
_COLUMN_MAP: dict[str, str] = {
    "PBR vs 10yr": "pbr_vs_10yr",
    "PBR_vs_10yr": "pbr_vs_10yr",
    "PEn vs 10yr": "pen_vs_10yr",
    "PEn_vs_10yr": "pen_vs_10yr",
    "PEGn": "pegn",
    "ADV Liquidity": "adv_liquidity",
    "ADV_Liquidity": "adv_liquidity",
    "Broker Target Upside": "broker_target_upside",
    "BrokerTP_Upside": "broker_target_upside",
    "Broker_Target_Upside": "broker_target_upside",
    "Peer Target Upside": "peer_target_upside",
    "Peer_Target_Upside": "peer_target_upside",
    "Price 6mo vs TPX": "price_6mo_vs_tpx",
    "Price_6mo_vs_TPX": "price_6mo_vs_tpx",
    "PARA Target Upside": "para_target_upside",
    "PARA_Target_Upside": "para_target_upside",
    "ParaTP_Upside": "para_target_upside",
}

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

def _load_valuation_supplement() -> pd.DataFrame:
    """Load valuation-related columns from the PARA.FS.data Google Sheet.

    Uses :class:`~src.data.gsheets.GoogleSheetsClient` with
    :data:`_COLUMN_MAP` to normalise sheet column names.

    Returns:
        DataFrame indexed by ``Code`` with supplement columns.
        Empty DataFrame on failure.
    """
    client = GoogleSheetsClient(column_map=_COLUMN_MAP)
    return client.read_sheet(gid=_VALUATION_SHEET_GID)


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

    * ``pbr_vs_10yr`` -- Current PBR (Price / BookValuePerShare).  Best-effort
      proxy; the Google Sheet provides the true ratio vs 10yr average when
      available.
    * ``pen_vs_10yr`` -- Current PE (Price / EarningsPerShare).  Same caveat
      as PBR; sheet overrides with the 10yr-relative value.
    * ``pegn`` -- PE / annualised forecast earnings growth (%).  Uses F1 EPS
      growth, falling back to annualised F2 growth.
    * ``adv_liquidity`` -- Trailing average daily traded value
      (AdjustmentClose * Volume).
    * ``peer_target_upside`` -- Sector median PE * company F2 EPS / current
      price - 1.  Peer-implied upside.
    * ``price_6mo_vs_tpx`` -- Stock 6-month return minus TOPIX 6-month
      return.  Requires daily bars and a TOPIX index DataFrame.

    **Supplement-only metrics** (loaded from PARA.FS.data via gsheets adapter):

    * ``broker_target_upside`` -- Consensus target upside.  Requires
      Bloomberg BEST / IBES; sourced via the supplement sheet.
    * ``para_target_upside`` -- Proprietary PARA model target upside.
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
    if "_latest_price" in df.columns and "BookValuePerShare" in df.columns:
        bvps = df["BookValuePerShare"].replace(0, np.nan)
        df["pbr_vs_10yr"] = df["_latest_price"] / bvps
    else:
        logger.debug("Missing columns for pbr_vs_10yr")

    # ==================================================================
    # 2. PEn vs 10yr average
    # ==================================================================
    if "_latest_price" in df.columns and "EarningsPerShare" in df.columns:
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan
        df["pen_vs_10yr"] = df["_latest_price"] / eps
    else:
        logger.debug("Missing columns for pen_vs_10yr")

    # ==================================================================
    # 3. PEGn -- PE / forecast earnings growth
    # ==================================================================
    if "_latest_price" in df.columns and "EarningsPerShare" in df.columns:
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan
        pe = df["_latest_price"] / eps

        growth_pct = pd.Series(np.nan, index=df.index)

        if "ForecastEarningsPerShare" in df.columns:
            f1_eps = df["ForecastEarningsPerShare"]
            growth_pct = _safe_ratio(f1_eps - df["EarningsPerShare"], eps) * 100
        elif "NextYearForecastEarningsPerShare" in df.columns:
            f2_eps = df["NextYearForecastEarningsPerShare"]
            ratio = _safe_ratio(f2_eps, eps)
            growth_pct = (np.power(ratio.clip(lower=0), 0.5) - 1) * 100

        growth_pct[growth_pct <= 1.0] = np.nan
        df["pegn"] = pe / growth_pct
    else:
        logger.debug("Missing columns for pegn")

    # ==================================================================
    # 4. ADV liquidity -- trailing average daily value traded
    # ==================================================================
    if has_prices and {"Volume"}.issubset(prices.columns) and code_col is not None:
        px = prices.copy()
        px["_daily_value"] = px["AdjustmentClose"] * px["Volume"]
        adv = px.groupby("Code")["_daily_value"].mean()
        df["adv_liquidity"] = code_col.map(adv)
    else:
        logger.debug("Missing data for adv_liquidity")

    # ==================================================================
    # 5. Broker target upside -- external data only
    # ==================================================================
    # Cannot be derived from J-Quants.  Loaded from supplement sheet below.

    # ==================================================================
    # 6. Peer target upside
    # ==================================================================
    if (
        "_latest_price" in df.columns
        and "EarningsPerShare" in df.columns
        and "Sector33Code" in df.columns
    ):
        eps = df["EarningsPerShare"].copy()
        eps[eps <= 0] = np.nan
        pe = df["_latest_price"] / eps
        pe_clipped = pe.clip(lower=0, upper=200)
        df["_pe_clipped"] = pe_clipped

        peer_median_pe = df.groupby("Sector33Code")["_pe_clipped"].transform("median")

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
    # 8. PARA target upside -- proprietary
    # ==================================================================
    # Not computed here.  Loaded from supplement sheet below.

    # ==================================================================
    # Merge supplement data from PARA.FS.data via gsheets adapter
    # ==================================================================
    # The sheet is the authoritative source for broker_target_upside,
    # para_target_upside, and the true PBR/PE vs 10yr ratios.  It also
    # provides fallback values for the other metrics.

    supplement = _load_valuation_supplement()

    # Normalise code_col to 4-digit base codes so it matches the
    # supplement index (Google Sheets uses 4-digit codes; J-Quants V2
    # uses 5-digit codes with a trailing check digit).
    code_col_norm: Optional[pd.Series] = None
    if code_col is not None:
        code_col_norm = code_col.map(normalise_code)

    # ── Diagnostic logging: before merge ──────────────────────────
    if not supplement.empty:
        logger.info(
            "Valuation supplement: %d rows, columns=%s",
            len(supplement), list(supplement.columns),
        )
        logger.info(
            "Valuation supplement index (first 5): %s",
            list(supplement.index[:5]),
        )
        if code_col is not None:
            logger.info(
                "Financials Code (first 5 raw): %s",
                list(code_col.head()),
            )
            logger.info(
                "Financials Code (first 5 normalised): %s",
                list(code_col_norm.head()),
            )
            matched = code_col_norm.isin(supplement.index).sum()
            logger.info(
                "Valuation supplement code match: %d / %d financials codes found in supplement index",
                matched, len(code_col_norm),
            )
    else:
        logger.warning("Valuation supplement is empty — no Sheets data available.")

    if not supplement.empty and code_col_norm is not None:
        for metric_col in _SUPPLEMENT_METRIC_COLS:
            if metric_col not in supplement.columns:
                continue

            supp_values = code_col_norm.map(supplement[metric_col])

            if metric_col in df.columns:
                before_nulls = df[metric_col].isna().sum()
                df[metric_col] = df[metric_col].fillna(
                    pd.Series(supp_values.values, index=df.index),
                )
                filled = before_nulls - df[metric_col].isna().sum()
            else:
                df[metric_col] = supp_values.values
                filled = int(pd.Series(supp_values.values).notna().sum())

            logger.info(
                "Valuation supplement %-20s: %d values filled, %d non-null after merge",
                metric_col, filled,
                int(df[metric_col].notna().sum()) if metric_col in df.columns else 0,
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

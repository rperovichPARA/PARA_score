"""Category 5: Value Impact (Kozo) scoring.

Core of Paradaim's alliance investing methodology.  Measures structural
improvement opportunities through balance-sheet efficiency, governance,
and engagement-related metrics.

Data sources
------------
**J-Quants** (computed directly):
    ROE (``Profit / Equity``) and payout ratio (``ResultPayoutRatioAnnual``).

**Google Sheets** (``PARA.FS.data`` via :class:`~src.data.gsheets.GoogleSheetsClient`):
    Board size, analyst coverage, excess cash / market cap, LTI / market
    cap, land / market cap, and raw cost of equity (Ke).

**Peer comparisons** (computed here):
    ``roe_vs_peer`` and ``ke_vs_peer`` compare company-level values against
    TOPIX ``Sector33Code`` group medians derived from J-Quants listed-info
    sector codes carried on the financials DataFrame.

**Unavailable** (set to NaN):
    ``underperf_segments`` and ``sga_vs_peer`` require segment-level data
    from EDINET / Bloomberg / Premium ``/fins/fs_details`` that is not yet
    integrated.

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
from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Sheets configuration
# ---------------------------------------------------------------------------
# GID for the kozo supplement tab in the PARA.FS.data spreadsheet.
# Default to the first tab (gid=0); update if kozo data moves to a
# dedicated tab.
_KOZO_SHEET_GID: int = 0

# Column map: **sheet** column names -> **internal** names.
# The GoogleSheetsClient applies this renaming before any metric-recognition
# logic.  Columns whose sheet names already match the YAML metric keys
# (e.g. ``board_size``) don't need entries here -- they are auto-recognised.
#
# Raw ``ke`` (cost of equity) is NOT a YAML metric key (the key is
# ``ke_vs_peer``), so it must be mapped here so it survives ``read_sheet``
# and is available for the peer-comparison step.
_COLUMN_MAP: dict[str, str] = {
    # Board size variants
    "Board Size": "board_size",
    "BoardSize": "board_size",
    # Analyst coverage variants
    "Analyst Coverage": "analyst_coverage",
    "AnalystCoverage": "analyst_coverage",
    # Balance-sheet / market-cap ratios
    "Excess Cash MktCap": "excess_cash_mktcap",
    "ExcessCash/MktCap": "excess_cash_mktcap",
    "Excess Cash / MktCap": "excess_cash_mktcap",
    "LTI MktCap": "lti_mktcap",
    "LTI/MktCap": "lti_mktcap",
    "LTI / MktCap": "lti_mktcap",
    "Land MktCap": "land_mktcap",
    "Land/MktCap": "land_mktcap",
    "Land / MktCap": "land_mktcap",
    # Cost of equity -- intermediate value, not a YAML metric key
    "Ke": "ke",
    "CostOfEquity": "ke",
    "Cost of Equity": "ke",
}

# Supplement columns pulled directly from the sheet that match YAML keys.
_DIRECT_METRIC_COLS: list[str] = [
    "excess_cash_mktcap",
    "lti_mktcap",
    "land_mktcap",
    "board_size",
    "analyst_coverage",
]

# All kozo YAML metric keys (for coverage reporting).
_ALL_METRICS: list[str] = [
    "underperf_segments",
    "sga_vs_peer",
    "roe_vs_peer",
    "excess_cash_mktcap",
    "lti_mktcap",
    "land_mktcap",
    "payout_ratio",
    "analyst_coverage",
    "ke_vs_peer",
    "board_size",
]


# ---------------------------------------------------------------------------
# Google Sheet loader
# ---------------------------------------------------------------------------

def _load_kozo_supplement() -> pd.DataFrame:
    """Load kozo-related columns from the PARA.FS.data Google Sheet.

    Uses :class:`~src.data.gsheets.GoogleSheetsClient` with
    :data:`_COLUMN_MAP` to normalise sheet column names.  Returns **all**
    columns (not just YAML-recognised metric columns) so that intermediate
    values like raw ``ke`` are available for peer-comparison calculations.

    Returns:
        DataFrame indexed by ``Code`` with supplement columns.  Empty
        DataFrame on failure.
    """
    client = GoogleSheetsClient(column_map=_COLUMN_MAP)
    return client.read_sheet(gid=_KOZO_SHEET_GID)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_kozo_metrics(financials: pd.DataFrame) -> pd.DataFrame:
    """Derive all 10 kozo (value impact) metrics.

    **J-Quants-derived metrics:**

    * ``roe_vs_peer`` -- Company ROE minus TOPIX sector median ROE.
      Reuses ``roe`` from the fundamentals computation when present;
      otherwise computes ``Profit / Equity``.
    * ``payout_ratio`` -- ``ResultPayoutRatioAnnual`` from J-Quants
      financial statements (percentage form).

    **Google Sheets metrics (PARA.FS.data):**

    * ``excess_cash_mktcap`` -- Excess cash / market capitalisation.
    * ``lti_mktcap`` -- Long-term investments / market capitalisation.
    * ``land_mktcap`` -- Land holdings / market capitalisation.
    * ``board_size`` -- Number of board members.
    * ``analyst_coverage`` -- Number of covering analysts.
    * ``ke_vs_peer`` -- Company cost of equity minus sector median Ke.
      Raw Ke is loaded from the sheet, then compared against the
      ``Sector33Code`` group median.

    **Unavailable (NaN):**

    * ``underperf_segments`` -- Requires EDINET / Bloomberg segment data.
    * ``sga_vs_peer`` -- Requires Premium ``/fins/fs_details`` for SGA
      breakdown.

    Args:
        financials: DataFrame with J-Quants financial fields.  Expected
            to contain ``Profit``, ``Equity``, ``ResultPayoutRatioAnnual``,
            ``Sector33Code``, and ``Code``.  If ``roe`` is already present
            (e.g. from :func:`~src.scoring.fundamentals.compute_fundamentals_metrics`),
            it will be reused rather than recomputed.

    Returns:
        Copy of *financials* with the 10 kozo metric columns added.
    """
    df = financials.copy()

    code_col: Optional[pd.Series] = None
    if "Code" in df.columns:
        code_col = df["Code"].astype(str).str.strip()

    # ==================================================================
    # 1. ROE -- reuse from fundamentals if already present
    # ==================================================================
    if "roe" not in df.columns:
        if {"Profit", "Equity"}.issubset(df.columns):
            denom = df["Equity"].replace(0, np.nan)
            df["roe"] = df["Profit"] / denom
        else:
            logger.debug("Missing Profit/Equity columns for ROE computation")

    # ==================================================================
    # 2. Payout ratio -- from J-Quants ResultPayoutRatioAnnual
    # ==================================================================
    if "ResultPayoutRatioAnnual" in df.columns:
        df["payout_ratio"] = pd.to_numeric(
            df["ResultPayoutRatioAnnual"], errors="coerce",
        )
    else:
        logger.debug("Missing ResultPayoutRatioAnnual for payout_ratio")

    # ==================================================================
    # 3. Load supplement data from Google Sheets
    # ==================================================================
    supplement = _load_kozo_supplement()

    if not supplement.empty and code_col is not None:
        # -- Direct metrics: merge sheet columns into df ---------------
        for metric_col in _DIRECT_METRIC_COLS:
            if metric_col not in supplement.columns:
                continue

            supp_values = code_col.map(supplement[metric_col])

            if metric_col in df.columns:
                # Fill only where the existing value is missing.
                before_nulls = df[metric_col].isna().sum()
                df[metric_col] = df[metric_col].fillna(
                    pd.Series(supp_values.values, index=df.index),
                )
                filled = before_nulls - df[metric_col].isna().sum()
            else:
                df[metric_col] = supp_values.values
                filled = int(pd.Series(supp_values.values).notna().sum())

            if filled > 0:
                logger.info(
                    "Kozo supplement filled %d values for %s",
                    filled, metric_col,
                )

        # -- Raw Ke for peer comparison --------------------------------
        if "ke" in supplement.columns:
            raw_ke = code_col.map(supplement["ke"])
            df["_ke_raw"] = pd.to_numeric(
                pd.Series(raw_ke.values, index=df.index), errors="coerce",
            )
            logger.info(
                "Loaded raw Ke for %d companies",
                df["_ke_raw"].notna().sum(),
            )

    # ==================================================================
    # 4. ROE vs peer avg -- ROE minus TOPIX sector median
    # ==================================================================
    if "roe" in df.columns and "Sector33Code" in df.columns:
        sector_median_roe = (
            df.groupby("Sector33Code")["roe"].transform("median")
        )
        df["roe_vs_peer"] = df["roe"] - sector_median_roe
    else:
        logger.debug(
            "Cannot compute roe_vs_peer -- missing roe or Sector33Code"
        )

    # ==================================================================
    # 5. Ke vs peer avg -- Ke minus TOPIX sector median
    # ==================================================================
    if "_ke_raw" in df.columns and "Sector33Code" in df.columns:
        sector_median_ke = (
            df.groupby("Sector33Code")["_ke_raw"].transform("median")
        )
        df["ke_vs_peer"] = df["_ke_raw"] - sector_median_ke
    else:
        logger.debug(
            "Cannot compute ke_vs_peer -- missing Ke data or Sector33Code"
        )

    # ==================================================================
    # 6. Unavailable metrics -- NaN placeholders
    # ==================================================================
    # These require segment-level data (EDINET / Bloomberg / Premium
    # /fins/fs_details) that is not yet integrated into the pipeline.
    df["underperf_segments"] = np.nan
    df["sga_vs_peer"] = np.nan

    # ==================================================================
    # 7. Coverage summary
    # ==================================================================
    for col in _ALL_METRICS:
        if col in df.columns:
            coverage = df[col].notna().mean()
            logger.info(
                "Kozo    %-25s coverage: %5.1f%%", col, coverage * 100,
            )
        else:
            logger.warning("Kozo metric %s was not computed", col)

    # Clean up internal helper columns.
    df.drop(columns=["_ke_raw"], errors="ignore", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_kozo(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Kozo (Value Impact) category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    Args:
        df: DataFrame with computed kozo metric columns (output of
            :func:`compute_kozo_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of kozo category scores indexed by company.
    """
    metric_defs = load_metric_defs("kozo", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

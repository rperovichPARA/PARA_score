"""Main pipeline orchestrator.

Coordinates: config loading -> J-Quants data pull -> metric computation ->
category scoring -> composite scoring -> ranked output.

CLI usage::

    python -m src.pipeline --universe 72030,86970 --output-dir ./output
    python -m src.pipeline --universe all --output-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data.jquants import JQuantsClient
from src.scoring.composite import CATEGORY_NAMES, compute_composite_scores
from src.scoring.factors import compute_factor_metrics
from src.scoring.fundamentals import compute_fundamentals_metrics
from src.scoring.kozo import compute_kozo_metrics
from src.scoring.sector import compute_sector_metrics
from src.scoring.valuation import compute_valuation_metrics

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "scoring_weights.yaml"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_pipeline_config(config_path: str) -> dict:
    """Load the scoring weights YAML configuration.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed YAML as a dictionary.
    """
    with open(config_path, "r") as fh:
        cfg: dict = yaml.safe_load(fh)
    logger.info("Pipeline config loaded from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------

def _filter_universe(
    listed: pd.DataFrame,
    config: dict,
    codes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter listed companies by market codes and/or explicit security codes.

    Args:
        listed: Full listed-company DataFrame from J-Quants.
        config: Parsed YAML config (expects ``universe.market_codes``).
        codes: Optional explicit list of security codes to keep.  When
            provided, only these codes are retained (market-code filtering
            is still applied first).

    Returns:
        Filtered DataFrame.
    """
    df = listed.copy()

    # --- Market-code filter from config ---
    market_codes: list[str] = (
        config.get("universe", {}).get("market_codes") or []
    )
    if market_codes and "MarketCode" in df.columns:
        before = len(df)
        df = df[df["MarketCode"].isin(market_codes)]
        logger.info(
            "Market-code filter %s: %d -> %d companies",
            market_codes, before, len(df),
        )
    elif market_codes:
        logger.warning(
            "MarketCode column missing; skipping market-code filter."
        )

    # --- Explicit security-code filter ---
    if codes and "Code" in df.columns:
        code_set = {c.strip() for c in codes}
        before = len(df)
        df = df[df["Code"].astype(str).str.strip().isin(code_set)]
        logger.info(
            "Security-code filter (%d requested): %d -> %d companies",
            len(code_set), before, len(df),
        )
        missing = code_set - set(df["Code"].astype(str).str.strip())
        if missing:
            logger.warning("Requested codes not found in universe: %s", missing)

    if df.empty:
        logger.error("Universe is empty after filtering — nothing to score.")

    return df


def _deduplicate_financials(financials: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent financial statement per security code.

    J-Quants ``/fins/statements`` may return multiple disclosures per
    company (quarterly, annual, revisions).  We keep the row with the
    latest ``DisclosedDate`` for each ``Code``.

    Args:
        financials: Raw financial-statements DataFrame.

    Returns:
        Deduplicated DataFrame with one row per company.
    """
    if financials.empty:
        return financials

    if "DisclosedDate" in financials.columns:
        financials = financials.copy()
        financials["DisclosedDate"] = pd.to_datetime(
            financials["DisclosedDate"], errors="coerce"
        )
        financials.sort_values("DisclosedDate", inplace=True)

    if "Code" in financials.columns:
        financials = financials.drop_duplicates(subset="Code", keep="last")
        financials = financials.set_index(
            financials["Code"].astype(str).str.strip(), drop=False
        )

    logger.info("Financials deduplicated: %d unique companies", len(financials))
    return financials


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_excel(
    results: pd.DataFrame,
    category_dfs: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Write scoring results to an Excel workbook with per-category sheets.

    Sheets:
        * **Composite** — Main scores and rankings.
        * **Fundamentals** — Raw fundamentals metrics.
        * **Valuation** — Raw valuation metrics.
        * **Sector** — Raw sector metrics.
        * **Factors** — Raw factor/theme metrics.
        * **Kozo** — Raw kozo (value impact) metrics.

    Args:
        results: Composite scores DataFrame (output of
            :func:`~src.scoring.composite.compute_composite_scores`).
        category_dfs: Dict of category name -> metric DataFrame.
        output_path: Full path for the ``.xlsx`` file.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Composite", index=False)

        for cat_name in CATEGORY_NAMES:
            if cat_name not in category_dfs:
                continue
            sheet_name = cat_name.capitalize()
            cat_df = category_dfs[cat_name]
            cat_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Excel workbook written to %s", output_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    universe_codes: Optional[list[str]] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = "",
) -> pd.DataFrame:
    """Run the full PARA scoring pipeline.

    1. Load configuration and initialise J-Quants client.
    2. Pull listed companies and filter to the scoring universe.
    3. Fetch financial statements and daily prices.
    4. Compute metrics for all five scoring categories.
    5. Pass category DataFrames to composite scorer for VI / SP.
    6. Write ranked results to ``output_dir/scores.csv`` and
       ``output_dir/scores.xlsx`` (with per-category sheets).

    Args:
        universe_codes: Explicit security codes to score.  ``None`` or
            empty list means score the full universe (subject to config
            market-code filtering).
        config_path: Path to ``scoring_weights.yaml``.
        output_dir: Directory for output files.  Falls back to
            ``OUTPUT_DIR`` env var, then ``./output``.

    Returns:
        DataFrame with composite scores and rankings.
    """
    output_dir = output_dir or os.getenv("OUTPUT_DIR", "./output")
    config = load_pipeline_config(config_path)

    # ── 1. Initialise J-Quants client ─────────────────────────────────
    logger.info("Initialising J-Quants client...")
    client = JQuantsClient()

    # ── 2. Build scoring universe ─────────────────────────────────────
    logger.info("Fetching listed companies...")
    listed = client.get_listed_companies()
    logger.info("Full listed universe: %d companies", len(listed))

    universe = _filter_universe(listed, config, codes=universe_codes)
    if universe.empty:
        logger.error("No companies in universe; aborting.")
        return pd.DataFrame()

    logger.info("Scoring universe: %d companies", len(universe))
    universe_code_list = universe["Code"].astype(str).str.strip().tolist()

    # ── 3. Fetch financial statements and prices ──────────────────────
    logger.info("Fetching financial statements...")
    financials = client.get_financial_statements()
    logger.info("Raw financial statements: %d rows", len(financials))

    # Restrict to universe and deduplicate
    if "Code" in financials.columns:
        financials = financials[
            financials["Code"].astype(str).str.strip().isin(universe_code_list)
        ]
    financials = _deduplicate_financials(financials)
    logger.info("Financials for universe: %d companies", len(financials))

    # Carry Sector33Code from listed-company data into financials for
    # kozo peer-group calculations.
    if "Sector33Code" not in financials.columns and "Sector33Code" in universe.columns:
        sector_map = (
            universe.set_index(universe["Code"].astype(str).str.strip())["Sector33Code"]
        )
        financials["Sector33Code"] = (
            financials["Code"].astype(str).str.strip().map(sector_map)
        )

    # Fetch recent daily quotes for valuation / liquidity metrics.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    six_months_ago = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")
    logger.info("Fetching daily quotes (%s to %s)...", six_months_ago, today)
    prices = client.get_daily_quotes(date_from=six_months_ago, date_to=today)

    if not prices.empty and "Code" in prices.columns:
        prices = prices[
            prices["Code"].astype(str).str.strip().isin(universe_code_list)
        ]
    logger.info("Daily quotes for universe: %d rows", len(prices))

    # Fetch TOPIX index for the same window (needed for price_6mo_vs_tpx).
    logger.info("Fetching TOPIX index (%s to %s)...", six_months_ago, today)
    topix = client.get_topix_index(date_from=six_months_ago, date_to=today)
    logger.info("TOPIX index: %d rows", len(topix))

    # ── 4. Compute metrics ────────────────────────────────────────────
    logger.info("Computing category metrics...")
    fundamentals_df = compute_fundamentals_metrics(financials, prices)
    valuation_df = compute_valuation_metrics(financials, prices, topix=topix)
    sector_df = compute_sector_metrics(financials)
    factors_df = compute_factor_metrics(financials)
    kozo_df = compute_kozo_metrics(fundamentals_df)

    category_dfs: dict[str, pd.DataFrame] = {
        "fundamentals": fundamentals_df,
        "valuation": valuation_df,
        "sector": sector_df,
        "factors": factors_df,
        "kozo": kozo_df,
    }

    # ── 5. Composite scoring ──────────────────────────────────────────
    logger.info("Computing composite VI / SP scores...")
    results = compute_composite_scores(category_dfs, config_path)

    # Attach company identifiers for readability.
    for info_col in ("CompanyName", "CompanyNameEnglish", "Sector33Code"):
        if info_col in universe.columns:
            col_map = universe.set_index(
                universe["Code"].astype(str).str.strip()
            )[info_col]
            if "Code" in results.columns:
                results[info_col] = (
                    results["Code"].astype(str).str.strip().map(col_map)
                )

    # Reorder so identifiers come first.
    id_cols = [c for c in ("Code", "CompanyName", "CompanyNameEnglish", "Sector33Code")
               if c in results.columns]
    score_cols = [c for c in results.columns if c not in id_cols]
    results = results[id_cols + score_cols]

    # ── 6. Write output ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "scores.csv")
    results.to_csv(csv_path, index=False)
    logger.info("Scores written to %s  (%d companies)", csv_path, len(results))

    xlsx_path = os.path.join(output_dir, "scores.xlsx")
    _write_excel(results, category_dfs, xlsx_path)

    # Summary for user.
    if "VI_rank" in results.columns:
        top_vi = results.nsmallest(5, "VI_rank")
        logger.info("Top 5 by VI rank:\n%s", top_vi.to_string(index=False))
    if "SP_rank" in results.columns:
        top_sp = results.nsmallest(5, "SP_rank")
        logger.info("Top 5 by SP rank:\n%s", top_sp.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="PARA Score — Paradaim Capital stock scoring pipeline",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="all",
        help=(
            "Comma-separated security codes to score, or 'all' for the full "
            "universe (filtered by market codes in the config).  "
            "Example: --universe 72030,86970,99840"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Directory for output files (scores.csv, scores.xlsx).  Defaults to the "
            "OUTPUT_DIR environment variable, then ./output."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to scoring_weights.yaml config file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse universe codes.
    codes: Optional[list[str]] = None
    if args.universe.lower() != "all":
        codes = [c.strip() for c in args.universe.split(",") if c.strip()]

    results = run_pipeline(
        universe_codes=codes,
        config_path=args.config,
        output_dir=args.output_dir,
    )

    if results.empty:
        logger.error("Pipeline produced no results.")
        sys.exit(1)

    print(f"Scored {len(results)} companies. Output: {args.output_dir or os.getenv('OUTPUT_DIR', './output')}/scores.csv")


if __name__ == "__main__":
    main()

"""Main pipeline orchestrator.

Coordinates: config loading -> J-Quants data pull -> metric computation ->
category scoring -> composite scoring -> ranked output.

Data fetching is cached to disk (parquet/JSON) to avoid redundant API calls.
The scoring step always runs from the cached data.

CLI usage::

    python -m src.pipeline --universe all --output-dir ./output
    python -m src.pipeline --universe 7203,8697 --output-dir ./output
    python -m src.pipeline --force-refresh          # bypass all caches
    python -m src.pipeline --update-only             # only score new listings
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

from src.data.cache import DataCache
from src.data.jquants import JQuantsClient
from src.scoring.composite import CATEGORY_NAMES, compute_composite_scores
from src.scoring.factors import compute_factor_metrics
from src.scoring.fundamentals import compute_fundamentals_metrics
from src.scoring.kozo import compute_kozo_metrics
from src.scoring.sector import compute_sector_metrics, fetch_sector_signals
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
# Code normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_code(code: str) -> str:
    """Return the 4-digit base security code.

    J-Quants V2 returns 5-digit codes where the 5th character is a check
    digit (e.g. ``"72030"`` for Toyota 7203).  Users typically pass the
    4-digit form.  We strip the trailing check digit so both formats can
    be compared consistently.
    """
    code = code.strip()
    if len(code) == 5 and code.isdigit():
        return code[:4]
    return code


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
            is still applied first).  Accepts both 4-digit (``"7203"``)
            and 5-digit (``"72030"``) formats.

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
    # V2 returns 5-digit codes; users may pass 4- or 5-digit.  Normalise
    # both sides to the 4-digit base code for matching.
    if codes and "Code" in df.columns:
        code_set = {_normalise_code(c) for c in codes}
        before = len(df)
        df_code_norm = df["Code"].astype(str).map(_normalise_code)
        df = df[df_code_norm.isin(code_set)]
        logger.info(
            "Security-code filter (%d requested): %d -> %d companies",
            len(code_set), before, len(df),
        )
        matched = set(df["Code"].astype(str).map(_normalise_code))
        missing = code_set - matched
        if missing:
            logger.warning("Requested codes not found in universe: %s", missing)

    if df.empty:
        logger.error("Universe is empty after filtering — nothing to score.")

    return df


def _deduplicate_financials(financials: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent financial statement per security code.

    J-Quants ``/fins/summary`` may return multiple disclosures per
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
# Cached data fetching
# ---------------------------------------------------------------------------

def _fetch_listed_companies(
    client: JQuantsClient,
    cache: DataCache,
) -> pd.DataFrame:
    """Fetch listed companies, using cache when fresh.

    Cache TTL: 7 days (listed companies change infrequently).
    """
    cached = cache.load_parquet("listed_companies")
    if cached is not None:
        return cached

    logger.info("Fetching listed companies from J-Quants API...")
    listed = client.get_listed_companies()
    cache.save_parquet("listed_companies", listed)
    return listed


def _fetch_financials(
    client: JQuantsClient,
    cache: DataCache,
    universe_code_list: list[str],
) -> pd.DataFrame:
    """Fetch financial statements, using cache when fresh.

    Cache TTL: 7 days (statements are quarterly).
    """
    cached = cache.load_parquet("financials")
    if cached is not None:
        return cached

    logger.info("Fetching financial statements for %d codes...", len(universe_code_list))
    financials = client.get_financial_statements_bulk(universe_code_list)
    if not financials.empty:
        cache.save_parquet("financials", financials)
    return financials


def _fetch_daily_quotes(
    client: JQuantsClient,
    cache: DataCache,
    universe_code_list: list[str],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """Fetch daily quotes with incremental caching.

    On re-fetch (cache older than 1 day), only pull quotes since the last
    cached date and append them to the existing cache.
    """
    if not cache.is_stale("daily_quotes"):
        cached = cache.load_parquet("daily_quotes")
        if cached is not None:
            return cached

    # Check if we can do an incremental fetch.
    last_cached_date = cache.get_last_date("daily_quotes", date_col="Date")

    if last_cached_date and not cache.force_refresh:
        # Incremental: fetch only from the day after the last cached date.
        incremental_from = (
            datetime.strptime(last_cached_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        if incremental_from > date_to:
            # Cache already covers the requested range.
            logger.info("Daily quotes cache already covers through %s.", last_cached_date)
            cached = cache.load_parquet("daily_quotes", max_age_days=365)
            if cached is not None:
                return cached

        logger.info(
            "Incremental daily quotes fetch: %s to %s for %d codes...",
            incremental_from, date_to, len(universe_code_list),
        )
        new_quotes = client.get_daily_quotes_bulk(
            codes=universe_code_list,
            date_from=incremental_from,
            date_to=date_to,
        )
        if not new_quotes.empty:
            combined = cache.append_parquet("daily_quotes", new_quotes, date_col="Date")
            # Filter to the requested window.
            if "Date" in combined.columns:
                combined["Date"] = pd.to_datetime(combined["Date"])
                combined = combined[
                    (combined["Date"] >= pd.Timestamp(date_from))
                    & (combined["Date"] <= pd.Timestamp(date_to))
                ]
            return combined
        else:
            # No new data; return existing cache.
            cached = cache.load_parquet("daily_quotes", max_age_days=365)
            if cached is not None:
                return cached
    else:
        # Full fetch.
        logger.info(
            "Full daily quotes fetch: %s to %s for %d codes...",
            date_from, date_to, len(universe_code_list),
        )

    prices = client.get_daily_quotes_bulk(
        codes=universe_code_list,
        date_from=date_from,
        date_to=date_to,
    )
    if not prices.empty:
        cache.save_parquet("daily_quotes", prices)
    return prices


def _fetch_topix(
    client: JQuantsClient,
    cache: DataCache,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """Fetch TOPIX index, stored inside the daily_quotes cache cycle.

    TOPIX is small and fast to fetch, so it follows the daily_quotes TTL.
    """
    cached = cache.load_parquet("topix", max_age_days=1)
    if cached is not None:
        return cached

    logger.info("Fetching TOPIX index (%s to %s)...", date_from, date_to)
    topix = client.get_topix_index(date_from=date_from, date_to=date_to)
    if not topix.empty:
        cache.save_parquet("topix", topix)
    return topix


def _fetch_sector_signals(cache: DataCache) -> dict[str, float]:
    """Fetch sector signals from Render, using cache when fresh.

    Cache TTL: 7 days.
    """
    cached = cache.load_json("sector_signals")
    if cached is not None:
        return cached

    logger.info("Fetching sector signals from Render...")
    signals = fetch_sector_signals()
    if signals:
        cache.save_json("sector_signals", signals)
    return signals


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    universe_codes: Optional[list[str]] = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = "",
    force_refresh: bool = False,
    update_only: bool = False,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full PARA scoring pipeline.

    1. Load configuration and initialise J-Quants client.
    2. Pull listed companies and filter to the scoring universe.
    3. Fetch financial statements and daily prices (cached).
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
        force_refresh: If ``True``, bypass all data caches and re-fetch
            everything from the API.
        update_only: If ``True``, skip stocks that already have scores
            in the existing output file — only score new additions.
        cache_dir: Override cache directory path.

    Returns:
        DataFrame with composite scores and rankings.
    """
    output_dir = output_dir or os.getenv("OUTPUT_DIR", "./output")
    config = load_pipeline_config(config_path)

    # ── 0. Initialise cache ───────────────────────────────────────────
    cache = DataCache(cache_dir=cache_dir, force_refresh=force_refresh)

    # ── 1. Initialise J-Quants client ─────────────────────────────────
    logger.info("Initialising J-Quants client...")
    client = JQuantsClient()

    # ── 2. Build scoring universe ─────────────────────────────────────
    listed = _fetch_listed_companies(client, cache)
    logger.info("Full listed universe: %d companies", len(listed))

    universe = _filter_universe(listed, config, codes=universe_codes)
    if universe.empty:
        logger.error("No companies in universe; aborting.")
        return pd.DataFrame()

    logger.info("Scoring universe: %d companies", len(universe))
    universe_code_list = universe["Code"].astype(str).str.strip().tolist()

    # ── 2b. Handle --update-only ──────────────────────────────────────
    existing_codes: set[str] = set()
    if update_only:
        csv_path = os.path.join(output_dir, "scores.csv")
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            if "Code" in existing_df.columns:
                existing_codes = set(
                    existing_df["Code"].astype(str).str.strip().map(_normalise_code)
                )
                # Filter universe to only new codes.
                new_mask = (
                    universe["Code"]
                    .astype(str)
                    .str.strip()
                    .map(_normalise_code)
                    .apply(lambda c: c not in existing_codes)
                )
                new_universe = universe[new_mask]

                if new_universe.empty:
                    logger.info(
                        "update-only: all %d companies already scored. Nothing to do.",
                        len(universe),
                    )
                    return existing_df

                logger.info(
                    "update-only: %d new companies to score (skipping %d already scored).",
                    len(new_universe),
                    len(universe) - len(new_universe),
                )
                universe = new_universe
                universe_code_list = universe["Code"].astype(str).str.strip().tolist()
        else:
            logger.info(
                "update-only: no existing scores at %s — scoring full universe.",
                csv_path,
            )

    # ── 3. Fetch financial statements and prices (cached) ─────────────
    financials = _fetch_financials(client, cache, universe_code_list)
    logger.info("Raw financial statements: %d rows", len(financials))

    # Deduplicate (keep most recent disclosure per code)
    financials = _deduplicate_financials(financials)
    logger.info("Financials for universe: %d companies", len(financials))

    # Carry sector codes from listed-company data into financials for
    # kozo peer-group calculations (Sector33Code) and sector
    # attractiveness scoring (Sector17Code / Sector17CodeName).
    code_index = universe.set_index(universe["Code"].astype(str).str.strip())
    fin_codes = financials["Code"].astype(str).str.strip()
    for sector_col in ("Sector33Code", "Sector17Code", "Sector17CodeName"):
        if sector_col not in financials.columns and sector_col in universe.columns:
            financials[sector_col] = fin_codes.map(code_index[sector_col]).values

    # Fetch recent daily quotes for valuation / liquidity metrics.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    six_months_ago = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")

    prices = _fetch_daily_quotes(
        client, cache, universe_code_list,
        date_from=six_months_ago, date_to=today,
    )
    logger.info("Daily quotes for universe: %d rows", len(prices))

    # Fetch TOPIX index for the same window (needed for price_6mo_vs_tpx).
    topix = _fetch_topix(client, cache, date_from=six_months_ago, date_to=today)
    logger.info("TOPIX index: %d rows", len(topix))

    # ── 4. Compute metrics ────────────────────────────────────────────
    logger.info("Computing category metrics...")
    fundamentals_df = compute_fundamentals_metrics(financials, prices)
    valuation_df = compute_valuation_metrics(financials, prices, topix=topix)

    sector_signals = _fetch_sector_signals(cache)
    sector_df = compute_sector_metrics(financials, sector_signals=sector_signals)

    factors_df = compute_factor_metrics(financials, prices, topix=topix)
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

    # ── 5b. Merge with existing scores if --update-only ───────────────
    if update_only and existing_codes:
        csv_path = os.path.join(output_dir, "scores.csv")
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            results = pd.concat([existing_df, results], ignore_index=True)
            # Re-rank across the merged set.
            if "VI_score" in results.columns:
                results["VI_rank"] = results["VI_score"].rank(
                    ascending=False, method="min"
                ).astype(int)
            if "SP_score" in results.columns:
                results["SP_rank"] = results["SP_score"].rank(
                    ascending=False, method="min"
                ).astype(int)
            logger.info(
                "update-only: merged new scores with %d existing -> %d total.",
                len(existing_df), len(results),
            )

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
            "universe (filtered by market codes in the config).  Accepts both "
            "4-digit (7203) and 5-digit V2 (72030) formats.  "
            "Example: --universe 7203,6758,8306"
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
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help=(
            "Bypass all data caches and re-fetch everything from the API. "
            "The scoring step still runs normally."
        ),
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        default=False,
        help=(
            "Skip stocks that already have scores in the output file. "
            "Only score new additions to the universe. Useful when a few "
            "new listings appear between full scoring runs."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help=(
            "Directory for cached data files. Defaults to DATA_CACHE_DIR "
            "environment variable, then ./cache."
        ),
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
        force_refresh=args.force_refresh,
        update_only=args.update_only,
        cache_dir=args.cache_dir or None,
    )

    if results.empty:
        logger.error("Pipeline produced no results.")
        sys.exit(1)

    print(f"Scored {len(results)} companies. Output: {args.output_dir or os.getenv('OUTPUT_DIR', './output')}/scores.csv")


if __name__ == "__main__":
    main()

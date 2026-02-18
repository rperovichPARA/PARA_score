"""Main pipeline orchestrator.

Coordinates: data loading -> metric computation -> category scoring ->
composite scoring -> output generation.
"""

import logging
import os

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data.jquants import JQuantsClient
from src.scoring.composite import compute_composite_scores
from src.scoring.factors import compute_factor_metrics, score_factors
from src.scoring.fundamentals import compute_fundamentals_metrics, score_fundamentals
from src.scoring.kozo import compute_kozo_metrics, score_kozo
from src.scoring.sector import compute_sector_metrics, score_sector
from src.scoring.valuation import compute_valuation_metrics, score_valuation

load_dotenv()
logger = logging.getLogger(__name__)


def _filter_universe(
    universe: pd.DataFrame, config_path: str
) -> pd.DataFrame:
    """Filter the universe based on market codes in the config.

    Parameters
    ----------
    universe : pd.DataFrame
        Full listed-company DataFrame from J-Quants (must contain
        ``MarketCode`` column).
    config_path : str
        Path to scoring weights YAML.

    Returns
    -------
    pd.DataFrame
        Filtered subset.  Returns ``universe`` unchanged when no
        ``market_codes`` are configured (or the list is empty).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    market_codes: list[str] = (
        cfg.get("universe", {}).get("market_codes") or []
    )
    if not market_codes:
        return universe

    if "MarketCode" not in universe.columns:
        logger.warning(
            "MarketCode column not found in listed-company data; "
            "skipping universe filter."
        )
        return universe

    filtered = universe[universe["MarketCode"].isin(market_codes)].copy()
    logger.info(
        "Universe filtered by MarketCode %s: %d -> %d companies",
        market_codes, len(universe), len(filtered),
    )
    return filtered


def run_pipeline(
    config_path: str = "config/scoring_weights.yaml",
    output_dir: str = "",
) -> pd.DataFrame:
    """Run the full PARA scoring pipeline.

    Args:
        config_path: Path to scoring weights YAML.
        output_dir: Directory for output files. Defaults to OUTPUT_DIR env var.

    Returns:
        DataFrame with composite scores and rankings.
    """
    output_dir = output_dir or os.getenv("OUTPUT_DIR", "./output")

    # --- Data Loading ---
    logger.info("Loading data from J-Quants...")
    client = JQuantsClient()
    universe = client.get_listed_companies()
    logger.info("Full universe: %d companies", len(universe))

    universe = _filter_universe(universe, config_path)
    logger.info("Scoring universe: %d companies", len(universe))

    # --- Metric Computation ---
    logger.info("Computing metrics...")
    fundamentals_df = compute_fundamentals_metrics(universe)
    valuation_df = compute_valuation_metrics(universe, pd.DataFrame())
    sector_df = compute_sector_metrics(universe)
    factors_df = compute_factor_metrics(universe)
    kozo_df = compute_kozo_metrics(fundamentals_df)

    # --- Category Scoring ---
    logger.info("Scoring categories...")
    category_scores = {
        "fundamentals": score_fundamentals(fundamentals_df, config_path),
        "valuation": score_valuation(valuation_df, config_path),
        "sector": score_sector(sector_df, config_path),
        "factors": score_factors(factors_df, config_path),
        "kozo": score_kozo(kozo_df, config_path),
    }

    # --- Composite Scoring ---
    logger.info("Computing composite scores...")
    results = compute_composite_scores(category_scores, config_path)

    # --- Output ---
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scores.csv")
    results.to_csv(csv_path)
    logger.info("Scores written to %s", csv_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_pipeline()

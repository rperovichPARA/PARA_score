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
    logger.info("Universe: %d companies", len(universe))

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

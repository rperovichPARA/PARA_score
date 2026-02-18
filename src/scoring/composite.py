"""Composite VI + SP scoring and ranking.

Combines the five category scores using category-level weights
to produce final Value Impact and Stock Pick composite scores.
"""

import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_composite_scores(
    category_scores: dict[str, pd.Series],
    config_path: str = "config/scoring_weights.yaml",
) -> pd.DataFrame:
    """Compute VI and SP composite scores from category scores.

    Args:
        category_scores: Dict mapping category name -> Series of scores.
            Expected keys: fundamentals, valuation, sector, factors, kozo.
        config_path: Path to scoring weights YAML.

    Returns:
        DataFrame with columns: VI_score, SP_score, VI_rank, SP_rank.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cat_weights = config["category_weights"]
    result = pd.DataFrame(index=category_scores.get("fundamentals", pd.Series()).index)

    for composite_name in ("VI", "SP"):
        weights = cat_weights[composite_name]
        total = pd.Series(0.0, index=result.index)
        total_weight = 0.0

        for cat_name, weight in weights.items():
            if weight == 0.0:
                continue
            if cat_name not in category_scores:
                logger.warning("Missing category %s for %s composite", cat_name, composite_name)
                continue
            total += category_scores[cat_name] * weight
            total_weight += weight

        if 0 < total_weight < 1:
            total /= total_weight

        result[f"{composite_name}_score"] = total
        result[f"{composite_name}_rank"] = total.rank(ascending=False, method="min")

    return result

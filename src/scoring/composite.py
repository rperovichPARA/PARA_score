"""Composite VI + SP scoring and ranking.

Takes the five category metric DataFrames, scores each using
:func:`~src.scoring.utils.score_category_from_config`, then combines
category scores with weights from ``scoring_weights.yaml`` to produce
final Value Impact (VI) and Stock Pick (SP) composite scores.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.scoring.utils import load_config, score_category_from_config

logger = logging.getLogger(__name__)

CATEGORY_NAMES: list[str] = [
    "fundamentals",
    "valuation",
    "sector",
    "factors",
    "kozo",
]


def compute_composite_scores(
    category_dfs: dict[str, pd.DataFrame],
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute VI and SP composite scores from category metric DataFrames.

    1. Scores each category via :func:`score_category_from_config`.
    2. Weights category scores using ``category_weights`` from config.
    3. Ranks companies in descending order (rank 1 = best).

    Args:
        category_dfs: Dict mapping category name to a DataFrame of raw
            metric columns (one row per company).  Expected keys:
            ``fundamentals``, ``valuation``, ``sector``, ``factors``,
            ``kozo``.
        config_path: Path to ``scoring_weights.yaml``.  Defaults to
            ``<repo_root>/config/scoring_weights.yaml``.

    Returns:
        DataFrame with columns: ``Code``, ``fundamentals_score``,
        ``valuation_score``, ``sector_score``, ``factors_score``,
        ``kozo_score``, ``VI_score``, ``SP_score``, ``VI_rank``,
        ``SP_rank``.
    """
    # --- 1. Score each category -------------------------------------------
    category_scores: dict[str, pd.Series] = {}
    for cat_name in CATEGORY_NAMES:
        if cat_name not in category_dfs:
            logger.warning("Missing category DataFrame: %s", cat_name)
            continue
        cat_df = category_dfs[cat_name]
        scores = score_category_from_config(cat_df, cat_name, config_path=config_path)
        category_scores[cat_name] = scores

        non_zero = (scores != 0).sum()
        logger.info(
            "  %-15s  non-zero scores: %d / %d",
            cat_name, non_zero, len(scores),
        )

    # Use index from first available category DataFrame.
    ref_df = next(iter(category_dfs.values()))
    result = pd.DataFrame(index=ref_df.index)

    # --- 2. Attach Code column if present ---------------------------------
    if "Code" in ref_df.columns:
        result.insert(0, "Code", ref_df["Code"])

    # --- 3. Add individual category scores --------------------------------
    for cat_name in CATEGORY_NAMES:
        col_name = f"{cat_name}_score"
        if cat_name in category_scores:
            result[col_name] = category_scores[cat_name]
        else:
            result[col_name] = 0.0

    # --- 4. Compute VI and SP composites ----------------------------------
    config = load_config(config_path)
    cat_weights = config["category_weights"]

    for composite_name in ("VI", "SP"):
        weights = cat_weights[composite_name]
        total = pd.Series(0.0, index=result.index)
        total_weight = 0.0

        for cat_name, weight in weights.items():
            weight = float(weight)
            if weight == 0.0:
                continue
            if cat_name not in category_scores:
                logger.warning(
                    "Missing category %s for %s composite", cat_name, composite_name,
                )
                continue
            total += category_scores[cat_name] * weight
            total_weight += weight

        if 0 < total_weight < 1:
            total /= total_weight

        result[f"{composite_name}_score"] = total
        result[f"{composite_name}_rank"] = total.rank(ascending=False, method="min")

    return result

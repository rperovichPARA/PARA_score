"""Category 3: Sector Attractiveness scoring.

Used only in the SP composite (0% weight in VI).

Fetches factor-adjusted alpha z-scores from the ``/sector-signals``
endpoint on the portfoliotools Render service.  Each stock is mapped
to its TOPIX-17 sector via ``Sector17Code`` from the J-Quants listed
companies data, and receives the corresponding sector's alpha z-score
as its sector attractiveness score.

All stocks in the same TOPIX-17 sector receive the same score — this
is a sector-level attribute, not a stock-level one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.scoring.utils import load_metric_defs, load_scoring_params, score_category

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector signal fetching
# ---------------------------------------------------------------------------

def fetch_sector_signals(render_url: Optional[str] = None) -> dict[str, float]:
    """Fetch sector rotation signals from the portfoliotools Render service.

    Calls ``GET {RENDER_API_URL}/sector-signals`` and returns a mapping
    of sector identifier to factor-adjusted alpha z-score.

    Args:
        render_url: Base URL of the portfoliotools Render service.
            Falls back to the ``RENDER_API_URL`` environment variable.

    Returns:
        Dict mapping sector identifier (string) to its alpha z-score
        (float).  Returns an empty dict on failure.
    """
    base_url = render_url or os.getenv("RENDER_API_URL", "")
    logger.info(
        "fetch_sector_signals called — RENDER_API_URL=%s",
        base_url if base_url else "(not set)",
    )

    if not base_url:
        logger.warning(
            "No RENDER_API_URL configured. Sector signals unavailable. "
            "Set the RENDER_API_URL environment variable to the portfoliotools "
            "Render service URL."
        )
        return {}

    url = f"{base_url.rstrip('/')}/sector-signals"
    logger.info("Fetching sector signals from %s", url)

    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
        except requests.ConnectionError as exc:
            if attempt == max_retries:
                logger.error(
                    "Sector signals connection failed after %d attempts: %s",
                    max_retries, exc,
                )
                return {}
            wait = 2 ** attempt
            logger.warning(
                "Sector signals connection error (attempt %d/%d), "
                "retrying in %ds: %s",
                attempt, max_retries, wait, exc,
            )
            import time
            time.sleep(wait)
            continue

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            wait = 2 ** attempt
            logger.warning(
                "Sector signals HTTP %d (attempt %d/%d), retrying in %ds.",
                resp.status_code, attempt, max_retries, wait,
            )
            import time
            time.sleep(wait)
            continue

        if 200 <= resp.status_code < 300:
            break

        logger.error(
            "Sector signals request failed (HTTP %d): %s",
            resp.status_code,
            resp.text[:500] if resp.text else "(empty)",
        )
        return {}
    else:
        logger.error("Sector signals failed after %d retries.", max_retries)
        return {}

    # Parse the response JSON into a sector -> alpha_z mapping.
    try:
        data: Any = resp.json()
    except ValueError:
        logger.error("Sector signals response is not valid JSON.")
        return {}

    return _parse_sector_signals(data)


def _parse_sector_signals(data: Any) -> dict[str, float]:
    """Extract a sector-code -> alpha z-score mapping from the API response.

    Handles multiple possible response shapes from the
    ``/sector-signals`` endpoint:

    * **List of dicts**: ``[{"sector_code": "X", "alpha_z": 1.2}, ...]``
    * **Dict keyed by sector**: ``{"TOPIX17_1": {"alpha_z": 1.2}, ...}``
    * **Simple dict**: ``{"TOPIX17_1": 1.2, ...}``

    The sector code is normalised to a string for consistent lookup
    against ``Sector17Code`` values from J-Quants.

    Args:
        data: Parsed JSON from the API response.

    Returns:
        Dict mapping sector identifier (string) to alpha z-score.
    """
    signals: dict[str, float] = {}

    if isinstance(data, list):
        # List of dicts: [{"sector_code": "1", "alpha_z": 0.5}, ...]
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # Try common key names for the sector identifier.
            sector_key = (
                entry.get("sector_code")
                or entry.get("sector")
                or entry.get("code")
                or entry.get("Sector17Code")
                or entry.get("name")
            )
            # Try common key names for the alpha z-score.
            alpha = (
                entry.get("alpha_z")
                or entry.get("alpha_zscore")
                or entry.get("factor_adjusted_alpha")
                or entry.get("z_score")
                or entry.get("signal")
                or entry.get("score")
            )
            if sector_key is not None and alpha is not None:
                try:
                    signals[str(sector_key)] = float(alpha)
                except (ValueError, TypeError):
                    continue

    elif isinstance(data, dict):
        # Could be {"sectors": [...]} wrapper, or direct keyed dict.
        if "sectors" in data and isinstance(data["sectors"], list):
            return _parse_sector_signals(data["sectors"])
        if "signals" in data and isinstance(data["signals"], (list, dict)):
            return _parse_sector_signals(data["signals"])

        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dict: {"TOPIX17_1": {"alpha_z": 1.2, ...}}
                alpha = (
                    value.get("alpha_z")
                    or value.get("alpha_zscore")
                    or value.get("factor_adjusted_alpha")
                    or value.get("z_score")
                    or value.get("signal")
                    or value.get("score")
                )
                if alpha is not None:
                    try:
                        signals[str(key)] = float(alpha)
                    except (ValueError, TypeError):
                        continue
            elif isinstance(value, (int, float)):
                # Simple dict: {"TOPIX17_1": 1.2, ...}
                try:
                    signals[str(key)] = float(value)
                except (ValueError, TypeError):
                    continue

    if signals:
        logger.info(
            "Parsed sector signals for %d sectors: %s",
            len(signals),
            {k: round(v, 3) for k, v in sorted(signals.items())},
        )
    else:
        logger.warning("No sector signals could be parsed from the response.")

    return signals


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_sector_metrics(
    financials: pd.DataFrame,
    sector_signals: Optional[dict[str, float]] = None,
    render_url: Optional[str] = None,
) -> pd.DataFrame:
    """Derive sector attractiveness metrics.

    Maps each stock's ``Sector17Code`` to the factor-adjusted alpha
    z-score from the sector rotation signals.  All stocks in the same
    TOPIX-17 sector receive the same ``sector_alpha`` value.

    If *sector_signals* is not provided, fetches them from
    ``{RENDER_API_URL}/sector-signals``.

    Args:
        financials: DataFrame with financial data.  Must contain a
            ``Sector17Code`` column (from J-Quants listed companies).
        sector_signals: Optional pre-fetched mapping of sector
            identifier to alpha z-score.  If ``None``, signals are
            fetched from the Render endpoint.
        render_url: Optional base URL override for the Render service.

    Returns:
        Copy of *financials* with a ``sector_alpha`` column added.
    """
    df = financials.copy()
    logger.info(
        "compute_sector_metrics called — %d rows, columns: %s",
        len(df),
        [c for c in df.columns if "ector" in c.lower()],
    )

    # Fetch signals if not provided.
    if sector_signals is None:
        logger.info("No sector_signals passed — fetching from Render endpoint.")
        sector_signals = fetch_sector_signals(render_url=render_url)
    else:
        logger.info(
            "sector_signals provided: %d sectors, keys=%s",
            len(sector_signals),
            list(sector_signals.keys())[:10],
        )

    if not sector_signals:
        logger.warning(
            "No sector signals available — sector_alpha will be NaN for all %d stocks. "
            "Check that RENDER_API_URL is set and /sector-signals is reachable.",
            len(df),
        )
        df["sector_alpha"] = np.nan
        return df

    if "Sector17Code" not in df.columns:
        logger.warning(
            "Sector17Code column missing from financials — "
            "sector_alpha will be NaN. Available columns: %s",
            [c for c in df.columns if "ector" in c.lower() or "code" in c.lower()],
        )
        df["sector_alpha"] = np.nan
        return df

    # Normalise Sector17Code to string for matching.
    sector_codes = df["Sector17Code"].astype(str).str.strip()

    # Try direct mapping first.
    mapped = sector_codes.map(sector_signals)

    # If most values are unmapped, try stripping common prefixes from
    # the signal keys (e.g. "TOPIX17_1" -> "1") and retry.
    if mapped.isna().mean() > 0.9 and len(sector_signals) > 0:
        logger.info(
            "Direct sector code mapping yielded <10%% coverage; "
            "attempting prefix-stripped mapping."
        )
        stripped_signals: dict[str, float] = {}
        for key, value in sector_signals.items():
            # Strip common prefixes: "TOPIX17_", "sector_", etc.
            stripped = key
            for prefix in ("TOPIX17_", "topix17_", "TOPIX_", "topix_",
                           "sector_", "Sector_", "S17_", "s17_"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            stripped_signals[stripped] = value
        mapped = sector_codes.map(stripped_signals)

    # If still unmapped, try matching on Sector17CodeName if available.
    if mapped.isna().mean() > 0.9 and "Sector17CodeName" in df.columns:
        logger.info(
            "Code-based mapping yielded <10%% coverage; "
            "attempting name-based mapping."
        )
        sector_names = df["Sector17CodeName"].astype(str).str.strip()
        name_mapped = sector_names.map(sector_signals)
        # Also try case-insensitive matching.
        if name_mapped.isna().mean() > 0.9:
            lower_signals = {k.lower(): v for k, v in sector_signals.items()}
            name_mapped = sector_names.str.lower().map(lower_signals)
        mapped = mapped.fillna(name_mapped)

    df["sector_alpha"] = mapped.values

    coverage = df["sector_alpha"].notna().mean()
    logger.info(
        "Sector  %-25s coverage: %5.1f%%", "sector_alpha", coverage * 100,
    )

    n_mapped = df["sector_alpha"].notna().sum()
    n_total = len(df)
    unique_sectors = sector_codes[df["sector_alpha"].notna()].nunique()
    logger.info(
        "Sector signals mapped: %d / %d stocks across %d sectors",
        n_mapped, n_total, unique_sectors,
    )

    return df


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def score_sector(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> pd.Series:
    """Score the Sector Attractiveness category.

    Loads metric definitions and scoring parameters from the YAML config,
    then delegates to :func:`~src.scoring.utils.score_category`.

    Since ``sector_alpha`` is already a z-score from the sector rotation
    framework, the winsorized z-score in ``score_category`` acts as a
    normalisation pass across the universe.

    Args:
        df: DataFrame with computed sector metric columns (output of
            :func:`compute_sector_metrics`).
        config_path: Path to scoring weights YAML.  Defaults to
            ``config/scoring_weights.yaml`` at the repository root.

    Returns:
        Series of sector category scores indexed by company.
    """
    metric_defs = load_metric_defs("sector", config_path=config_path)
    params = load_scoring_params(config_path=config_path)
    return score_category(df, metric_defs, min_coverage=params["min_coverage"])

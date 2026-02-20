#!/usr/bin/env python
"""Run the full scoring pipeline with the 5-stock synthetic test universe.

Produces output/scores.csv and output/scores.xlsx using the same scoring
functions as the production pipeline, then calls export --dry-run to
generate and print the JSON payload.

No J-Quants API access required — uses synthetic data identical to the
integration test suite.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Ensure project root is on sys.path.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scoring.composite import compute_composite_scores
from src.scoring.factors import compute_factor_metrics
from src.scoring.fundamentals import compute_fundamentals_metrics
from src.scoring.kozo import compute_kozo_metrics
from src.scoring.sector import compute_sector_metrics
from src.scoring.valuation import compute_valuation_metrics
from src.export import format_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic test data (same as tests/test_pipeline_integration.py)
# ---------------------------------------------------------------------------

TEST_CODES = ["7203", "6758", "8306", "9984", "6861"]
TEST_NAMES = [
    "Toyota Motor Corp",
    "Sony Group Corp",
    "Mitsubishi UFJ Financial Group",
    "SoftBank Group Corp",
    "Keyence Corp",
]
TEST_SECTORS = ["3050", "3650", "3150", "3650", "3600"]


def _make_financials() -> pd.DataFrame:
    np.random.seed(42)
    n = len(TEST_CODES)
    return pd.DataFrame({
        "Code": TEST_CODES,
        "CompanyName": TEST_NAMES,
        "CompanyNameEnglish": TEST_NAMES,
        "Sector33Code": TEST_SECTORS,
        "MarketCode": ["0111"] * n,
        "NetSales": [30_000_000, 12_000_000, 8_000_000, 6_500_000, 900_000],
        "OperatingProfit": [3_000_000, 1_200_000, 1_500_000, 800_000, 400_000],
        "Profit": [2_500_000, 900_000, 1_200_000, 500_000, 350_000],
        "Equity": [25_000_000, 5_000_000, 18_000_000, 4_000_000, 1_500_000],
        "TotalAssets": [60_000_000, 30_000_000, 380_000_000, 45_000_000, 2_000_000],
        "EquityToAssetRatio": [41.7, 16.7, 4.7, 8.9, 75.0],
        "EarningsPerShare": [180.0, 73.0, 95.0, 34.0, 620.0],
        "BookValuePerShare": [1800.0, 405.0, 1400.0, 270.0, 2500.0],
        "ForecastOperatingProfit": [3_200_000, 1_300_000, 1_600_000, 900_000, 420_000],
        "ForecastEarningsPerShare": [195.0, 80.0, 100.0, 40.0, 660.0],
        "NextYearForecastNetSales": [32_000_000, 13_000_000, 8_500_000, 7_000_000, 980_000],
        "NextYearForecastOperatingProfit": [3_400_000, 1_400_000, 1_700_000, 1_000_000, 450_000],
        "NextYearForecastEarningsPerShare": [210.0, 88.0, 108.0, 45.0, 700.0],
        "ResultPayoutRatioAnnual": [30.0, 15.0, 40.0, 5.0, 20.0],
        "ForecastNetSales": [31_000_000, 12_500_000, 8_200_000, 6_800_000, 950_000],
    })


def _make_prices() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2025-08-19", periods=120, freq="B")
    base_prices = {"7203": 2400, "6758": 3000, "8306": 1500, "9984": 9000, "6861": 65000}
    records = []
    for code, base_px in base_prices.items():
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = base_px * np.cumprod(1 + returns)
        volumes = np.random.randint(500_000, 5_000_000, len(dates))
        for i, date in enumerate(dates):
            records.append({
                "Code": code,
                "Date": date,
                "AdjustmentClose": round(prices[i], 1),
                "Volume": int(volumes[i]),
            })
    return pd.DataFrame(records)


def _make_topix() -> pd.DataFrame:
    dates = pd.bdate_range("2025-08-19", periods=120, freq="B")
    np.random.seed(99)
    returns = np.random.normal(0.0002, 0.008, len(dates))
    levels = 2700 * np.cumprod(1 + returns)
    return pd.DataFrame({"Date": dates, "Close": levels.round(2)})


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PARA Score — 5-stock test universe pipeline run")
    logger.info("=" * 60)

    # Build synthetic data.
    financials = _make_financials()
    prices = _make_prices()
    topix = _make_topix()
    logger.info("Test universe: %s", ", ".join(TEST_CODES))

    # Mock Google Sheets to return empty DataFrame.
    with patch(
        "src.data.gsheets.GoogleSheetsClient._fetch_csv",
        return_value=pd.DataFrame(),
    ):
        # ── Compute category metrics ──
        logger.info("Computing fundamentals metrics...")
        fund_df = compute_fundamentals_metrics(financials, prices)

        logger.info("Computing valuation metrics...")
        val_df = compute_valuation_metrics(financials, prices, topix=topix)

        logger.info("Computing sector metrics...")
        sector_df = compute_sector_metrics(financials)

        logger.info("Computing factor metrics...")
        factors_df = compute_factor_metrics(financials)

        logger.info("Computing kozo metrics...")
        kozo_df = compute_kozo_metrics(fund_df)

    category_dfs = {
        "fundamentals": fund_df,
        "valuation": val_df,
        "sector": sector_df,
        "factors": factors_df,
        "kozo": kozo_df,
    }

    # ── Composite scoring ──
    logger.info("Computing composite VI / SP scores...")
    results = compute_composite_scores(category_dfs)

    # Attach company identifiers.
    for col in ["CompanyNameEnglish", "Sector33Code"]:
        if col in financials.columns and col not in results.columns:
            mapping = financials.set_index(financials["Code"].str.strip())[col]
            if "Code" in results.columns:
                results[col] = results["Code"].str.strip().map(mapping)

    # Reorder columns.
    id_cols = [c for c in ("Code", "CompanyNameEnglish", "Sector33Code") if c in results.columns]
    score_cols = [c for c in results.columns if c not in id_cols]
    results = results[id_cols + score_cols]

    # ── Write CSV ──
    csv_path = os.path.join(output_dir, "scores.csv")
    results.to_csv(csv_path, index=False)
    logger.info("Scores written to %s", csv_path)

    # ── Print composite results ──
    print("\n" + "=" * 70)
    print("COMPOSITE SCORES (5-stock test universe)")
    print("=" * 70)
    print(results.to_string(index=False))
    print()

    # ── Format export payload ──
    logger.info("Formatting export payload...")
    payload = format_payload(results, category_dfs)

    json_path = os.path.join(output_dir, "payload.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("EXPORT DRY-RUN — JSON PAYLOAD")
    print("=" * 70)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
    print(f"Payload written to {json_path}")

    # ── Schema validation summary ──
    print("\n" + "=" * 70)
    print("SCHEMA VALIDATION")
    print("=" * 70)
    errors = []
    # Check top-level keys
    for key in ["stocks", "metadata", "coverage"]:
        if key not in payload:
            errors.append(f"Missing top-level key: {key}")

    # Check scores records
    required_score_fields = ["code", "name", "sector", "VI_score", "SP_score",
                             "VI_rank", "SP_rank", "fundamentals_score",
                             "valuation_score", "sector_score", "factors_score",
                             "kozo_score"]
    for i, rec in enumerate(payload.get("stocks", [])):
        for field in required_score_fields:
            if field not in rec:
                errors.append(f"scores[{i}] missing field: {field}")

    # Check metadata
    required_meta = ["run_timestamp", "universe_size", "pipeline_version", "data_sources"]
    for field in required_meta:
        if field not in payload.get("metadata", {}):
            errors.append(f"metadata missing field: {field}")

    # Check coverage structure
    for metric, stats in payload.get("coverage", {}).items():
        if "count" not in stats or "pct" not in stats:
            errors.append(f"coverage[{metric}] missing count/pct")

    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"PASSED — All {len(payload['scores'])} score records match schema")
        print(f"  • {len(payload['scores'])} stocks scored")
        print(f"  • {len(payload['coverage'])} coverage metrics reported")
        print(f"  • metadata: version={payload['metadata']['pipeline_version']}, "
              f"universe_size={payload['metadata']['universe_size']}")


if __name__ == "__main__":
    main()

"""Export pipeline results to JSON and upload to Render.

Formats the scored output from :func:`~src.pipeline.run_pipeline` into the
JSON schema consumed by the ``/para-score/upload`` endpoint on the
portfoliotools Render service, and optionally POSTs it.

JSON payload schema::

    {
        "scores": [
            {
                "code": "7203",
                "name": "Toyota Motor Corp",
                "sector": "Transportation Equipment",
                "VI_score": 1.42,
                "SP_score": 0.87,
                "VI_rank": 12,
                "SP_rank": 45,
                "fundamentals_score": 0.9,
                "valuation_score": 1.1,
                "sector_score": 0.3,
                "factors_score": 0.0,
                "kozo_score": 1.8
            }
        ],
        "metadata": {
            "run_timestamp": "2026-02-14T01:00:00Z",
            "universe_size": 3742,
            "pipeline_version": "0.1.0",
            "data_sources": ["jquants", "google_sheets"]
        },
        "coverage": {
            "roe": {"count": 3650, "pct": 0.975},
            "opm": {"count": 3620, "pct": 0.967}
        }
    }

CLI usage::

    python -m src.export --input ./output/scores.csv --upload
    python -m src.export --input ./output/scores.csv --dry-run
    python -m src.export --input ./output/scores.csv --dry-run --output-file ./output/payload.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"

# Score and category columns expected in the pipeline output.
_SCORE_COLS: list[str] = [
    "VI_score",
    "SP_score",
    "VI_rank",
    "SP_rank",
    "fundamentals_score",
    "valuation_score",
    "sector_score",
    "factors_score",
    "kozo_score",
]

# Metric columns to report coverage for.  These are the most commonly
# available metrics from J-Quants and supplement sheets.
_COVERAGE_METRICS: list[str] = [
    "roe",
    "opm",
    "equity_ratio",
    "f2_f0_sales_growth",
    "f2_f0_ebit_growth",
    "f1_f0_op_growth",
    "f2_f1_op_growth",
    "net_cash_mktcap",
    "altman_z",
    "pbr_vs_10yr",
    "pen_vs_10yr",
    "pegn",
    "adv_liquidity",
    "broker_target_upside",
    "peer_target_upside",
    "price_6mo_vs_tpx",
    "para_target_upside",
    "payout_ratio",
    "roe_vs_peer",
    "excess_cash_mktcap",
    "lti_mktcap",
    "land_mktcap",
    "board_size",
    "analyst_coverage",
    "ke_vs_peer",
]


# ---------------------------------------------------------------------------
# Payload formatting
# ---------------------------------------------------------------------------

def format_payload(
    scores_df: pd.DataFrame,
    category_dfs: Optional[dict[str, pd.DataFrame]] = None,
    data_sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Format pipeline output into the JSON upload payload.

    Args:
        scores_df: Composite scores DataFrame from
            :func:`~src.pipeline.run_pipeline`.  Expected columns:
            ``Code``, ``CompanyName`` or ``CompanyNameEnglish``,
            ``Sector33Code``, plus all score/rank columns.
        category_dfs: Optional dict of category name -> metric DataFrame.
            Used to compute per-metric coverage statistics.  If ``None``,
            coverage is computed from columns present in *scores_df*.
        data_sources: List of data source names to include in metadata.
            Defaults to ``["jquants", "google_sheets"]``.

    Returns:
        Dict matching the ``/para-score/upload`` JSON schema.
    """
    if data_sources is None:
        data_sources = ["jquants", "google_sheets"]

    # --- Build scores array ------------------------------------------------
    records: list[dict[str, Any]] = []
    for _, row in scores_df.iterrows():
        code = str(row.get("Code", "")).strip()
        name = row.get("CompanyNameEnglish") or row.get("CompanyName", "")
        sector = row.get("Sector33Code", "")

        record: dict[str, Any] = {
            "code": code,
            "name": str(name) if pd.notna(name) else "",
            "sector": str(sector) if pd.notna(sector) else "",
        }

        for col in _SCORE_COLS:
            val = row.get(col)
            if pd.notna(val):
                if col.endswith("_rank"):
                    record[col] = int(val)
                else:
                    record[col] = round(float(val), 4)
            else:
                record[col] = None

        records.append(record)

    # --- Build metadata ----------------------------------------------------
    metadata: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "universe_size": len(scores_df),
        "pipeline_version": PIPELINE_VERSION,
        "data_sources": data_sources,
    }

    # --- Build coverage stats ----------------------------------------------
    coverage: dict[str, dict[str, Any]] = {}

    # Collect all available DataFrames for coverage scanning.
    all_metric_dfs: list[pd.DataFrame] = []
    if category_dfs:
        all_metric_dfs.extend(category_dfs.values())
    else:
        all_metric_dfs.append(scores_df)

    for metric in _COVERAGE_METRICS:
        for mdf in all_metric_dfs:
            if metric in mdf.columns:
                count = int(mdf[metric].notna().sum())
                total = len(mdf)
                coverage[metric] = {
                    "count": count,
                    "pct": round(count / total, 3) if total > 0 else 0.0,
                }
                break

    payload: dict[str, Any] = {
        "scores": records,
        "metadata": metadata,
        "coverage": coverage,
    }

    logger.info(
        "Payload formatted: %d scores, %d coverage metrics",
        len(records),
        len(coverage),
    )
    return payload


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_payload(
    payload: dict[str, Any],
    render_url: Optional[str] = None,
    upload_key: Optional[str] = None,
) -> bool:
    """POST the payload to the Render upload endpoint.

    Args:
        payload: Formatted JSON payload from :func:`format_payload`.
        render_url: Base URL of the portfoliotools Render service.
            Falls back to ``RENDER_API_URL`` env var.
        upload_key: Bearer token for authentication.  Falls back to
            ``PARA_SCORE_UPLOAD_KEY`` env var.

    Returns:
        ``True`` if the upload succeeded (HTTP 2xx), ``False`` otherwise.
    """
    base_url = render_url or os.getenv("RENDER_API_URL", "")
    token = upload_key or os.getenv("PARA_SCORE_UPLOAD_KEY", "")

    if not base_url:
        logger.error(
            "No Render API URL configured. Set RENDER_API_URL env var "
            "or pass render_url."
        )
        return False

    if not token:
        logger.error(
            "No upload key configured. Set PARA_SCORE_UPLOAD_KEY env var "
            "or pass upload_key."
        )
        return False

    url = f"{base_url.rstrip('/')}/para-score/upload"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Retry with exponential backoff on transient errors.
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )
        except requests.ConnectionError as exc:
            if attempt == max_retries:
                logger.error("Upload connection failed after %d attempts: %s", max_retries, exc)
                return False
            wait = 2 ** attempt
            logger.warning(
                "Upload connection error (attempt %d/%d), retrying in %ds: %s",
                attempt, max_retries, wait, exc,
            )
            import time
            time.sleep(wait)
            continue

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            wait = 2 ** attempt
            logger.warning(
                "Upload HTTP %d (attempt %d/%d), retrying in %ds.",
                resp.status_code, attempt, max_retries, wait,
            )
            import time
            time.sleep(wait)
            continue

        if 200 <= resp.status_code < 300:
            logger.info(
                "Upload successful (HTTP %d): %s",
                resp.status_code,
                resp.text[:200] if resp.text else "(empty)",
            )
            return True

        logger.error(
            "Upload failed (HTTP %d): %s",
            resp.status_code,
            resp.text[:500] if resp.text else "(empty)",
        )
        return False

    logger.error("Upload failed after %d retries.", max_retries)
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run export."""
    parser = argparse.ArgumentParser(
        description=(
            "PARA Score Export — format pipeline output as JSON and upload "
            "to the Render scoring endpoint."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./output/scores.csv",
        help="Path to the pipeline output CSV (scores.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Write the JSON payload to a local file instead of uploading. "
            "Output path is controlled by --output-file."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload the payload to the Render endpoint.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help=(
            "Path for the JSON output file (used with --dry-run). "
            "Defaults to ./output/payload.json."
        ),
    )
    parser.add_argument(
        "--render-url",
        type=str,
        default="",
        help="Override RENDER_API_URL for the upload target.",
    )
    parser.add_argument(
        "--upload-key",
        type=str,
        default="",
        help="Override PARA_SCORE_UPLOAD_KEY for authentication.",
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

    # Neither --dry-run nor --upload: default to dry-run.
    if not args.dry_run and not args.upload:
        logger.info("No action specified; defaulting to --dry-run.")
        args.dry_run = True

    # Load scores CSV.
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading scores from %s", input_path)
    scores_df = pd.read_csv(input_path)
    logger.info("Loaded %d company scores.", len(scores_df))

    # Format payload.
    payload = format_payload(scores_df)

    # Dry-run: write to local file.
    if args.dry_run:
        output_file = args.output_file or str(
            input_path.parent / "payload.json"
        )
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"Dry-run: payload written to {output_file} ({len(payload['scores'])} scores)")

    # Upload.
    if args.upload:
        success = upload_payload(
            payload,
            render_url=args.render_url or None,
            upload_key=args.upload_key or None,
        )
        if success:
            print(f"Upload successful ({len(payload['scores'])} scores)")
        else:
            logger.error("Upload failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()

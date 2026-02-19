"""PARA Score serving blueprint — pure read layer for pre-computed scores.

This Flask blueprint caches the JSON payload uploaded by the scoring
pipeline (via ``POST /para-score/upload``) and serves it through
lightweight GET endpoints consumed by n8n AI agents and internal tools.

No computation happens here.  Scores are pre-computed by the pipeline
in ``src/pipeline.py``, formatted by ``src/export.py``, and POSTed to
the ``/para-score/upload`` endpoint on a weekly cron via GitHub Actions.

Endpoints
---------
GET  /para-score             Full scored universe
GET  /para-score/stock/<code>  Single stock breakdown
GET  /para-score/top/<n>     Top N stocks (sort by VI or SP)
GET  /para-score/screen      Filter by score thresholds, rank caps, sector
GET  /para-score/summary     Coverage stats and score distributions
GET  /para-score/status      Cache freshness check
POST /para-score/upload      Receive pre-computed JSON (bearer-token auth)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger(__name__)

para_score_bp = Blueprint("para_score", __name__)

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {
    "scores": [],
    "metadata": {},
    "coverage": {},
    "uploaded_at": None,
}


def _get_cache() -> dict[str, Any]:
    """Return the current cache dict."""
    return _cache


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_upload_auth() -> Optional[Response]:
    """Validate Bearer token for upload endpoint.

    Returns ``None`` if authorised, or a JSON error ``Response`` otherwise.
    """
    expected = os.getenv("PARA_SCORE_UPLOAD_KEY", "")
    if not expected:
        logger.error("PARA_SCORE_UPLOAD_KEY not configured on server")
        return jsonify({"error": "Upload not configured"}), 500

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or malformed Authorization header"}), 401

    token = auth_header[len("Bearer "):]
    if token != expected:
        return jsonify({"error": "Invalid token"}), 403

    return None


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _build_code_index(scores: list[dict]) -> dict[str, dict]:
    """Build a code -> record lookup from the scores list."""
    index: dict[str, dict] = {}
    for record in scores:
        code = str(record.get("code", "")).strip()
        if code:
            index[code] = record
    return index


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a sorted list of floats."""
    if not values:
        return 0.0
    arr = np.array(values)
    return float(np.percentile(arr, pct))


# ---------------------------------------------------------------------------
# POST /para-score/upload
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/upload", methods=["POST"])
def upload_scores() -> tuple[Response, int]:
    """Receive pre-computed scores JSON from the pipeline.

    Expects the payload schema documented in ``src/export.py``::

        {
            "scores": [...],
            "metadata": {...},
            "coverage": {...}
        }

    Authenticated via ``Authorization: Bearer <PARA_SCORE_UPLOAD_KEY>``.
    """
    auth_err = _check_upload_auth()
    if auth_err is not None:
        return auth_err

    data = request.get_json(silent=True)
    if not data or "scores" not in data:
        return jsonify({"error": "Invalid payload: 'scores' key required"}), 400

    _cache["scores"] = data["scores"]
    _cache["metadata"] = data.get("metadata", {})
    _cache["coverage"] = data.get("coverage", {})
    _cache["uploaded_at"] = datetime.now(timezone.utc).isoformat()

    count = len(data["scores"])
    logger.info("Scores uploaded: %d records cached", count)
    return jsonify({"status": "ok", "scores_cached": count}), 200


# ---------------------------------------------------------------------------
# GET /para-score
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score", methods=["GET"])
def get_all_scores() -> tuple[Response, int]:
    """Return the full scored universe."""
    cache = _get_cache()
    if not cache["scores"]:
        return jsonify({"error": "No scores available", "hint": "Pipeline has not uploaded yet"}), 404

    return jsonify({
        "scores": cache["scores"],
        "metadata": cache["metadata"],
        "count": len(cache["scores"]),
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/stock/<code>
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/stock/<code>", methods=["GET"])
def get_stock_score(code: str) -> tuple[Response, int]:
    """Return score breakdown for a single stock by code."""
    cache = _get_cache()
    if not cache["scores"]:
        return jsonify({"error": "No scores available"}), 404

    index = _build_code_index(cache["scores"])
    code = code.strip()

    record = index.get(code)
    if record is None:
        return jsonify({"error": f"Stock {code} not found"}), 404

    return jsonify(record), 200


# ---------------------------------------------------------------------------
# GET /para-score/top/<n>
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/top/<int:n>", methods=["GET"])
def get_top_scores(n: int) -> tuple[Response, int]:
    """Return the top *n* stocks sorted by VI or SP score.

    Query params:
        sort: ``vi`` (default) or ``sp``
    """
    cache = _get_cache()
    if not cache["scores"]:
        return jsonify({"error": "No scores available"}), 404

    sort_by = request.args.get("sort", "vi").lower()
    if sort_by not in ("vi", "sp"):
        return jsonify({"error": "sort must be 'vi' or 'sp'"}), 400

    score_key = "VI_score" if sort_by == "vi" else "SP_score"

    scored = [s for s in cache["scores"] if s.get(score_key) is not None]
    scored.sort(key=lambda s: s[score_key], reverse=True)

    n = max(1, min(n, len(scored)))

    return jsonify({
        "sort": sort_by,
        "count": n,
        "scores": scored[:n],
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/screen
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/screen", methods=["GET"])
def screen_scores() -> tuple[Response, int]:
    """Filter scores by thresholds, rank caps, sector, and limit.

    Query params:
        min_vi:       Minimum VI_score
        min_sp:       Minimum SP_score
        max_vi_rank:  Maximum VI_rank (e.g. top 100)
        max_sp_rank:  Maximum SP_rank
        sector:       Exact sector name match
        limit:        Max number of results (default: all)
    """
    cache = _get_cache()
    if not cache["scores"]:
        return jsonify({"error": "No scores available"}), 404

    results = list(cache["scores"])

    # --- Apply filters -------------------------------------------------------
    min_vi = request.args.get("min_vi", type=float)
    if min_vi is not None:
        results = [s for s in results if (s.get("VI_score") or -float("inf")) >= min_vi]

    min_sp = request.args.get("min_sp", type=float)
    if min_sp is not None:
        results = [s for s in results if (s.get("SP_score") or -float("inf")) >= min_sp]

    max_vi_rank = request.args.get("max_vi_rank", type=int)
    if max_vi_rank is not None:
        results = [s for s in results if s.get("VI_rank") is not None and s["VI_rank"] <= max_vi_rank]

    max_sp_rank = request.args.get("max_sp_rank", type=int)
    if max_sp_rank is not None:
        results = [s for s in results if s.get("SP_rank") is not None and s["SP_rank"] <= max_sp_rank]

    sector = request.args.get("sector", type=str)
    if sector is not None:
        results = [s for s in results if s.get("sector", "").lower() == sector.lower()]

    limit = request.args.get("limit", type=int)
    if limit is not None and limit > 0:
        results = results[:limit]

    return jsonify({
        "count": len(results),
        "filters": {
            k: v for k, v in {
                "min_vi": min_vi,
                "min_sp": min_sp,
                "max_vi_rank": max_vi_rank,
                "max_sp_rank": max_sp_rank,
                "sector": sector,
                "limit": limit,
            }.items() if v is not None
        },
        "scores": results,
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/summary
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/summary", methods=["GET"])
def get_summary() -> tuple[Response, int]:
    """Return coverage stats and score distribution summaries."""
    cache = _get_cache()
    if not cache["scores"]:
        return jsonify({"error": "No scores available"}), 404

    scores = cache["scores"]

    def _distribution(key: str) -> dict[str, Any]:
        vals = [s[key] for s in scores if s.get(key) is not None]
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(min(vals), 4),
            "p25": round(_percentile(vals, 25), 4),
            "median": round(_percentile(vals, 50), 4),
            "p75": round(_percentile(vals, 75), 4),
            "max": round(max(vals), 4),
        }

    distributions = {
        "VI_score": _distribution("VI_score"),
        "SP_score": _distribution("SP_score"),
        "fundamentals_score": _distribution("fundamentals_score"),
        "valuation_score": _distribution("valuation_score"),
        "sector_score": _distribution("sector_score"),
        "factors_score": _distribution("factors_score"),
        "kozo_score": _distribution("kozo_score"),
    }

    # Sector breakdown
    sector_counts: dict[str, int] = {}
    for s in scores:
        sec = s.get("sector", "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return jsonify({
        "universe_size": len(scores),
        "metadata": cache["metadata"],
        "coverage": cache["coverage"],
        "distributions": distributions,
        "sector_counts": sector_counts,
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/status
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/status", methods=["GET"])
def get_status() -> tuple[Response, int]:
    """Return cache freshness info."""
    cache = _get_cache()
    return jsonify({
        "has_data": bool(cache["scores"]),
        "scores_count": len(cache["scores"]),
        "uploaded_at": cache["uploaded_at"],
        "pipeline_run": cache["metadata"].get("run_timestamp"),
        "pipeline_version": cache["metadata"].get("pipeline_version"),
        "universe_size": cache["metadata"].get("universe_size"),
    }), 200

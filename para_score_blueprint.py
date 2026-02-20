"""
PARA Score Blueprint
====================
Adds /para-score endpoints to the portfolio-optimizer Flask app.

Endpoints:
    GET  /para-score                    Latest cached scores (fast, <100ms)
    GET  /para-score/stock/<code>       Single stock breakdown
    GET  /para-score/top/<n>            Top N stocks (sort by VI or SP)
    GET  /para-score/screen             Filter by score thresholds, rank caps, sector
    GET  /para-score/summary            Coverage stats and score distributions
    GET  /para-score/status             Cache freshness check
    POST /para-score/upload             Receive pre-computed JSON (bearer-token auth)

Caching:
    Results are cached to disk as JSON.  The scoring pipeline
    (GitHub Actions, weekly cron) POSTs pre-computed scores to
    /para-score/upload.  All GET endpoints serve from the cached file.

This is a pure serving layer -- no imports from the PARA_score repo.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
from flask import Blueprint, jsonify, request

para_score_bp = Blueprint("para_score", __name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get("PARA_SCORE_CACHE_DIR", "/tmp/para_score")
CACHE_FILE = os.path.join(CACHE_DIR, "latest_scores.json")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _read_cache():
    """Read cached scores from disk.  Returns dict or None."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(data: dict):
    """Write scores payload to disk with upload timestamp."""
    _ensure_cache_dir()
    data["_uploaded_at"] = datetime.now(timezone.utc).isoformat()
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_code_index(scores: list) -> dict:
    """Build a code -> record lookup from the scores list."""
    index = {}
    for record in scores:
        code = str(record.get("code", "")).strip()
        if code:
            index[code] = record
    return index


def _percentile(values: list, pct: float) -> float:
    """Compute a percentile from a list of floats."""
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_upload_auth():
    """Validate Bearer token for upload endpoint.

    Returns None if authorised, or a (response, status) tuple otherwise.
    """
    expected = os.environ.get("PARA_SCORE_UPLOAD_KEY", "")
    if not expected:
        return jsonify({"error": "Upload not configured"}), 500

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or malformed Authorization header"}), 401

    token = auth_header[len("Bearer "):]
    if token != expected:
        return jsonify({"error": "Invalid token"}), 403

    return None


# ---------------------------------------------------------------------------
# POST /para-score/upload
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/upload", methods=["POST"])
def upload_scores():
    """Receive pre-computed scores JSON from the pipeline.

    Expects::

        {
            "stocks": [...],
            "metadata": {...},
            "coverage": {...}
        }

    Authenticated via ``Authorization: Bearer <PARA_SCORE_UPLOAD_KEY>``.
    """
    auth_err = _check_upload_auth()
    if auth_err is not None:
        return auth_err

    data = request.get_json(silent=True)
    if not data or "stocks" not in data:
        return jsonify({"error": "Payload must include a non-empty stocks array"}), 400

    _write_cache(data)

    count = len(data["stocks"])
    return jsonify({"status": "ok", "stocks_cached": count}), 200


# ---------------------------------------------------------------------------
# GET /para-score
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score", methods=["GET"])
def get_all_scores():
    """Return the full scored universe."""
    cached = _read_cache()
    if not cached or not cached.get("stocks"):
        return jsonify({"error": "No scores available", "hint": "Pipeline has not uploaded yet"}), 404

    return jsonify({
        "stocks": cached["stocks"],
        "metadata": cached.get("metadata", {}),
        "count": len(cached["stocks"]),
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/stock/<code>
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/stock/<code>", methods=["GET"])
def get_stock_score(code):
    """Return score breakdown for a single stock by code."""
    cached = _read_cache()
    if not cached or not cached.get("stocks"):
        return jsonify({"error": "No scores available"}), 404

    index = _build_code_index(cached["stocks"])
    record = index.get(code.strip())
    if record is None:
        return jsonify({"error": f"Stock {code} not found"}), 404

    return jsonify(record), 200


# ---------------------------------------------------------------------------
# GET /para-score/top/<n>
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/top/<int:n>", methods=["GET"])
def get_top_scores(n):
    """Return the top *n* stocks sorted by VI or SP score.

    Query params:
        sort: ``vi`` (default) or ``sp``
    """
    cached = _read_cache()
    if not cached or not cached.get("stocks"):
        return jsonify({"error": "No scores available"}), 404

    sort_by = request.args.get("sort", "vi").lower()
    if sort_by not in ("vi", "sp"):
        return jsonify({"error": "sort must be 'vi' or 'sp'"}), 400

    score_key = "VI_score" if sort_by == "vi" else "SP_score"

    scored = [s for s in cached["stocks"] if s.get(score_key) is not None]
    scored.sort(key=lambda s: s[score_key], reverse=True)

    n = max(1, min(n, len(scored)))

    return jsonify({
        "sort": sort_by,
        "count": n,
        "stocks": scored[:n],
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/screen
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/screen", methods=["GET"])
def screen_scores():
    """Filter scores by thresholds, rank caps, sector, and limit.

    Query params:
        min_vi:       Minimum VI_score
        min_sp:       Minimum SP_score
        max_vi_rank:  Maximum VI_rank (e.g. top 100)
        max_sp_rank:  Maximum SP_rank
        sector:       Exact sector name match
        limit:        Max number of results (default: all)
    """
    cached = _read_cache()
    if not cached or not cached.get("stocks"):
        return jsonify({"error": "No scores available"}), 404

    results = list(cached["stocks"])

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
        "stocks": results,
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/summary
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/summary", methods=["GET"])
def get_summary():
    """Return coverage stats and score distribution summaries."""
    cached = _read_cache()
    if not cached or not cached.get("stocks"):
        return jsonify({"error": "No scores available"}), 404

    scores = cached["stocks"]

    def _distribution(key):
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

    sector_counts = {}
    for s in scores:
        sec = s.get("sector", "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    return jsonify({
        "universe_size": len(scores),
        "metadata": cached.get("metadata", {}),
        "coverage": cached.get("coverage", {}),
        "distributions": distributions,
        "sector_counts": sector_counts,
    }), 200


# ---------------------------------------------------------------------------
# GET /para-score/status
# ---------------------------------------------------------------------------

@para_score_bp.route("/para-score/status", methods=["GET"])
def get_status():
    """Return cache freshness info."""
    cached = _read_cache()
    if not cached:
        return jsonify({
            "has_data": False,
            "scores_count": 0,
            "uploaded_at": None,
            "pipeline_run": None,
            "pipeline_version": None,
            "universe_size": None,
        }), 200

    metadata = cached.get("metadata", {})
    return jsonify({
        "has_data": bool(cached.get("stocks")),
        "scores_count": len(cached.get("stocks", [])),
        "uploaded_at": cached.get("_uploaded_at"),
        "pipeline_run": metadata.get("run_timestamp"),
        "pipeline_version": metadata.get("pipeline_version"),
        "universe_size": metadata.get("universe_size"),
    }), 200

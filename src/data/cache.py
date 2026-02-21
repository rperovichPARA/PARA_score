"""Data caching layer for the PARA scoring pipeline.

Stores fetched raw data as parquet or JSON files with timestamps to avoid
redundant J-Quants API calls.  Each cache entry tracks when it was last
refreshed and supports configurable TTLs.

Cache directory layout::

    {DATA_CACHE_DIR}/
        listed_companies.parquet    # TTL: 7 days
        financials.parquet          # TTL: 7 days
        daily_quotes.parquet        # TTL: 1 day (incremental append)
        sector_signals.json         # TTL: 7 days
        _metadata.json              # Per-file timestamps

The ``--force-refresh`` pipeline flag bypasses all caches.

Usage::

    cache = DataCache()                 # uses DATA_CACHE_DIR env var or ./cache
    cache = DataCache(force_refresh=True)  # ignore all cached data

    # Store / retrieve DataFrames
    cache.save_parquet("listed_companies", df)
    df = cache.load_parquet("listed_companies", max_age_days=7)

    # Incremental daily quotes
    cache.append_parquet("daily_quotes", new_df, date_col="Date")
    df = cache.load_parquet("daily_quotes", max_age_days=1)
    last = cache.get_last_date("daily_quotes", date_col="Date")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "./cache"

# Default TTLs (days) for each cache entry.
CACHE_TTLS: dict[str, int] = {
    "listed_companies": 7,
    "financials": 7,
    "daily_quotes": 1,
    "sector_signals": 7,
}


class DataCache:
    """File-based cache for pipeline data.

    Parameters
    ----------
    cache_dir : str, optional
        Path to the cache directory.  Falls back to ``DATA_CACHE_DIR``
        env var, then ``./cache``.
    force_refresh : bool
        If ``True``, all cache reads return ``None`` (forcing re-fetch).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        force_refresh: bool = False,
    ) -> None:
        self.cache_dir = Path(
            cache_dir or os.getenv("DATA_CACHE_DIR", DEFAULT_CACHE_DIR)
        )
        self.force_refresh = force_refresh
        self._metadata_path = self.cache_dir / "_metadata.json"
        self._metadata: dict[str, str] = {}

        # Create cache directory if it doesn't exist.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()

        if self.force_refresh:
            logger.info("Cache: force-refresh mode — all cached data will be ignored.")
        else:
            logger.info("Cache directory: %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Load per-file timestamps from _metadata.json."""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r") as fh:
                    self._metadata = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache metadata corrupted, resetting: %s", exc)
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Persist per-file timestamps to _metadata.json."""
        with open(self._metadata_path, "w") as fh:
            json.dump(self._metadata, fh, indent=2)

    def _set_timestamp(self, key: str) -> None:
        """Record the current UTC time as the refresh timestamp for *key*."""
        self._metadata[key] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

    def _get_timestamp(self, key: str) -> Optional[datetime]:
        """Return the last-refresh timestamp for *key*, or ``None``."""
        ts = self._metadata.get(key)
        if ts is None:
            return None
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None

    def is_stale(self, key: str, max_age_days: Optional[int] = None) -> bool:
        """Check whether the cached entry for *key* is stale.

        Returns ``True`` if the cache should be refreshed — either because
        ``force_refresh`` is set, the entry doesn't exist, or its age
        exceeds *max_age_days*.

        Args:
            key: Cache entry name (e.g. ``"listed_companies"``).
            max_age_days: Maximum age in days.  Defaults to the value in
                :data:`CACHE_TTLS` for the key, or 7 days.
        """
        if self.force_refresh:
            return True

        ts = self._get_timestamp(key)
        if ts is None:
            return True

        if max_age_days is None:
            max_age_days = CACHE_TTLS.get(key, 7)

        age = datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)
        return age > timedelta(days=max_age_days)

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _parquet_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def save_parquet(self, key: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to parquet and update the refresh timestamp.

        Args:
            key: Cache entry name.
            df: DataFrame to store.
        """
        path = self._parquet_path(key)
        df.to_parquet(path, index=False)
        self._set_timestamp(key)
        logger.info("Cache: saved %s (%d rows) to %s", key, len(df), path)

    def load_parquet(
        self,
        key: str,
        max_age_days: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Load a cached parquet file if it exists and is fresh.

        Args:
            key: Cache entry name.
            max_age_days: Maximum age in days before the cache is
                considered stale.

        Returns:
            The cached DataFrame, or ``None`` if stale / missing.
        """
        if self.is_stale(key, max_age_days):
            reason = "force-refresh" if self.force_refresh else "stale/missing"
            logger.info("Cache miss (%s): %s", reason, key)
            return None

        path = self._parquet_path(key)
        if not path.exists():
            logger.info("Cache miss (file missing): %s", key)
            return None

        df = pd.read_parquet(path)
        logger.info("Cache hit: %s (%d rows)", key, len(df))
        return df

    def append_parquet(
        self,
        key: str,
        new_df: pd.DataFrame,
        date_col: str = "Date",
    ) -> pd.DataFrame:
        """Append new rows to an existing parquet cache, deduplicating by date+code.

        Used for incremental daily quote updates: only new dates are
        fetched from the API and appended to the existing cache.

        Args:
            key: Cache entry name.
            new_df: New rows to append.
            date_col: Column name containing the date for deduplication.

        Returns:
            The merged DataFrame (existing + new, deduplicated).
        """
        path = self._parquet_path(key)

        if path.exists() and not self.force_refresh:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)

            # Deduplicate: keep last occurrence per (Code, Date).
            dedup_cols = ["Code", date_col] if "Code" in combined.columns else [date_col]
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

            if date_col in combined.columns:
                combined[date_col] = pd.to_datetime(combined[date_col])
                combined.sort_values(
                    dedup_cols, inplace=True
                )
                combined.reset_index(drop=True, inplace=True)

            logger.info(
                "Cache: appended %d new rows to %s (total: %d rows)",
                len(new_df), key, len(combined),
            )
        else:
            combined = new_df.copy()
            logger.info("Cache: created %s with %d rows", key, len(combined))

        combined.to_parquet(path, index=False)
        self._set_timestamp(key)
        return combined

    def get_last_date(
        self,
        key: str,
        date_col: str = "Date",
    ) -> Optional[str]:
        """Return the most recent date in a cached parquet file.

        Used for incremental daily quote fetching — only fetch dates
        after the last cached date.

        Args:
            key: Cache entry name.
            date_col: Column containing dates.

        Returns:
            Date string (``YYYY-MM-DD``) or ``None`` if no cache exists.
        """
        if self.force_refresh:
            return None

        path = self._parquet_path(key)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path, columns=[date_col])
            if df.empty or date_col not in df.columns:
                return None
            last = pd.to_datetime(df[date_col]).max()
            if pd.isna(last):
                return None
            return last.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning("Could not read last date from %s: %s", key, exc)
            return None

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def save_json(self, key: str, data: Any) -> None:
        """Save data as JSON and update the refresh timestamp.

        Args:
            key: Cache entry name.
            data: JSON-serialisable data.
        """
        path = self._json_path(key)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        self._set_timestamp(key)
        logger.info("Cache: saved %s to %s", key, path)

    def load_json(
        self,
        key: str,
        max_age_days: Optional[int] = None,
    ) -> Optional[Any]:
        """Load cached JSON data if it exists and is fresh.

        Args:
            key: Cache entry name.
            max_age_days: Maximum age in days.

        Returns:
            Parsed JSON data, or ``None`` if stale / missing.
        """
        if self.is_stale(key, max_age_days):
            reason = "force-refresh" if self.force_refresh else "stale/missing"
            logger.info("Cache miss (%s): %s", reason, key)
            return None

        path = self._json_path(key)
        if not path.exists():
            logger.info("Cache miss (file missing): %s", key)
            return None

        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            logger.info("Cache hit: %s", key)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read error for %s: %s", key, exc)
            return None

"""Tests for the data caching layer (src/data/cache.py).

Validates cache hit/miss behaviour, TTL expiration, incremental append
logic, and force-refresh bypass.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache import DataCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> str:
    """Provide a temporary cache directory for each test."""
    return str(tmp_path / "test_cache")


class TestDataCacheParquet:
    """Parquet cache read/write and TTL behaviour."""

    def test_save_and_load(self, cache_dir: str) -> None:
        """Saved parquet can be loaded back."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({"Code": ["7203", "6758"], "Value": [1.0, 2.0]})
        cache.save_parquet("test_data", df)

        loaded = cache.load_parquet("test_data", max_age_days=1)
        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded["Code"]) == ["7203", "6758"]

    def test_cache_miss_when_not_saved(self, cache_dir: str) -> None:
        """Load returns None when no cache file exists."""
        cache = DataCache(cache_dir=cache_dir)
        result = cache.load_parquet("nonexistent")
        assert result is None

    def test_cache_stale_after_ttl(self, cache_dir: str) -> None:
        """Cache is stale when timestamp exceeds max_age_days."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({"Code": ["7203"], "Value": [1.0]})
        cache.save_parquet("test_data", df)

        # Manually backdate the timestamp by 8 days.
        old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        cache._metadata["test_data"] = old_ts
        cache._save_metadata()

        assert cache.is_stale("test_data", max_age_days=7)
        result = cache.load_parquet("test_data", max_age_days=7)
        assert result is None

    def test_cache_fresh_within_ttl(self, cache_dir: str) -> None:
        """Cache is not stale when within max_age_days."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({"Code": ["7203"], "Value": [1.0]})
        cache.save_parquet("test_data", df)

        assert not cache.is_stale("test_data", max_age_days=7)
        result = cache.load_parquet("test_data", max_age_days=7)
        assert result is not None

    def test_force_refresh_bypasses_cache(self, cache_dir: str) -> None:
        """Force-refresh mode always returns None from load."""
        cache = DataCache(cache_dir=cache_dir, force_refresh=False)
        df = pd.DataFrame({"Code": ["7203"], "Value": [1.0]})
        cache.save_parquet("test_data", df)

        # Reopen with force_refresh=True.
        cache_fr = DataCache(cache_dir=cache_dir, force_refresh=True)
        assert cache_fr.is_stale("test_data")
        result = cache_fr.load_parquet("test_data")
        assert result is None

    def test_default_ttl_from_cache_ttls(self, cache_dir: str) -> None:
        """is_stale uses CACHE_TTLS defaults when max_age_days not specified."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({"Code": ["7203"], "Value": [1.0]})
        cache.save_parquet("listed_companies", df)

        # Just saved — should not be stale (default TTL is 7 days).
        assert not cache.is_stale("listed_companies")

        # Backdate by 8 days — should be stale.
        old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        cache._metadata["listed_companies"] = old_ts
        cache._save_metadata()
        assert cache.is_stale("listed_companies")


class TestDataCacheAppend:
    """Incremental append logic for daily quotes."""

    def test_append_creates_new(self, cache_dir: str) -> None:
        """Append to non-existent cache creates a new file."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({
            "Code": ["7203", "7203"],
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Close": [2400.0, 2410.0],
        })
        result = cache.append_parquet("daily_quotes", df, date_col="Date")
        assert len(result) == 2

    def test_append_deduplicates(self, cache_dir: str) -> None:
        """Append deduplicates by (Code, Date), keeping the latest."""
        cache = DataCache(cache_dir=cache_dir)

        # Initial data.
        df1 = pd.DataFrame({
            "Code": ["7203", "7203"],
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Close": [2400.0, 2410.0],
        })
        cache.append_parquet("daily_quotes", df1, date_col="Date")

        # Append overlapping + new data.
        df2 = pd.DataFrame({
            "Code": ["7203", "7203"],
            "Date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "Close": [2415.0, 2420.0],  # Updated close for 01-02
        })
        result = cache.append_parquet("daily_quotes", df2, date_col="Date")

        assert len(result) == 3  # 01-01, 01-02 (updated), 01-03
        row_0102 = result[result["Date"] == pd.Timestamp("2026-01-02")]
        assert row_0102["Close"].iloc[0] == 2415.0  # Updated value

    def test_get_last_date(self, cache_dir: str) -> None:
        """get_last_date returns the most recent date in cache."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({
            "Code": ["7203", "6758"],
            "Date": pd.to_datetime(["2026-01-15", "2026-01-20"]),
            "Close": [2400.0, 3000.0],
        })
        cache.save_parquet("daily_quotes", df)

        last = cache.get_last_date("daily_quotes", date_col="Date")
        assert last == "2026-01-20"

    def test_get_last_date_empty_cache(self, cache_dir: str) -> None:
        """get_last_date returns None when no cache exists."""
        cache = DataCache(cache_dir=cache_dir)
        assert cache.get_last_date("daily_quotes") is None

    def test_get_last_date_force_refresh(self, cache_dir: str) -> None:
        """get_last_date returns None when force_refresh is set."""
        cache = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({
            "Code": ["7203"],
            "Date": pd.to_datetime(["2026-01-15"]),
            "Close": [2400.0],
        })
        cache.save_parquet("daily_quotes", df)

        cache_fr = DataCache(cache_dir=cache_dir, force_refresh=True)
        assert cache_fr.get_last_date("daily_quotes") is None


class TestDataCacheJSON:
    """JSON cache read/write."""

    def test_save_and_load_json(self, cache_dir: str) -> None:
        """Saved JSON can be loaded back."""
        cache = DataCache(cache_dir=cache_dir)
        data = {"TOPIX17_1": 0.5, "TOPIX17_2": -0.3}
        cache.save_json("sector_signals", data)

        loaded = cache.load_json("sector_signals", max_age_days=7)
        assert loaded is not None
        assert loaded["TOPIX17_1"] == 0.5
        assert loaded["TOPIX17_2"] == -0.3

    def test_json_cache_miss(self, cache_dir: str) -> None:
        """Load returns None when no JSON cache exists."""
        cache = DataCache(cache_dir=cache_dir)
        assert cache.load_json("nonexistent") is None

    def test_json_stale_after_ttl(self, cache_dir: str) -> None:
        """JSON cache is stale when timestamp exceeds max_age_days."""
        cache = DataCache(cache_dir=cache_dir)
        cache.save_json("sector_signals", {"a": 1})

        old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        cache._metadata["sector_signals"] = old_ts
        cache._save_metadata()

        result = cache.load_json("sector_signals", max_age_days=7)
        assert result is None


class TestDataCacheMetadata:
    """Metadata persistence across cache instances."""

    def test_metadata_persists(self, cache_dir: str) -> None:
        """Timestamps survive cache re-initialisation."""
        cache1 = DataCache(cache_dir=cache_dir)
        df = pd.DataFrame({"Code": ["7203"], "Value": [1.0]})
        cache1.save_parquet("test_data", df)

        # Re-open the cache from the same directory.
        cache2 = DataCache(cache_dir=cache_dir)
        assert not cache2.is_stale("test_data", max_age_days=1)
        loaded = cache2.load_parquet("test_data", max_age_days=1)
        assert loaded is not None

    def test_corrupted_metadata_resets(self, cache_dir: str) -> None:
        """Corrupted _metadata.json is handled gracefully."""
        cache = DataCache(cache_dir=cache_dir)
        # Write garbage to metadata file.
        meta_path = Path(cache_dir) / "_metadata.json"
        meta_path.write_text("not valid json {{{")

        # Re-open — should reset metadata without crashing.
        cache2 = DataCache(cache_dir=cache_dir)
        assert cache2._metadata == {}

    def test_cache_dir_created_automatically(self, tmp_path: Path) -> None:
        """Cache directory is created if it doesn't exist."""
        new_dir = str(tmp_path / "deep" / "nested" / "cache")
        cache = DataCache(cache_dir=new_dir)
        assert Path(new_dir).exists()

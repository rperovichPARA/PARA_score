"""J-Quants API client and data retrieval.

Handles authentication (refresh token -> ID token) and data fetching
for listed companies, financial statements, daily quotes, and TOPIX index.
"""

import logging
import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.jquants.com/v1"


class JQuantsClient:
    """Client for the J-Quants API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("JQUANTS_API_KEY", "")
        self._id_token: Optional[str] = None

    def _get_id_token(self) -> str:
        """Exchange refresh token for an ID token."""
        if self._id_token:
            return self._id_token
        resp = requests.post(
            f"{BASE_URL}/token/auth_refresh",
            params={"refreshtoken": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        self._id_token = resp.json()["idToken"]
        return self._id_token

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Authenticated GET request."""
        token = self._get_id_token()
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=headers,
            params=params or {},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def get_listed_companies(self) -> pd.DataFrame:
        """Fetch the full universe of listed companies."""
        data = self._get("/listed/info")
        return pd.DataFrame(data.get("info", []))

    def get_financial_statements(self, code: str) -> pd.DataFrame:
        """Fetch financial statement summaries for a company."""
        data = self._get("/fins/statements", params={"code": code})
        return pd.DataFrame(data.get("statements", []))

    def get_daily_quotes(self, code: str, date_from: str, date_to: str) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a company."""
        data = self._get(
            "/prices/daily_quotes",
            params={"code": code, "from": date_from, "to": date_to},
        )
        return pd.DataFrame(data.get("daily_quotes", []))

    def get_topix_index(self, date_from: str, date_to: str) -> pd.DataFrame:
        """Fetch TOPIX index values."""
        data = self._get(
            "/indices/topix",
            params={"from": date_from, "to": date_to},
        )
        return pd.DataFrame(data.get("topix", []))

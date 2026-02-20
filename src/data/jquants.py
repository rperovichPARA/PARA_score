"""J-Quants API V2 client and data retrieval.

Handles authentication and data fetching for listed companies, financial
statements, daily quotes, and TOPIX index.

Authentication:
    Store your J-Quants API key as JQUANTS_API_KEY in .env.
    The V2 API uses a simple ``x-api-key`` header — no token exchange needed.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.jquants.com/v2"

# Retry configuration for transient HTTP errors.
_MAX_RETRIES = 4
_RETRY_BACKOFF_BASE = 2  # seconds; doubles each attempt
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# V2 → V1 column name mappings
# ---------------------------------------------------------------------------
# The V2 API uses abbreviated column names.  We rename them back to the
# V1 full names so all downstream scoring code works unchanged.

_V2_LISTED_COLUMNS: dict[str, str] = {
    "CoName": "CompanyName",
    "CoNameEn": "CompanyNameEnglish",
    "S17": "Sector17Code",
    "S17Nm": "Sector17CodeName",
    "S33": "Sector33Code",
    "S33Nm": "Sector33CodeName",
    "ScaleCat": "ScaleCategory",
    "Mkt": "MarketCode",
    "MktNm": "MarketCodeName",
    "Mrgn": "MarginCode",
    "MrgnNm": "MarginCodeName",
}

_V2_FINS_COLUMNS: dict[str, str] = {
    "DiscDate": "DisclosedDate",
    "DiscTime": "DisclosedTime",
    "DiscNo": "DisclosureNumber",
    "DocType": "TypeOfDocument",
    "CurPerType": "TypeOfCurrentPeriod",
    "CurPerSt": "CurrentPeriodStartDate",
    "CurPerEn": "CurrentPeriodEndDate",
    "CurFYSt": "CurrentFiscalYearStartDate",
    "CurFYEn": "CurrentFiscalYearEndDate",
    "NxtFYSt": "NextFiscalYearStartDate",
    "NxtFYEn": "NextFiscalYearEndDate",
    "Sales": "NetSales",
    "OP": "OperatingProfit",
    "OdP": "OrdinaryProfit",
    "NP": "Profit",
    "EPS": "EarningsPerShare",
    "DEPS": "DilutedEarningsPerShare",
    "TA": "TotalAssets",
    "Eq": "Equity",
    "EqAR": "EquityToAssetRatio",
    "BPS": "BookValuePerShare",
    "CFO": "CashFlowsFromOperatingActivities",
    "CFI": "CashFlowsFromInvestingActivities",
    "CFF": "CashFlowsFromFinancingActivities",
    "CashEq": "CashAndEquivalents",
    "DivAnn": "ResultDividendPerShareAnnual",
    "PayoutRatioAnn": "ResultPayoutRatioAnnual",
    "FDivAnn": "ForecastDividendPerShareAnnual",
    "FPayoutRatioAnn": "ForecastPayoutRatioAnnual",
    "NxFDivAnn": "NextYearForecastDividendPerShareAnnual",
    "NxFPayoutRatioAnn": "NextYearForecastPayoutRatioAnnual",
    "FSales": "ForecastNetSales",
    "FOP": "ForecastOperatingProfit",
    "FOdP": "ForecastOrdinaryProfit",
    "FNP": "ForecastProfit",
    "FEPS": "ForecastEarningsPerShare",
    "NxFSales": "NextYearForecastNetSales",
    "NxFOP": "NextYearForecastOperatingProfit",
    "NxFOdP": "NextYearForecastOrdinaryProfit",
    "NxFNp": "NextYearForecastProfit",
    "NxFEPS": "NextYearForecastEarningsPerShare",
    "ShOutFY": "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
    "TrShFY": "TreasurySharesFiscalYear",
    "AvgSh": "AverageShares",
}

_V2_DAILY_COLUMNS: dict[str, str] = {
    "O": "Open",
    "H": "High",
    "L": "Low",
    "C": "Close",
    "Vo": "Volume",
    "Va": "TurnoverValue",
    "AdjFactor": "AdjustmentFactor",
    "AdjO": "AdjustmentOpen",
    "AdjH": "AdjustmentHigh",
    "AdjL": "AdjustmentLow",
    "AdjC": "AdjustmentClose",
    "AdjVo": "AdjustmentVolume",
}

_V2_TOPIX_COLUMNS: dict[str, str] = {
    "O": "Open",
    "H": "High",
    "L": "Low",
    "C": "Close",
}


def _rename_v2_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename V2 abbreviated columns to V1 full names (in place, if present)."""
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    if rename:
        df.rename(columns=rename, inplace=True)
        logger.debug("Renamed V2 columns: %s", list(rename.keys()))
    return df


class JQuantsError(Exception):
    """Raised for J-Quants API errors."""


class JQuantsClient:
    """Client for the J-Quants API V2 (api.jquants.com/v2).

    Parameters
    ----------
    api_key : str, optional
        J-Quants API key.  Falls back to ``JQUANTS_API_KEY`` env var.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("JQUANTS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No J-Quants API key provided. Set JQUANTS_API_KEY in .env "
                "or pass api_key to JQuantsClient()."
            )

    # ── low-level HTTP ───────────────────────────────────────────────

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Authenticated GET with retry on transient errors."""
        headers = {"x-api-key": self.api_key}
        url = f"{BASE_URL}{endpoint}"
        merged_params = params or {}

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=merged_params,
                    timeout=60,
                )
            except requests.ConnectionError as exc:
                if attempt == _MAX_RETRIES:
                    raise JQuantsError(
                        f"Connection failed after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Connection error on %s (attempt %d/%d), retrying in %ds.",
                    endpoint, attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "HTTP %d on %s (attempt %d/%d), retrying in %ds.",
                    resp.status_code, endpoint, attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise JQuantsError(
                    f"GET {endpoint} failed (HTTP {resp.status_code}): {resp.text}"
                )

            return resp.json()

        raise JQuantsError(f"GET {endpoint} failed after {_MAX_RETRIES} retries.")

    def _get_paginated(
        self, endpoint: str, result_key: str, params: Optional[dict] = None
    ) -> list[dict]:
        """Fetch all pages for a paginated endpoint.

        J-Quants signals more pages via a ``pagination_key`` field in the
        response body.  We keep requesting until no key is returned.
        """
        all_records: list[dict] = []
        merged_params = dict(params or {})
        page = 0

        while True:
            page += 1
            data = self._get(endpoint, params=merged_params)
            records = data.get(result_key, [])
            all_records.extend(records)
            logger.debug(
                "%s page %d: %d records (total so far: %d).",
                endpoint, page, len(records), len(all_records),
            )

            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
            merged_params["pagination_key"] = pagination_key

        return all_records

    # ── public API methods ───────────────────────────────────────────

    def get_listed_companies(
        self, code: Optional[str] = None, date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch listed company info.

        Parameters
        ----------
        code : str, optional
            Filter by security code (e.g. ``"72030"``).
        date : str, optional
            Filter by listing date (``YYYY-MM-DD``).

        Returns
        -------
        pd.DataFrame
            Columns include ``Code``, ``CompanyName``, ``CompanyNameEnglish``,
            ``Sector17Code``, ``Sector33Code``, ``MarketCode``, etc.
        """
        params: dict = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date

        records = self._get_paginated("/equities/master", "data", params)
        df = pd.DataFrame(records)
        _rename_v2_columns(df, _V2_LISTED_COLUMNS)
        logger.info("Listed companies fetched: %d rows, columns: %s", len(df), list(df.columns))
        return df

    def get_financial_statements(
        self,
        code: Optional[str] = None,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch financial statement summaries.

        Provide at least ``code`` or ``date`` to scope the query.  Omitting
        both returns the full universe for the latest disclosure date, which
        may be very large.

        Parameters
        ----------
        code : str, optional
            Security code (e.g. ``"72030"``).
        date : str, optional
            Disclosure date filter (``YYYY-MM-DD``).

        Returns
        -------
        pd.DataFrame
            Columns from J-Quants ``/fins/summary`` including revenue,
            operating profit, forecasts, per-share data, etc.
        """
        params: dict = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date

        records = self._get_paginated("/fins/summary", "data", params)
        df = pd.DataFrame(records)
        _rename_v2_columns(df, _V2_FINS_COLUMNS)

        # Convert numeric columns that J-Quants returns as strings.
        numeric_hints = [
            "NetSales", "OperatingProfit", "OrdinaryProfit", "Profit",
            "EarningsPerShare", "BookValuePerShare", "Equity",
            "EquityToAssetRatio", "TotalAssets",
            "ForecastNetSales", "ForecastOperatingProfit",
            "ForecastOrdinaryProfit", "ForecastProfit",
            "ForecastEarningsPerShare",
            "NextYearForecastNetSales", "NextYearForecastOperatingProfit",
            "NextYearForecastOrdinaryProfit", "NextYearForecastProfit",
            "NextYearForecastEarningsPerShare",
            "ResultDividendPerShareAnnual", "ResultPayoutRatioAnnual",
            "ForecastDividendPerShareAnnual", "ForecastPayoutRatioAnnual",
            "NextYearForecastDividendPerShareAnnual",
            "NextYearForecastPayoutRatioAnnual",
            "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
        ]
        for col in numeric_hints:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Financial statements fetched: %d rows.", len(df))
        return df

    def get_daily_quotes(
        self,
        code: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars.

        Parameters
        ----------
        code : str, optional
            Security code.
        date_from : str, optional
            Start date (``YYYY-MM-DD``), inclusive.
        date_to : str, optional
            End date (``YYYY-MM-DD``), inclusive.
        date : str, optional
            Single date (``YYYY-MM-DD``).  Use instead of from/to for a
            cross-sectional snapshot.

        Returns
        -------
        pd.DataFrame
            Columns include ``Date``, ``Code``, ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume``, ``AdjustmentClose``, etc.
        """
        params: dict = {}
        if code:
            params["code"] = code
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        if date:
            params["date"] = date

        records = self._get_paginated(
            "/equities/bars/daily", "data", params
        )
        df = pd.DataFrame(records)
        _rename_v2_columns(df, _V2_DAILY_COLUMNS)

        if not df.empty:
            price_cols = [
                "Open", "High", "Low", "Close", "AdjustmentOpen",
                "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose",
                "Volume", "TurnoverValue",
            ]
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.sort_values(["Code", "Date"], inplace=True)
                df.reset_index(drop=True, inplace=True)

        logger.info("Daily quotes fetched: %d rows.", len(df))
        return df

    def get_topix_index(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch TOPIX index daily values.

        Parameters
        ----------
        date_from : str, optional
            Start date (``YYYY-MM-DD``), inclusive.
        date_to : str, optional
            End date (``YYYY-MM-DD``), inclusive.

        Returns
        -------
        pd.DataFrame
            Columns include ``Date``, ``Close`` (TOPIX level), ``Open``,
            ``High``, ``Low``.
        """
        params: dict = {}
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to

        records = self._get_paginated("/indices/bars/daily/topix", "data", params)
        df = pd.DataFrame(records)
        _rename_v2_columns(df, _V2_TOPIX_COLUMNS)

        if not df.empty:
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.sort_values("Date", inplace=True)
                df.reset_index(drop=True, inplace=True)

        logger.info("TOPIX index fetched: %d rows.", len(df))
        return df

    # ── bulk / convenience helpers ───────────────────────────────────

    def get_all_financial_statements(self, date: str) -> pd.DataFrame:
        """Fetch financial statements for the entire universe on a date.

        This is a convenience wrapper around :meth:`get_financial_statements`
        that queries by date rather than by code, returning all companies
        that disclosed on ``date``.

        Parameters
        ----------
        date : str
            Disclosure date (``YYYY-MM-DD``).
        """
        return self.get_financial_statements(date=date)

    def get_universe_quotes(
        self,
        date: str,
    ) -> pd.DataFrame:
        """Fetch daily quotes for the full universe on a single date.

        Parameters
        ----------
        date : str
            Trading date (``YYYY-MM-DD``).
        """
        return self.get_daily_quotes(date=date)

    def get_price_history(
        self,
        code: str,
        months: int = 6,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch trailing price history for a single stock.

        Parameters
        ----------
        code : str
            Security code.
        months : int
            Number of months of history (default 6).
        end_date : str, optional
            End date (``YYYY-MM-DD``).  Defaults to today.

        Returns
        -------
        pd.DataFrame
            Daily OHLCV sorted by date.
        """
        end = (
            datetime.strptime(end_date, "%Y-%m-%d")
            if end_date
            else datetime.utcnow()
        )
        start = end - timedelta(days=months * 31)
        return self.get_daily_quotes(
            code=code,
            date_from=start.strftime("%Y-%m-%d"),
            date_to=end.strftime("%Y-%m-%d"),
        )

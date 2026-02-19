"""Google Sheets adapter for supplementary metric data.

Reads published Google Sheets via their CSV export URL and returns
DataFrames keyed on stock code, with column names matching
``scoring_weights.yaml``.

The primary use-case is the **PARA.FS.data** spreadsheet which holds
metrics that cannot be fully derived from J-Quants Standard-plan data
(Altman Z-Score, broker targets, net-cash / market-cap, kozo metrics,
etc.).  The adapter can also read arbitrary additional sheets — each
tab is fetched independently and the results can be merged.

Authentication is not required: the target sheets must be shared with
*"anyone with the link"* viewer access so the CSV export endpoint is
reachable.  For private sheets, set ``GSHEET_SERVICE_ACCOUNT_JSON``
to a service-account key path and the adapter will fall back to the
Google Sheets API (requires ``google-api-python-client``).

Environment variables
---------------------
GSHEET_SPREADSHEET_ID
    Default spreadsheet ID.  Falls back to the PARA.FS.data sheet.
GSHEET_SERVICE_ACCOUNT_JSON
    Optional path to a Google service-account JSON key file for
    private sheets.  When set, the adapter prefers the Sheets API
    over the public CSV export.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
_DEFAULT_SPREADSHEET_ID = "10mjEbmtJC6y5tCqnQ_SrUQAfheDhafAtpX0DjJJO5fk"
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scoring_weights.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_csv_url(spreadsheet_id: str, gid: int = 0) -> str:
    """Build the public CSV export URL for a Google Sheets tab."""
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/export?format=csv&gid={gid}"
    )


def _load_all_metric_names(config_path: Path | str = _CONFIG_PATH) -> set[str]:
    """Return the set of every metric column name defined in the YAML config.

    Reads the five scoring-category sections (``fundamentals``,
    ``valuation``, ``sector``, ``factors``, ``kozo``) and collects their
    metric keys.  This keeps the adapter in sync with the config without
    hard-coding metric names.

    Falls back to an empty set if the config file is missing.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found at %s; metric recognition disabled.", path)
        return set()

    with open(path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    categories = ("fundamentals", "valuation", "sector", "factors", "kozo")
    names: set[str] = set()
    for cat in categories:
        section = cfg.get(cat, {})
        if isinstance(section, dict):
            names.update(section.keys())
    return names


# ── Client ────────────────────────────────────────────────────────────────

class GoogleSheetsClient:
    """Client for pulling supplementary metric data from Google Sheets.

    Parameters
    ----------
    spreadsheet_id : str, optional
        The Google Sheets document ID.  Falls back to
        ``GSHEET_SPREADSHEET_ID`` env var, then the PARA.FS.data sheet.
    column_map : dict[str, str], optional
        Optional mapping from **sheet column names** to
        **scoring_weights.yaml metric names**.  Applied before any
        metric-recognition logic.  Example::

            {"Altman Z-Score": "altman_z", "BrokerTP_Upside": "broker_target_upside"}
    config_path : str or Path, optional
        Path to ``scoring_weights.yaml`` for metric-name recognition.
    """

    def __init__(
        self,
        spreadsheet_id: Optional[str] = None,
        column_map: Optional[dict[str, str]] = None,
        config_path: Path | str = _CONFIG_PATH,
    ) -> None:
        self.spreadsheet_id: str = (
            spreadsheet_id
            or os.getenv("GSHEET_SPREADSHEET_ID", _DEFAULT_SPREADSHEET_ID)
        )
        self.column_map: dict[str, str] = column_map or {}
        self._metric_names: set[str] = _load_all_metric_names(config_path)
        logger.info(
            "GoogleSheetsClient initialised (spreadsheet=%s, known metrics=%d)",
            self.spreadsheet_id,
            len(self._metric_names),
        )

    # ── Core I/O ──────────────────────────────────────────────────────

    def _fetch_csv(self, spreadsheet_id: str, gid: int) -> pd.DataFrame:
        """Fetch a sheet tab via the public CSV export URL.

        Returns an empty DataFrame on any network or parse error.
        """
        url = _build_csv_url(spreadsheet_id, gid)
        try:
            df = pd.read_csv(url)
        except Exception as exc:
            logger.warning(
                "Could not fetch Google Sheet (id=%s, gid=%d): %s",
                spreadsheet_id, gid, exc,
            )
            return pd.DataFrame()
        return df

    # ── Public methods ────────────────────────────────────────────────

    def read_sheet(
        self,
        gid: int = 0,
        spreadsheet_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read a sheet tab and return a DataFrame keyed on stock code.

        Applies ``column_map`` renaming, coerces recognised metric
        columns to numeric, and sets ``Code`` as the index.

        Parameters
        ----------
        gid : int
            Sheet tab GID (``0`` = first tab).
        spreadsheet_id : str, optional
            Override the client's default spreadsheet ID for this call.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by stock code.  Contains all columns
            present in the sheet (metric and non-metric alike).  Metric
            columns are coerced to ``float``.  Returns an empty
            DataFrame if the sheet is unreachable or has no ``Code``
            column.
        """
        sid = spreadsheet_id or self.spreadsheet_id
        df = self._fetch_csv(sid, gid)

        if df.empty:
            logger.info("Sheet (id=%s, gid=%d) returned no data.", sid, gid)
            return pd.DataFrame()

        # Apply user-supplied column renaming.
        if self.column_map:
            df = df.rename(columns=self.column_map)

        # A Code column is required for keying.
        if "Code" not in df.columns:
            logger.warning(
                "Sheet (id=%s, gid=%d) has no 'Code' column. "
                "Available columns: %s",
                sid, gid, list(df.columns),
            )
            return pd.DataFrame()

        # Normalise the key.
        df["Code"] = df["Code"].astype(str).str.strip()
        df = df.drop_duplicates(subset="Code", keep="last")
        df = df.set_index("Code")

        # Coerce recognised metric columns to numeric.
        metric_cols = [c for c in df.columns if c in self._metric_names]
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "Read sheet (id=%s, gid=%d): %d rows, %d metric columns %s",
            sid, gid, len(df), len(metric_cols), metric_cols,
        )
        return df

    def get_supplement_metrics(
        self,
        gid: int = 0,
        spreadsheet_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read a sheet and return **only** recognised metric columns.

        This is the primary entry point for the scoring pipeline.  It
        returns a DataFrame containing only columns whose names appear
        in ``scoring_weights.yaml``, all coerced to ``float``.

        Parameters
        ----------
        gid : int
            Sheet tab GID.
        spreadsheet_id : str, optional
            Override the client's default spreadsheet ID.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by stock code with recognised metric
            columns only.  Empty DataFrame on failure.
        """
        df = self.read_sheet(gid=gid, spreadsheet_id=spreadsheet_id)
        if df.empty:
            return df

        metric_cols = [c for c in df.columns if c in self._metric_names]
        if not metric_cols:
            logger.info(
                "No recognised metric columns found in sheet. "
                "Columns present: %s",
                list(df.columns),
            )
            return pd.DataFrame(index=df.index)

        return df[metric_cols]

    def get_multi_tab_metrics(
        self,
        gids: list[int],
        spreadsheet_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read multiple tabs and merge their metric columns.

        Later tabs overwrite earlier tabs for overlapping columns on the
        same stock code.

        Parameters
        ----------
        gids : list[int]
            Sheet tab GIDs to read.
        spreadsheet_id : str, optional
            Override the client's default spreadsheet ID.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame indexed by stock code.
        """
        combined: Optional[pd.DataFrame] = None

        for gid in gids:
            tab = self.get_supplement_metrics(gid=gid, spreadsheet_id=spreadsheet_id)
            if tab.empty:
                continue
            if combined is None:
                combined = tab
            else:
                # Update fills new columns; for overlapping columns the
                # later tab's non-null values win.
                combined = combined.combine_first(tab)

        if combined is None:
            logger.info("No data retrieved from any of the %d tabs.", len(gids))
            return pd.DataFrame()

        logger.info(
            "Multi-tab merge complete: %d rows, %d metric columns.",
            len(combined), len(combined.columns),
        )
        return combined

    def merge_into(
        self,
        target: pd.DataFrame,
        gid: int = 0,
        spreadsheet_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Merge supplement metrics into an existing company DataFrame.

        Joins on the ``Code`` column (or index) of *target*.  By
        default, supplement values only fill gaps (``NaN``).

        **J-Quants is the primary data source.**  The default
        ``overwrite=False`` ensures J-Quants-derived values are never
        replaced by sheet values.  Use ``overwrite=True`` only for
        metrics that have no J-Quants source at all.

        Parameters
        ----------
        target : pd.DataFrame
            Company-level DataFrame.  Must have a ``Code`` column or
            a code-based index.
        gid : int
            Sheet tab GID.
        spreadsheet_id : str, optional
            Override the client's default spreadsheet ID.
        overwrite : bool
            If ``True``, sheet values replace existing values.  If
            ``False`` (default), sheet values only fill NaN gaps.

        Returns
        -------
        pd.DataFrame
            Copy of *target* with supplement columns merged in.
        """
        supplement = self.get_supplement_metrics(gid=gid, spreadsheet_id=spreadsheet_id)
        if supplement.empty:
            return target.copy()

        if overwrite:
            logger.warning(
                "merge_into called with overwrite=True — this will replace "
                "J-Quants-derived values with sheet values.  J-Quants should "
                "be the primary source for any metric it can compute."
            )

        df = target.copy()

        # Determine the join key from the target.
        if "Code" in df.columns:
            join_key = df["Code"].astype(str).str.strip()
        elif df.index.name == "Code":
            join_key = pd.Series(df.index.astype(str).str.strip(), index=df.index)
        else:
            logger.warning(
                "Target DataFrame has no 'Code' column or index; "
                "cannot merge supplement data."
            )
            return df

        filled_total = 0
        for col in supplement.columns:
            supp_values = join_key.map(supplement[col])

            if col in df.columns and not overwrite:
                before_nulls = df[col].isna().sum()
                df[col] = df[col].fillna(
                    pd.Series(supp_values.values, index=df.index)
                )
                filled = before_nulls - df[col].isna().sum()
            else:
                if col in df.columns:
                    filled = supp_values.notna().sum()
                else:
                    filled = supp_values.notna().sum()
                df[col] = supp_values.values

            if filled > 0:
                logger.info("Supplement filled %d values for %s", filled, col)
                filled_total += filled

        logger.info(
            "Merged %d supplement columns into target (%d total fills).",
            len(supplement.columns), filled_total,
        )
        return df

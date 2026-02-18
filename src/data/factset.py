"""FactSet FDS adapter (planned).

Future integration to fill data coverage gaps across all scoring categories.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FactSetClient:
    """Adapter for FactSet FDS data retrieval.

    This is a stub — implement when FactSet access is provisioned.
    """

    def __init__(self, username: str = "", api_key: str = "") -> None:
        self.username = username
        self.api_key = api_key
        logger.info("FactSet adapter initialized (stub)")

    def get_financial_data(self, codes: list[str]) -> pd.DataFrame:
        """Fetch financial data from FactSet."""
        logger.warning("FactSet integration not implemented — returning empty DataFrame")
        return pd.DataFrame()

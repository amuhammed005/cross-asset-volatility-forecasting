"""Handles storage and retrieval of asset data in a SQLite database."""

import sqlite3

import pandas as pd


class SQLRepository:
    """Handles all database transactions for asset data."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def insert_table(self, table_name: str, records: pd.DataFrame, if_exists: str = "fail") -> dict:
        """Insert a DataFrame into the database as a table.

        Returns
        -------
        dict
            Keys: "transaction_successful" (bool), "records_inserted" (int).
        """
        n_inserted = records.to_sql(
            name=table_name, con=self.connection, if_exists=if_exists
        )
        return {
            "transaction_successful": True,
            "records_inserted": n_inserted,
        }

    def read_table(self, table_name: str, limit: int = None) -> pd.DataFrame:
        """Read a table from the database.

        Returns
        -------
        pd.DataFrame
            Index is a DatetimeIndex named "date". Columns are
            "open", "high", "low", "close", "volume".
        """
        if limit:
            sql = f'SELECT * FROM "{table_name}" ORDER BY date DESC LIMIT {limit}'
        else:
            sql = f'SELECT * FROM "{table_name}"'

        df = pd.read_sql(sql=sql, con=self.connection, parse_dates=["date"], index_col="date")
        return df
import sqlite3

import pandas as pd
import pytest

from data.repository import SQLRepository


@pytest.fixture
def repo():
    connection = sqlite3.connect(":memory:")
    return SQLRepository(connection=connection)


def test_insert_and_read_table(repo):
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.5], "low": [0.9], "close": [1.2], "volume": [100.0]},
        index=pd.to_datetime(["2026-01-01"]),
    )
    df.index.name = "date"

    response = repo.insert_table(table_name="TEST", records=df, if_exists="replace")
    assert response["transaction_successful"] is True
    assert response["records_inserted"] == 1

    df_read = repo.read_table(table_name="TEST")
    assert df_read.shape == (1, 5)
    assert df_read.index.name == "date"
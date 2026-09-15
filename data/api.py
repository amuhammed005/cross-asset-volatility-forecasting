"""Handles data extraction from Yahoo Finance for equities and cryptocurrencies."""

import pandas as pd
import yfinance as yf


class MarketDataAPI:
    """Fetch daily OHLCV data for an equity or cryptocurrency ticker via Yahoo Finance."""

    def get_daily(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Get daily time series for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol. Works for equities (e.g. "AAPL") and
            cryptocurrencies (e.g. "BTC-USD").
        period : str, optional
            History length (yfinance period string). By default "5y".

        Returns
        -------
        pd.DataFrame
            Columns are 'open', 'high', 'low', 'close', 'volume'.
            Index is a DatetimeIndex named 'date'. All columns numeric.
        """
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)

        if raw.empty:
            raise Exception(f"No data returned for ticker: {ticker}")

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.astype(float)

        return df
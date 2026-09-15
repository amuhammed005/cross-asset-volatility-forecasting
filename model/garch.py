"""GARCH-family volatility modeling class."""

import os
from glob import glob

import joblib
import pandas as pd
from arch import arch_model

from config import settings


class GarchModel:
    """Fit, evaluate, and serve GARCH-family volatility models for a single asset."""

    def __init__(self, ticker: str, repo, use_new_data: bool):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations: int):
        """Extract returns for self.ticker from the repository.

        Parameters
        ----------
        n_observations : int
            Number of most recent return observations to use.
        """
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations + 1)
        df.sort_index(ascending=True, inplace=True)
        df["return"] = df["close"].pct_change() * 100
        self.data = df["return"].dropna()

    def fit(self, p: int = 1, q: int = 1, vol_model: str = "GARCH"):
        """Fit a GARCH-family model to self.data.

        Parameters
        ----------
        vol_model : str, optional
            One of "GARCH", "EGARCH", "GJR-GARCH". By default "GARCH".
        """
        if vol_model == "GARCH":
            spec = arch_model(self.data, p=p, q=q, vol="GARCH", rescale=False)
        elif vol_model == "EGARCH":
            spec = arch_model(self.data, p=p, o=1, q=q, vol="EGARCH", rescale=False)
        elif vol_model == "GJR-GARCH":
            spec = arch_model(self.data, p=p, o=1, q=q, vol="GARCH", rescale=False)
        else:
            raise ValueError(f"Unknown vol_model: {vol_model}")

        self.vol_model = vol_model
        self.model = spec.fit(disp=0)
        return self.model

    def __clean_prediction(self, prediction) -> dict:
        """Reformat a variance forecast DataFrame into a JSON-serializable dict."""
        start = prediction.index[0] + pd.DateOffset(days=1)
        prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])
        prediction_index = [d.isoformat() for d in prediction_dates]
        data = prediction.values.flatten() ** 0.5
        return pd.Series(data, index=prediction_index).to_dict()

    def predict_volatility(self, horizon: int) -> dict:
        """Generate a volatility forecast.

        Returns
        -------
        dict
            Keys are ISO 8601 dates, values are predicted volatility.
        """
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance
        return self.__clean_prediction(prediction)

    def dump(self) -> str:
        """Save self.model to self.model_directory. Returns the filepath."""
        timestamp = pd.Timestamp.now().isoformat().replace(":", "-")
        filepath = os.path.join(
            self.model_directory, f"{timestamp}_{self.ticker}_{self.vol_model}.pkl"
        )
        joblib.dump(self.model, filepath)
        return filepath

    def load(self):
        """Load the most recently saved model for self.ticker."""
        pattern = os.path.join(self.model_directory, f"*{self.ticker}*.pkl")
        try:
            model_path = sorted(glob(pattern))[-1]
        except IndexError:
            raise Exception(f"No model trained for '{self.ticker}'.")
        self.model = joblib.load(model_path)
        return self.model
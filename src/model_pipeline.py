import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


class AlternativeDataAlphaEngine:
    def __init__(self, ticker: str, wiki_page: str, start_date: str, end_date: str,
                 prices: pd.DataFrame = None, views: pd.DataFrame = None):
        """prices / views: optional already-loaded raw data (offline path, used by the tests).

        prices needs an 'Adj Close' column, views a 'views' column, both on a DatetimeIndex.
        When given, nothing is downloaded.
        """
        self.ticker = ticker
        self.wiki_page = wiki_page
        self.start_date = start_date
        self.end_date = end_date
        self._prices = prices
        self._views = views

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetches daily ETF market data using yfinance (unless prices were passed in)."""
        if self._prices is not None:
            df = self._prices.copy()
        else:
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Calculate log returns to ensure stationarity I(0)
        df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df['realized_volatility'] = df['log_return'].rolling(window=5).std()
        return df[['log_return', 'realized_volatility']].dropna()

    def fetch_wikipedia_views(self) -> pd.DataFrame:
        """Fetches daily pageviews proxy from Wikipedia API as alternative data (unless views were passed in)."""
        if self._views is not None:
            df = self._views.copy()
            df['views_diff'] = df['views'].pct_change()
            return df[['views_diff']].dropna()

        # Standardized Wikimedia API format
        formatted_start = self.start_date.replace("-", "")
        formatted_end = self.end_date.replace("-", "")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{self.wiki_page}/daily/{formatted_start}/{formatted_end}"

        headers = {'User-Agent': 'QuantResearchBot/1.0 (alemarti.2001@gmail.com)'}
        response = requests.get(url, headers=headers).json()

        records = [{'date': item['timestamp'][:8], 'views': item['views']} for item in response['items']]
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)
        # Smooth and difference the alternative data to prevent spurious correlation
        df['views_diff'] = df['views'].pct_change()
        return df[['views_diff']].dropna()

    def construct_design_matrix(self) -> tuple:
        """Merges datasets and generates lagged alternative data features."""
        market = self.fetch_market_data()
        alternative = self.fetch_wikipedia_views()

        # Inner join on datetime index
        merged = market.join(alternative, how='inner').dropna()

        # Create lagged features (t-1, t-2) to avoid look-ahead bias
        for lag in [1, 2]:
            merged[f'lag_views_{lag}'] = merged['views_diff'].shift(lag)
            merged[f'lag_vol_{lag}'] = merged['realized_volatility'].shift(lag)

        merged = merged.dropna()

        # Features (X) and Target (y: next-day realized volatility)
        feature_cols = ['lag_views_1', 'lag_views_2', 'lag_vol_1', 'lag_vol_2']
        X = merged[feature_cols]
        y = merged['realized_volatility']

        return X, y

    def run_backtest_pipeline(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Executes a rigorous Time-Series Walk-Forward validation.

        Each fold compares three forecasts of the same target:
          - model:       Ridge on all features (volatility lags + attention lags)
          - persistence: naive forecast vol_t = vol_{t-1} (lag_vol_1), nothing fitted
          - vol_only:    same Ridge on the volatility lags only
        The attention features add value only if the model beats vol_only (and persistence).
        Returns one row of metrics per fold.
        """
        # Avoid standard K-Fold to prevent data leakage across temporal boundaries
        tscv = TimeSeriesSplit(n_splits=5)

        vol_cols = [c for c in X.columns if c.startswith('lag_vol_')]

        def make_pipeline():
            # Scaler inside the pipeline: fitted on each fold's training slice only
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ])

        print(f"--- Running Out-of-Sample Backtest: Target -> {self.ticker} Volatility ---")

        rows = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            full = make_pipeline().fit(X_train, y_train)
            vol_only = make_pipeline().fit(X_train[vol_cols], y_train)

            preds = {
                'model': full.predict(X_test),
                'persistence': X_test['lag_vol_1'].to_numpy(),
                'vol_only': vol_only.predict(X_test[vol_cols]),
            }

            row = {
                'fold': fold + 1,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'train_end': y_train.index[-1],
                'test_start': y_test.index[0],
                'test_end': y_test.index[-1],
            }
            for name, p in preds.items():
                row[f'rmse_{name}'] = np.sqrt(mean_squared_error(y_test, p))
                row[f'r2_{name}'] = r2_score(y_test, p)
            rows.append(row)

            print(f"Fold {fold + 1} | "
                  f"model RMSE {row['rmse_model']:.5f} R^2 {row['r2_model']:.4f} | "
                  f"persistence RMSE {row['rmse_persistence']:.5f} R^2 {row['r2_persistence']:.4f} | "
                  f"vol-only RMSE {row['rmse_vol_only']:.5f} R^2 {row['r2_vol_only']:.4f}")

        return pd.DataFrame(rows)

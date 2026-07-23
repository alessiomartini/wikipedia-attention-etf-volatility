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
    def __init__(self, ticker: str, wiki_page: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.wiki_page = wiki_page
        self.start_date = start_date
        self.end_date = end_date

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetches daily ETF market data using yfinance."""
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Calculate log returns to ensure stationarity I(0)
        df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df['realized_volatility'] = df['log_return'].rolling(window=5).std()
        return df[['log_return', 'realized_volatility']].dropna()

    def fetch_wikipedia_views(self) -> pd.DataFrame:
        """Fetches daily pageviews proxy from Wikipedia API as alternative data."""
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

        merged.dropna(inplace=True)

        # Features (X) and Target (y: next-day realized volatility)
        feature_cols = ['lag_views_1', 'lag_views_2', 'lag_vol_1', 'lag_vol_2']
        X = merged[feature_cols]
        y = merged['realized_volatility']

        return X, y

    def run_backtest_pipeline(self, X: pd.DataFrame, y: pd.Series):
        """Executes a rigorous Time-Series Walk-Forward validation."""
        # Avoid standard K-Fold to prevent data leakage across temporal boundaries
        tscv = TimeSeriesSplit(n_splits=5)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])

        print(f"--- Running Out-of-Sample Backtest: Target -> {self.ticker} Volatility ---")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            print(f"Fold {fold + 1} | Out-of-Sample RMSE: {rmse:.5f} | R^2 Score: {r2:.4f}")


if __name__ == "__main__":
    # Example deployment: Predicting QQQ Volatility via Nvidia Wikipedia Interest Dynamics
    engine = AlternativeDataAlphaEngine(
        ticker="QQQ",
        wiki_page="Nvidia",
        start_date="2025-01-01",
        end_date="2026-06-01"
    )
    X, y = engine.construct_design_matrix()
    engine.run_backtest_pipeline(X, y)
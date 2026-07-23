# Alternative Data Predictive Engine: Market Microstructure & Attention Dynamics

An institutional-grade quantitative research framework designed to evaluate the predictive power of non-traditional alternative datasets (attention metrics and proxy indicators) on ETF volatility structures. 

The infrastructure enforces strict statistical stationarity and implements a temporal walk-forward validation pipeline to systematically mitigate information leakage.

## Core Methodology

### 1. Data Ingestion & Synchronization
* **Market Layer**: Daily open-high-low-close-volume (OHLCV) arrays extracted for broad-market and thematic ETFs (`SPY`, `QQQ`, `GLD`) via Yahoo Finance.
* **Alternative Layer**: Structural user attention proxies derived via the Wikimedia REST API (daily granular pageviews for macroeconomic and systemic asset nodes).
* **Alternative Macro (Sov.ai API integration stub)**: Modular schema ready to ingest predictive structural features.

### 2. Statistical Transformations & Stationarity
Raw financial time-series are structurally non-stationary, presenting integrated order $I(1)$ dynamics. To prevent spurious regressions and unstable covariance matrices:
* Price arrays are converted into daily log returns:
  $$r_t = \ln(P_t / P_{t-1})$$
* Target variable is formulated as realized rolling volatility over a 5-day trading window.
* Alternative metrics are processed using percentage first-differences to isolate pure interest shifts from deterministic trends.

### 3. Validation Framework (Anti-Contamination Protocol)
Standard $K$-Fold cross-validation violates temporal dependencies, leading to massive look-ahead bias due to data leakage. This framework strictly utilizes a **Walk-Forward Time-Series Split (`TimeSeriesSplit`)**:

Fold 1: [ Train (t_0 to t_1) ] -> [ Test (t_1 to t_2) ]

Fold 2: [ Train (t_0 to t_2) ] -> [ Test (t_2 to t_3) ]

Fold 3: [ Train (t_0 to t_3) ] -> [ Test (t_3 to t_4) ]


Features are scaled dynamically within each cross-validation slice using standard deviations computed *solely* on the active training subset via `sklearn.pipeline.Pipeline`.

## Project Structure
`src/data_ingestion.py`: Asynchronous download mechanics for API endpoints.
`src/features.py`: Stationarity verification (Augmented Dickey-Fuller tests) and cross-lag generation.
`src/model_pipeline.py`: Scaled Ridge/Lasso regularization algorithms and out-of-sample statistical parsing.

## Setup & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/alternative-data-alpha.git](https://github.com/yourusername/alternative-data-alpha.git)
   cd alternative-data-alpha
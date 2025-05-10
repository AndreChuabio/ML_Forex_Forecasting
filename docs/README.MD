# Forex Forecasting ML Project

## Overview

This project builds and evaluates machine‑learning models to forecast 1–5 day‑ahead returns of the EUR/USD and USD/JPY currency pairs. It:

* **Collects** historical FX data and macroeconomic indicators
* **Creates** technical indicators (SMA, EMA, MACD, Bollinger Bands, RSI, Stochastic) and lagged returns
* **Trains** multiple regressors (Linear, Ridge, Lasso, ElasticNet, RandomForest, XGBoost) with time‑series cross‑validation
* **Evaluates** performance (MSE, RMSE, MAE, Hit‑rate, R²) across horizons
* **Visualizes** accuracy vs. horizon, actual vs. predicted series, and feature importances

## Repository Structure

```
ML-Term-Project/
├── docs/                       # Design docs and slides
│   └── README.md               # This file
├── model_info/                 # Saved feature‑to‑column mappings
│   └── EURUSD=X_feature_mapping.json
│   └── USDJPY=X_feature_mapping.json
├── output/                     # Raw CSV downloads
│   └── EURUSD_data.csv
│   └── USDJPY_data.csv
├── results/                    # Generated plots & CSV summaries
├── src/
│   ├── data/                   # Data-collection scripts
│   │   └── collect_data.py
│   └── models/                 # Training and evaluation code
│       └── Train_EURUSD_models.py
│       └── train_jpy_models.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview & reproducibility steps
```

## Setup & Installation

1. Clone this repository:

   ```bash
   git clone git@github.com:AndreChuabio/Forex_Forecasting.git
   cd Forex_Forecasting
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Data Collection

* `src/data/collect_data.py` pulls FX prices from Yahoo Finance and macro series via FRED
* Output CSVs are saved under `output/`

## Model Training & Evaluation

1. Configure horizons and models in `Train_EURUSD_models.py` (similarly for USD/JPY).
2. Run training script:

   ```bash
   python src/models/Train_EURUSD_models.py
   ```
3. Results (metrics, feature importance tables, and plots) will appear in `results/`.

## Reproducing the Analysis

1. Ensure you have raw data in `output/EURUSD_data.csv` and `USDJPY_data.csv`.
2. Run the training scripts for each currency:

   ```bash
   python src/models/Train_EURUSD_models.py
   python src/models/train_jpy_models.py
   ```
3. Open the generated PNGs (`results/`) to inspect performance by horizon.
4. Use the CSVs in `results/` for tables in your report or slides.

## Visualizations

* **R², RMSE, MAE, HitRate by Horizon**: line plots per model
* **Actual vs. Predicted Returns**: time‑series comparators
* **Feature Importances**: bar tables and top‑10 rankings for XGBoost

## Contributing

Feel free to open issues or submit pull requests to extend model types or include new features.

---

*Created by Andre Chuabio – May 2025*

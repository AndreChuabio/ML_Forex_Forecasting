# src/models/train_eur_models.py

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(path: Path):  # load EURUSD raw csv and parse dates
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return df


# compute log returns and n-day lags for EURUSD
def add_return_lags(df: pd.DataFrame, n_lags: int = 5):
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in range(1, n_lags + 1):
        df[f'lag{lag}'] = df['log_ret'].shift(lag)
    return df.dropna()


def split_features_target(df: pd.DataFrame, feature_cols, target_col, split_date):
    X = df[feature_cols]
    y = df[target_col]
    X_train = X.loc[:split_date]
    X_test = X.loc[split_date:]
    y_train = y.loc[:split_date]
    y_test = y.loc[split_date:]
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    dir_acc = (np.sign(y_test) == np.sign(preds)).mean()
    print(f"{name}: RMSE={rmse:.5f}, MAE={mae:.5f}, R2={r2:.4f}, DirAcc={dir_acc:.3f}")


def main():
    root = Path(__file__).resolve().parents[2]
    data_path = root / 'output' / 'EURUSD_data.csv'
 # 1)load and feature engineer
    df = load_data(data_path)
    df = add_return_lags(df, n_lags=5)

    # 2) prepare X and y
    feature_cols = [f'lag{i}' for i in range(1, 6)]
    target_col = 'log_ret'
    split_date = "2023-01-01"
    X_train, X_test, y_train, y_test = split_features_target(
        df, feature_cols, target_col, split_date
    )

    # 3) set up cross validation and StandardScaler

    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    # 4) train and evaluate models

    # linear regression
    lin = Pipeline([
        ('scale', scaler),
        ('lr', LinearRegression())
    ])
    lin.fit(X_train, y_train)
    evaluate("LinearRegression", lin, X_test, y_test)

    # Ridge with built in CV
    ridge = Pipeline([
        ('scale', scaler),
        ('model', RidgeCV(alphas=np.logspace(-4, 4, 50), cv=tscv))
    ])

    ridge.fit(X_train, y_train)
    print("-> Best alpha (ridge)", ridge.named_steps['model'].alpha_)
    evaluate("RidgeCV", ridge, X_test, y_test)

    # Lasso with built in CV
    lasso = Pipeline([
        ('scale', scaler),
        ('model', LassoCV(alphas=np.logspace(-4, 1, 50), cv=tscv, max_iter=5000))
    ])

    lasso.fit(X_train, y_train)
    print("-> Best alpha (Lasso)", lasso.named_steps['model'].alpha_)
    evaluate("RidgeCV", ridge, X_test, y_test)


if __name__ == "__main__":
    main()

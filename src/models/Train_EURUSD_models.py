
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate


# Load data and build targets/lagged features
df = pd.read_csv('../../output/EURUSD_data.csv')

df['return'] = df['Close'].pct_change()  # this is only one day

print(df.head(5))


print(df.columns.tolist())


# Create lagged features: for closing price (not return ) shifted by a few days back (1, 2 and 5 here)
for lag in [1, 2, 5]:
    df[f'Close_lag_{lag}'] = df['Close'].shift(
        lag)  # how i calculate _ day ahead!


# Create forward-looking targets for multi day forecasting : 1 to 5 days ahead
for days in range(1, 6):
    # Build Target as Returns
    df[f'target_{days}d'] = df['Close'].shift(-days)/df['Close']-1


# helper to split TA vs Econ
ta_patterns = ['SMA', 'EMA', 'MACD', 'ADX', 'Bollinger', 'RSI', 'Stochastic']

print(df.head(5))


def categorize(feat):
    return 'TA' if any(p in feat for p in ta_patterns) else \
        'Econ' if feat not in ('return', 'return_lag_1',
                               'return_lag_2', 'return_lag_5') else 'Return'


# build feature list by exculiding raw target, data, dividends, and split columns ( can aslo include return lags)
raw_drop = ['Close', 'Date', 'Dividends', 'Stock Splits']
features = [col for col in df.columns
            if col not in raw_drop
            and not col.startswith('target_')
            and categorize(col) in ('TA', "Return")
            ]

for lag in [1, 2, 5]:
    # make sure using past to predict future # ACTUALLY PREDICTING RETURN NOW
    df[f'return_lag_{lag}'] = df['return'].shift(lag)
    features.append(f'return_lag_{lag}')

# Define base estimators and their param grids
base_models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'XGB': XGBRegressor(silent=True, verbosity=0),
    'ENet': ElasticNet(max_iter=10_000, random_state=42),
    'RF': RandomForestRegressor(random_state=42)
}
param_grids = {
    'Linear': {
        'model__fit_intercept': [True, False]
    },
    'Ridge': {
        'model__alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    },
    'Lasso': {
        'model__alpha': np.logspace(-4, 1, 20),
        'model__max_iter': [5_000, 10_000, 20_000],
        'model__tol': [1e-4, 1e-3]
    },
    'XGB': {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.6, 0.8, 1.0]
    },

    'ENet': {
        'model__alpha': np.logspace(-4, 1, 20),
        'model__l1_ratio': [0.01, 0.1, 0.25, 0.5],
        'model__tol': [1e-4, 1e-3]

    },
    'RF': {
        'model__n_estimators': [200, 500, 1000],
        'model__max_depth': [3, 5, 8],
        'model__max_features': ['sqrt', 'log2', 0.5],

    },
}

all_metrics = []

# Loop over each forecast horizon ( 1 to 5 days ahead )
for days in range(1, 6):

    # drop NaNs in feat or in current target
    df_ = df.dropna(subset=features + [f'target_{days}d'])
    X = df_[features].values  # Separate features
    y = df_[F'target_{days}d'].values

    print(f'\nForecasting {days}-day ahead:')
    # time-series CV: 5 folds, no shuffling!
    tscv = TimeSeriesSplit(n_splits=5)

    for name, estimator in base_models.items():  # build two step pipeline: scaler ->  model
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])
        grid = GridSearchCV(
            pipe,
            param_grid=param_grids[name],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X, y)
        best_pipe = grid.best_estimator_

        # Final train/test split ( last 20% as test, no shuffle)
        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Retrain on training portion
        best_pipe.fit(X_train, y_train)
        pred = best_pipe.predict(X_test)

        # Actual vs Predicted Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, pred, alpha=0.4)
        lims = max(abs(y_test).max(), abs(pred).max())
        plt.plot([-lims, lims], [-lims, lims], '--')    # 45° reference line
        plt.xlabel("Actual return")
        plt.ylabel("Predicted return")
        plt.title(f"{name} {days}-day ahead: Actual vs Predicted EURUSD")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{name}_{days}d_actual_vs_pred_EURUSD.png")
        plt.close()

        # Up/Down Confusion Matrix
        #    Convert to +1 / -1 labels
        y_true_sign = (y_test > 0).astype(int)
        y_pred_sign = (pred > 0).astype(int)

        cm = confusion_matrix(y_true_sign, y_pred_sign, labels=[1, 0])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Up", "Down"]
        )
        plt.figure(figsize=(4, 4))
        disp.plot(cmap=None, colorbar=False)  # matplotlib default colormap
        plt.title(f"{name} {days}-day ahead Sign Confusion_EURUSD")
        plt.savefig(f"results/{name}_{days}d_confusion_EURUSD.png")
        plt.close()

        test_idx = df_.index[split:]  # Row indices in the filtered df_
        dates = pd.to_datetime(df_.loc[test_idx, 'Date'])

        plt.figure(figsize=(10, 4))  # Actual vs Predicted returns time series
        plt.plot(dates, y_test, label='Actual return', linewidth=1)
        plt.plot(dates, pred, label="Predicted return", linewidth=1, alpha=0.8)
        plt.title(
            f"{days}-Day Ahead: Actual vs Predicted Returns for EURUSD ({name})")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/actual_vs_predicted_EURUSD_{name}_{days}d.png")
        plt.close()

        # Compute metrics

        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        hit = np.mean(np.sign(pred) == np.sign(y_test))
        r2 = r2_score(y_test, pred)
        print(
            f"{name:6s} | "
            f"MSE={mse:.4e} | RMSE={rmse:.4e} | "
            f"MAE={mae:.4e} | HitRate={hit:.2%} | R2={r2:.4f} | "
            f"best_params={grid.best_params_}"
        )

        all_metrics.append({
            "horizon": days,
            "model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "HitRate": hit
        })

       # Generic feature-selection: keep only top quartile of features by score

        model = best_pipe.named_steps['model']

        if hasattr(model, 'coef_'):  # Grab absolute importances
            scores = np.abs(model.coef_)
        else:
            scores = model.feature_importances_

        imp_df = (
            pd.DataFrame({
                'feature': features,
                'importance': scores
            })
            .sort_values('importance', ascending=False)
        )

        # print and save full feature importance table
        print(
            f"\nEURUSD Full feature importances for {name} ({days}-day ahead):")
        print(imp_df.to_markdown(index=False))

        imp_df.to_csv(
            f"results/{name}_{days}d_feature_importancesEURUSD.csv",
            index=False
        )

        # Keep 50th percentile of features by absolute importance
        cutoff = np.percentile(scores, 50)

        selector = SelectFromModel(model, prefit=True, threshold=cutoff)

        mask = selector.get_support()
        kept = [features[i] for i, keep in enumerate(mask) if keep]
        ta_kept = [f for f in kept if categorize(f) == 'TA']
        econ_kept = [f for f in kept if categorize(f) == 'ECON']
        ret_kept = [f for f in kept if categorize(f) == 'Return']

        print(f"{name} kept {len(kept)}/{len(features)} features:")
        print(f" TA  ({len(ta_kept)} :{ta_kept} ")
        print(f" Econ  ({len(econ_kept)} :{econ_kept} ")
        print(f" Return  ({len(ret_kept)} :{ret_kept} ")

        # For 5-day ahead XGB model return
        if days == 5 and name == 'XGB':
            imps = best_pipe.named_steps['model'].feature_importances_
            top_10 = np.argsort(imps)[::-1][:10]
            print("\n Top 10 XGBoost Feature Importances (5-day ahead):")
            for i in top_10:  # top 10 features
                print(f'{features[i]:20s}:{imps[i]:.4f}')


metrics_df = pd.DataFrame(all_metrics)  # Metrics table Viz
metrics_df.to_csv('results/metrics_by_horizonEURUSD.csv', index=False)
print(metrics_df.to_markdown(index=False))

for metric in ["R2", "RMSE", "MAE", "HitRate"]:
    plt.figure()
    for mdl in metrics_df["model"].unique():
        dfm = metrics_df[metrics_df["model"] == mdl]
        plt.plot(dfm["horizon"], dfm[metric], marker='o', label=mdl)
    plt.xlabel("Forecast horizon(days)for EURUSD")
    plt.ylabel(metric)
    plt.title(f"{metric} by horizon")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metric.lower()}_by_horizon_EURUSD")
    plt.close()

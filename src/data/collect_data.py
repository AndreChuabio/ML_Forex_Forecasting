#!/usr/bin/env python
"""
Forex data collection Script

This script creates all necessary directories for our forex forecasting project. Please run this before using any other scripts to ensure the required folder exists.
"""
from pandas_datareader.data import DataReader
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import yfinance as yf
import ta

from src.utils.setup_directories import print_status

import os
import sys
from datetime import datetime, timedelta
import time
import random
import json
import warnings


# from src.utils.setup_directories import print_status


warnings.filterwarnings('ignore')  # Ignore warnings


class ForexDataCollector:
    def __init__(self, tickers=None, start_date=None, end_date=None,
                 fetch_economic=True, max_retries=5):
        """
        Initialize ForexDataCollector with forex pair tickers and date range.

        Paramaters:
        tickers : list of forex tickers in yahoo finance format 
        start_date: start date for data retrieval ( default 5 years ago)
        end_date: end date of data retrieval ( default:today)
        fetch_economic: whether to fetch economic indicators from FRED
        max_retries : Maximum number of retry attempts for API calls

        """

        self.tickers = tickers or ['EURUSD=X', 'USDJPY=X']
        self.start_date = start_date or (
            datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.fetch_economic = fetch_economic
        self.max_retries = max_retries

        self.data = {}
        self.fred_indicators = {}
        self.feature_mappings = {}

        for directory in ['output', 'model_info']:
            os.makedirs(directory, exist_ok=True)

    # fetch historical data for forex pairs with retry mechanism
    def fetch_forex_data(self):
        print_status(
            f"fetching forex data from {self.start_date} to {self.end_date}...")

        for ticker in self.tickers:
            print_status(f"downloading {ticker}...")

            # implement retry with exponential backoff
            for retry in range(self.max_retries):
                try:
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date,
                        interval='1d',
                        auto_adjust=False,
                        back_adjust=False)

                    if df.empty or df.shape[0] < 5:  # sanity check
                        raise ValueError(
                            f"Not enough data points for {ticker}")
                    print_status(
                        f"Data shape for {ticker}:{df.shape}", "success")

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]

                    if 'Volume' not in df.columns or df['Volume'].isnull().all():
                        df['Volume'] = 1000

                    if df.isnull().any().any():
                        df = df.fillna(method='ffill').fillna(method='bfill')

                    print_status(
                        f"successfully processed data for {ticker}", "success"
                    )
                    self.data[ticker] = df
                    break

                except Exception as e:
                    wait_time = (2 ** retry) + random.uniform(0, 1)
                    if retry < self.max_retries - 1:
                        print_status(
                            f"error downloading {ticker}:{str(e)},retrying in {wait_time:.1f} seconds...", "warning")
                        time.sleep(wait_time)
                    else:
                        print_status(
                            f"failed to download {ticker} after {self.max_retries} attempts: {str(e)}", "error")
        return self

    def add_technical_indicators(self):
        for ticker in self.tickers:
            if ticker not in self.data:
                print_status(
                    f"no data available for {ticker}, skipping indicators", "warning")
                continue
            try:
                print_status(f"Adding technical indicators for {ticker}...")
                df = self.data[ticker].copy()
                feature_mapping = {}
                feature_idx = 0

                # add OHLCV columns as initial features
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_{col}'
                    feature_idx += 1

                # moving averages
                for period in [20, 50, 200]:
                    df[f'SMA_{period}'] = ta.trend.sma_indicator(
                        df['Close'], window=period)
                    feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_SMA_{period}'
                    feature_idx += 1

                df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
                feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_EMA_20'
                feature_idx += 1

                # MACD
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_Hist'] = ta.trend.macd_diff(df['Close'])

                for name in ['MACD', 'MACD_Signal', 'MACD_Hist']:
                    feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_{name}]'
                    feature_idx += 1

                # ADX

                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
                feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_ADX'
                feature_idx += 1

                # Bollinger Bands
                df['Bollinger_High'] = ta.volatility.bollinger_hband(
                    df['Close'])
                df['Bollinger_Low'] = ta.volatility.bollinger_lband(
                    df['Close'])
                df['Bollinger_Width'] = df['Bollinger_High'] - \
                    df['Bollinger_Low']
                df['Bollinger_%b'] = ta.volatility.bollinger_pband(
                    df['Close'])

                for name in ['Boillinger_High', 'Boillinger_Low', 'Boillinger_Width', 'Boillinger_%b']:
                    feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_{name}'
                    feature_idx += 1

                # RSI

                df['RSI'] = ta.momentum.rsi(df['Close'])
                feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_RSI'
                feature_idx += 1

                # Stochastic

                df['Stochastic_K'] = ta.momentum.stoch(
                    df['High'], df['Low'], df['Close'])
                df['Stochastic_D'] = ta.momentum.stoch_signal(
                    df['High'], df['Low'], df['Close'])

                for name in ['Stochastic_K', 'Stochastic_D']:
                    feature_mapping[f'Feature_{feature_idx}'] = f'{ticker}_{name}]'
                    feature_idx += 1

                # save feature mapping
                self.feature_mappings[ticker] = feature_mapping
                mapping_file = os.path.join(
                    'model_info', f'{ticker}_feature_mapping.json')
                with open(mapping_file, 'w') as f:
                    json.dump(feature_mapping, f, indent=4)
                print_status(
                    f" feature mapping saved to {mapping_file}", "success")

                # update forex data
                self.data[ticker] = df
                print_status(
                    f"successfully added technical indicators for {ticker}", "success")

            except Exception as e:
                print_status(
                    f" error adding technical indicators for {ticker}:{str(e)}", "error")

        return self

    # fetch economic indicators from fred with retry mechanisms

    from pandas_datareader.data import DataReader

    def fetch_economic_indicators(self):
        from pandas_datareader.data import DataReader

        if not self.fetch_economic:
            print_status(
                "skipping economic indicators fetch disabled", "warning")
            return self

        print_status("fetching economic indicators from FRED..")
        # dictionary of economic indicators to fetch with friendly names

        indicators = {
            "USD_RATES": "DFF",  # Federal Funds rate,
            "EUR_RATES": "ECBDFR",  # ECB Deposit facility rate,
            "JPY_RATES": "IRSTCB01JPM156N",  # Bank of Japan policy rate
            "US_CPI": "CPIAUCSL",  # US consumer price index
            "EU_CPI": "CP0000EZ19M086NEST",  # EU Harmonized index of consumer prices
            "JP_CPI": "JPNCPIALLMINMEI",  # Japan consumer price index
            'US_GDP': 'GDP',                   # US Gross Domestic Product
            'EU_GDP': 'CPMNACSCAB1GQEU272020',  # EU Gross Domestic Product
            'JP_GDP': 'JPNNGDP',               # Japan Gross Domestic Product
            'US_UNEMPLOYMENT': 'UNRATE',       # US Unemployment Rate
            'EU_UNEMPLOYMENT': 'LRHUTTTTEZM156S',  # EU Unemployment Rate
            'JP_UNEMPLOYMENT': 'LRUNTTTTJPM156S',  # Japan Unemployment Rate
            'DOLLAR_INDEX': 'DTWEXBGS',        # Dollar Index
            'VIX': 'VIXCLS',                    # VIX Volatility Index
        }

        for name, series_id in indicators.items():
            for retry in range(self.max_retries):
                try:
                    print_status(f"fetching{name}(series ID:{series_id})...")
                    data = web.DataReader(
                        series_id, 'fred', self.start_date, self.end_date
                    )

                    if not data.empty:
                        print_status(
                            f"succesfully fetched {name} data", "success")
                        self.fred_indicators[name] = data
                        break
                    else:
                        print_status(f"No data found for {name}", "warning")
                        break

                except Exception as e:
                    wait_time = (2 ** retry) + random.uniform(0, 1)
                    if retry < self.max_retries - 1:
                        print_status(
                            f"error fetching {name}:{str(e)},retrying in {wait_time:.1f} seconds...", "warning")
                        time.sleep(wait_time)
                    else:
                        print_status(
                            f"failed to fetch {name} after {self.max_retries} attempts: {str(e)}", "error")
        print_status(f"fetched {len(self.fred_indicators)}/{len(indicators)} economic indicators",
                     "success" if len(self.fred_indicators) > 0 else "warning")
        return self

    def merge_data(self):  # merge forex data with economic indicators
        if not self.fred_indicators:
            print_status("NO econonomic indicators to merge", "warning")
            return self

        print_status("merging forex data with economic indicators...")

        for ticker, df in self.data.items():
            print_status(f"Processing {ticker}...")

            # copy df
            merged_df = df.copy()
            if getattr(merged_df.index, 'tz', None) is not None:
                merged_df.index = merged_df.index.tz_localize(None)

            for name, econ_df in self.fred_indicators.items():
                try:
                    # rename col to indicator name
                    econ = econ_df.rename(columns={econ_df.columns[0]: name})
                    if getattr(econ.index, 'tz', None) is not None:
                        econ.index = econ.index.tz_localize(None)

                    if econ_df.index.freq != 'B':
                        econ_df = econ_df.resample('B').last().fillna(
                            method='ffill')  # resampe to business days

                    merged_df = merged_df.join(econ_df, how='left')
                    print_status(f"added {name} to {ticker}", "success")
                except Exception as e:
                    print_status(
                        f"Error adding {name} to {ticker}: {str(e)}", "error")

            merged_df = merged_df.fillna(method='ffill')
            merged_df = merged_df.fillna(merged_df.mean())
            self.data[ticker] = merged_df

        return self

    def save_data(self):  # save data to csv files

        print_status("saving preppared data...")
        for ticker, df in self.data.items():
            symbol = ticker.replace('=X', '')
            filename = f"output/{symbol}_data.csv"

            try:
                df.to_csv(filename)
                print_status(f"saved {symbol} data to {filename}", "success")
            except Exception as e:
                print_status(
                    f" Error saving {symbol} data : {str(e)}", "error")

        return self

    # print feature mapping for a specific ticker or all tickers
    def print_feature_mapping(self, ticker=None):
        if ticker is not None and ticker in self.feature_mappings:
            print(f"\n Feature mapping for {ticker}:")
            for feature, name in sorted(self.feature_mappings[ticker].items()):
                print(f"{feature}:{name}")

        else:
            for ticker in self.feature_mappings:
                print(f"\nFeature mapping for {ticker}:")
                for feature, name in sorted(self.feature_mappings[ticker].items()):
                    print(f"{feature}:{name}")

    def run_full_pipeline(self):  # run full data prep pipeline with progress tracking
        start_time = time.time()
        print_status("Starting data collection pipieline...")

        steps = [
            ("Fetching forex data", self.fetch_forex_data),
            ("adding technical indicators", self.add_technical_indicators),
            ("Fetching economic indicators", self.fetch_economic_indicators),
            ("Merging Data", self.merge_data),
            ("Saving Data", self.save_data)

        ]

        for step_name, step_func in steps:
            print_status(f"\n{step_name}...", "info")
            step_func()

        end_time = time.time()

        duration = end_time - start_time
        print_status(
            f"\n Dara collection pipeline completed in {duration:.2f} seconds", "success")
        return self


def main():
    """run data collection process"""
    print(f"\n{'='*70}")
    print(f"Forex Data Collection".center(70))
    print(f"{'='*70}\n")

    collector = ForexDataCollector()
    collector.run_full_pipeline()  # run full pipeline

    print(f"\n{'='*70}")
    print(f"Forex Data Collection COMPLETE".center(70))
    print(f"{'='*70}\n")

    collector.print_feature_mapping()  # display feat mappings)

    return 0


if __name__ == "__main__":
    sys.exit(main())

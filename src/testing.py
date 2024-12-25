from utils import *
from strategy import *
import optuna
from ..test_param import params
import os
from typing import List
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')   

def testing():
    dir = input("Enter Directory: ")
    os.makedirs(dir, exist_ok=True)

    # Download Data
    downloader = Downloader()
    data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")
    downloader.close()

    ratio = 0.7
    train_size = int(ratio * len(data))

    train = data.iloc[:train_size].copy()
    test = data.iloc[train_size:].copy()

    del data

    # Define Strategy
    strategy = [s for name, s in strategy_options if name in params['strategies']]
    params['strategy'] = strategy

    insample = Backtesting(
        data=train,
        **params
        )

    try:
        insample.backtest()
        insample.data.to_csv(f"{dir}/insample.csv") 
        insample._history.to_csv(f"{dir}/insample_history.csv")
    except Exception as e:
        #print(f"Error: {e}")
        exit()

    outsample = Backtesting(
        data=test,
        **params
        )

    try:
        outsample.backtest()
        outsample.data.to_csv(f"{dir}/outsample.csv")
        outsample._history.to_csv(f"{dir}/outsample_history.csv")
    except Exception as e:
        #print(f"Error: {e}")





    
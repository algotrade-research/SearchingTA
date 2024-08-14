from utils import *
from alpha import *
import optuna
from typing import List
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')



downloader = Downloader()

print("Downloading Data...")
data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")

downloader.close()
del downloader

print("Train test split")
# Train test  Split
ratio = 0.7
train_size = int(ratio * len(data))

train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()
del data

params = [
    {
        "TP": 0.6,
        "SL": 1,
        "max_pos": 1,
        "strategy": [MACD, BBL]
    },
    {
        "TP": 0.7,
        "SL": 1.4,
        "max_pos": 1,
        "strategy": [MACD, BBL, VWAP]
    },
    {
        "TP": 1,
        "SL": 2,
        "max_pos": 1,
        "strategy": [MACD, BBL, VWAP, RSI]
    }
]

def testing(params: List[dict], train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Function to test the strategy
    """
    # Insample
    insample = Backtesting(data=train, **params)
    try:
        insample.backtest()
        print("Insample results", insample.evaluate())
        insample.data.to_csv(f"{params}.csv")
        insample._history.to_csv("f{params}_hist.csv")
    except Exception as e:
        print("Error", e)
        insample.data.to_csv(f"{params}.csv")
        insample._history.to_csv("f{params}_hist.csv")

    del insample


    # Outsample
    outsample = Backtesting(data=test, **params)

    try:
        outsample.backtest()
        print("Outsample results", outsample.evaluate())
        outsample.data.to_csv(f"{params}-outsample.csv")
        outsample._history.to_csv(f"{params}-outsample_history.csv")
    except Exception as e:
        print("Error", e)
        outsample.data.to_csv(f"{params}-outsample.csv")
        outsample._history.to_csv(f"{params}-outsample_history.csv")

    del outsample

    return "Completed" + params


with ThreadPoolExecutor(max_workers=5) as executor:
    # Submitting tasks
    futures = [executor.submit(testing, param, train, test) for param in params]
    
    # Collecting results
    results = [future.result() for future in futures]

print("Results:", results)

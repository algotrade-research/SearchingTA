from utils import *
from alpha import *
import optuna
import os
from typing import List
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')   

dir = input("Enter Directory: ")

downloader = Downloader()
data = downloader.get_historical_data(start_date="2021-01-01", end_date="2024-08-12")
downloader.close()

ratio = 0.7
train_size = int(ratio * len(data))

train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()

del data

TP = 5.5
SL = 3.0
Strategies = [BBL, MACD, MA5, MA20, PPO, ROC, CCI, Momentum, PSAR , OBV , Donchian, UO, FI, Vortex]
min_signals = 2
max_pos = 5

insample = Backtesting(
    data=train,
    initial_balance=2e9,
    TP=TP,
    SL=SL,
    max_pos=max_pos,
    min_signals=min_signals,
    strategy=Strategies
    )
try:
    insample.backtest()
except Exception as e:
    print(f"Error: {e}")

outsample = Backtesting(
    data=test,
    initial_balance=2e9,
    SL=SL,
    TP=TP,
    max_pos=max_pos,
    min_signals=min_signals,
    strategy=Strategies
    )
try:
    outsample.backtest()
except Exception as e:
    print(f"Error: {e}")

os.makedirs(dir, exist_ok=True)

insample.data.to_csv(f"{dir}/insample.csv") 
insample._history.to_csv(f"{dir}/insample_history.csv")

outsample.data.to_csv(f"{dir}/outsample.csv")
outsample._history.to_csv(f"{dir}/outsample_history.csv")
    
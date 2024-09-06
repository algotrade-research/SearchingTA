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
data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")
downloader.close()

ratio = 0.7
train_size = int(ratio * len(data))

train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()

del data

TP = 5.0
SL = 3.0
Strategies = [MomentumBBL, MA20, PPO, Momentum, Donchian, UO, FI, Volume_MA]
min_signals = 3
max_pos = 10

# TP: 5.0
#  SL: 3.0
# Min Signals: 3
# Max Pos:10
# Strategies: ['Donchian', 'Force Index', 'MA20', 'Momentum', 'MomentumBBL', 'PPO', 'UO', 'Volume_MA']
# Winrate: 0.4128532360984503
# Mean Pnl: 0.735255241567894
# Sharpe Ratio: 1.5038902925981839

insample = Backtesting(
    data=train,
    initial_balance=2e9,
    cost=0.07,
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
    cost=0.07,
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
    
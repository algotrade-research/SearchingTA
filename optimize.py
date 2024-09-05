from utils import *
from alpha import *
import numpy as np
import pandas as pd
import optuna
import os
from typing import List
import warnings
import logging

dir = input("Enter Directory: ")
os.makedirs(dir, exist_ok=True)

warnings.filterwarnings('ignore')
logging.basicConfig(filename=f'{dir}/strategy_optimize.log', level=logging.INFO, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Define the strategy options
strategy_options = [
    ('RSI', RSI), 
    ('Bollinger Bands', BBL), 
    ('MACD', MACD), 
    ('VWAP', VWAP), 
    ('MA5', MA5), 
    ('MA20', MA20), 
    ('PPO', PPO), 
    ('ROC', ROC), 
    ('TSI', TSI), 
    ('ATR', ATR), 
    ('ADX', ADX), 
    ('CCI', CCI), 
    ('Momentum', Momentum), 
    ('Volume_MA', Volume_MA), 
    ('MomentumBBL', MomentumBBL), 
    ('Keltner', Keltner), 
    ('SO', SO), 
    ('Williams R', W_R), 
    ('PSAR', PSAR), 
    ('OBV', OBV), 
    ('Donchian', Donchian), 
    ('UO', UO), 
    ('Force Index', FI), 
    ('Vortex', Vortex) 
]

# Download historical data
print("Downloading Data...")
downloader = Downloader()
data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")
downloader.close()
del downloader

# Train test split
print("Train test split")
ratio = 0.7
train_size = int(ratio * len(data))
train = data.iloc[:train_size].copy()
test = data.iloc[train_size:].copy()
del data

optimize_data = train.iloc[:int(0.5 * len(train))].copy()

# Read best parameters from CSV
try:
    best_params = pd.read_csv("optuna_trials.csv").set_index("number").sort_values("value", ascending=False).iloc[0].to_dict()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    best_params = {}

# Initialize global variables
TP_ = 0
SL_ = 0
min_signals = 0
strategies = []
strategies_name = []

# Parse best parameters
for key, val in best_params.items():
    key = key.replace("params_", "")
    if key == "TP":
        TP_ = float(val)
    elif key == "SL":
        SL_ = float(val)
    elif key == "min_signals":
        min_signals = int(val)
    else:
        if val:
            strategies.append(key)
            strategies_name.append(key)

print(strategies)

strategies = [strategy for strategy_name, strategy in strategy_options if strategy_name in strategies]

print("Params:", TP_, SL_, min_signals, strategies)

def objective(trial):
    TP = TP_ + trial.suggest_float("TP", -1, 1, step=0.5)
    SL = SL_ + trial.suggest_float("SL", -1, 1, step=0.5)
    max_pos = trial.suggest_int("max_pos", 1, 10)
    min_signals_ =  min_signals + trial.suggest_int("min_signals", -2, 1, step=1)

    backtester = Backtesting(
        strategy=strategies, 
        data=optimize_data, 
        initial_balance=2e9, 
        cost=0.07, 
        slippage=0.25, 
        TP=TP, 
        SL=SL, 
        max_pos=max_pos, 
        min_signals=min_signals_
    )
    
    try:
        backtester.backtest()
        
        pnl_series = backtester._history.set_index("close_time")['pnl']
        pnl_series.index = pd.to_datetime(pnl_series.index)
        mean_pnl = pnl_series.mean()
        sharpe_ratio = np.sqrt(252) * pnl_series.resample("1D").sum().mean() / pnl_series.resample("1D").sum().std()
        winrate = (pnl_series > 0).astype(int).sum() / len(pnl_series)
        
        print(f"Mean PnL: {mean_pnl}, Winrate: {winrate}")
        
        hitting_prob = SL / (SL + TP) - 1
        expected_pnl = TP * hitting_prob
        
        loss = float((sharpe_ratio - 1))
        
        subdir_name = f"{trial.number}"
        os.makedirs(os.path.join(dir, subdir_name), exist_ok=True)
        backtester._history.to_csv(os.path.join(dir, subdir_name, "history.csv"))
        backtester.data.Equity.to_csv(os.path.join(dir, subdir_name, "Equity.csv"))
        
        file_path = os.path.join(dir, subdir_name, "params.txt")
        with open(file_path, "w") as file:
            file.write(f"TP: {TP}\n SL: {SL}\nMin Signals: {min_signals_}\nMax Pos:{max_pos}\nStrategies: {strategies_name}\nWinrate: {winrate}\nMean Pnl: {mean_pnl}\nSharpe Ratio: {sharpe_ratio}")
        
        logging.info(f"\n\n=============================")
        logging.info(f"-- {trial.number} --")
        logging.info(f"Strategy: {strategies}, TP: {TP}, SL: {SL}")
        logging.info(f"Winrate: {winrate}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio}")
        logging.info(f"Mean PnL: {mean_pnl}")
        logging.info(f"Loss: {loss}")
        logging.info(f"Number Of Trades: {len(pnl_series)}")
        
        return loss
    except Exception as e:
        print(f"Error: {e}")
        return -1

print("Start Searching...")

try:
    study = optuna.create_study(direction="maximize")  # Assuming you want to minimize the loss
    study.optimize(objective, n_trials=100)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"{dir}/strategy_optimize_trial.csv", index=False)

    best_params = study.best_params
    best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

    with open(f"{dir}strategy_optimize_params.log", "w") as file:
        file.write("Best Trial: " + str(study.best_trial.number) + "\n")
        file.write("Best Loss: " + str(study.best_value) + "\n")
        file.write("Best Parameters:\n" + best_params_str)
        
except KeyboardInterrupt:
    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"{dir}strategy_optimize_trial.csv", index=False)

    best_params = study.best_params
    best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

    with open(f"{dir}strategy_optimize_params.log", "w") as file:
        file.write("Best Trial: " + str(study.best_trial.number) + "\n")
        file.write("Best Loss: " + str(study.best_value) + "\n")
        file.write("Best Parameters:\n" + best_params_str)
        
except Exception as e:
    df_trials = study.trials_dataframe()
    df_trials.to_csv(f"{dir}strategy_optimize_trial.csv", index=False)

    best_params = study.best_params
    best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

    with open(f"{dir}strategy_optimize_params.log", "w") as file:
        file.write("Best Trial: " + str(study.best_trial.number) + "\n")
        file.write("Best Loss: " + str(study.best_value) + "\n")
        file.write("Best Parameters:\n" + best_params_str)

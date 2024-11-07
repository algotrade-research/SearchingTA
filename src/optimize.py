import os
import numpy as np
import pandas as pd
import optuna
import warnings
import logging
from typing import List
from utils import *
from strategy import *

def initialize_logging(log_dir: str) -> None:
    """Initializes logging settings."""
    log_path = os.path.join(log_dir, "strategy_optimize.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Start Optimization")

def download_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads historical data within a specified date range."""
    downloader = Downloader()
    data = downloader.get_historical_data(start_date=start_date, end_date=end_date)
    downloader.close()
    return data

def objective(trial, optimize_data, TP_, SL_, min_signals, strategies, dir) -> float:
    """Defines the objective function for optimization with Optuna."""
    TP = TP_ + trial.suggest_float("TP", -1, 1, step=0.5)
    SL = SL_ + trial.suggest_float("SL", -1, 1, step=0.5)
    max_pos = trial.suggest_int("max_pos", 1, 10)
    min_signals_ = min_signals + trial.suggest_int("min_signals", -2, 1, step=1)

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

        # Log and save results
        trial_dir = os.path.join(dir, f"{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        backtester._history.to_csv(os.path.join(trial_dir, "history.csv"))
        backtester.data.Equity.to_csv(os.path.join(trial_dir, "Equity.csv"))

        with open(os.path.join(trial_dir, "params.txt"), "w") as file:
            file.write(f"TP: {TP}\nSL: {SL}\nMin Signals: {min_signals_}\n"
                       f"Max Pos: {max_pos}\nStrategies: {strategies}\n"
                       f"Winrate: {winrate}\nMean PnL: {mean_pnl}\nSharpe Ratio: {sharpe_ratio}")

        logging.info(f"Trial {trial.number} - Strategies: {strategies}, TP: {TP}, SL: {SL}")
        logging.info(f"Winrate: {winrate}, Sharpe Ratio: {sharpe_ratio}, Mean PnL: {mean_pnl}")

        loss = sharpe_ratio - 1
        return loss
    except Exception as e:
        print(f"Error in objective function: {e}")
        return -1

def optimize():
    dir = input("Enter Directory: ")
    os.makedirs(dir, exist_ok=True)

    initialize_logging(dir)
    warnings.filterwarnings('ignore')

    print("Downloading Data...")
    data = download_data(start_date="2022-01-01", end_date="2024-08-12")
    train, test = train_test_split(data, ratio=0.7)
    optimize_data = train.iloc[:int(0.5 * len(train))].copy()

    best_params = load_best_parameters("optuna_trials.csv")
    TP_, SL_, min_signals, strategies_names = parse_best_params(best_params)

    strategies = [strategy for strategy_name, strategy in strategy_options if strategy_name in strategies_names]
    print("Params:", TP_, SL_, min_signals, strategies_names)

    # Start optimization
    try:
        print("Start Searching...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, optimize_data, TP_, SL_, min_signals, strategies, dir), n_trials=100)

        # Save trials and best parameters
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(dir, "strategy_optimize_trial.csv"), index=False)

        best_params = study.best_params
        best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())
        with open(os.path.join(dir, "strategy_optimize_best_params.txt"), "w") as file:
            file.write(f"Best Trial: {study.best_trial.number}\nBest Loss: {study.best_value}\nBest Parameters:\n{best_params_str}")

    except KeyboardInterrupt:
        print("Optimization interrupted.")
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(dir, "strategy_optimize_trial.csv"), index=False)
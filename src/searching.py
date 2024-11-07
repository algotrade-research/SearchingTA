from utils import *
from strategy import *
import optuna
import os
from typing import List
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')
def searching():
    dir = 'searching'
    os.makedirs(dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(dir, "searching.log"), level=logging.INFO, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    number_of_trials = int(input("Number of Trials: "))

    # Download historical data
    downloader = Downloader()

    print("Downloading Data...")
    data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")
    downloader.close()

    del downloader

    # Train test  Split
    print("Train test split")
    # Train test  Split
    ratio = 0.7
    train_size = int(ratio * len(data))

    train = data.iloc[:train_size].copy()
    test = data.iloc[train_size:].copy()
    del data

    optimize_data = train.iloc[:int(0.5 * len(train))].copy()

    # Optmize
    optimize_result = []

    def objective(trial):
        cost = 0.07
        TP = trial.suggest_float("TP", 1, 5, step=0.5)
        SL = trial.suggest_float("SL", 1, 5, step=0.5)
        
        
        selected_strategies = []
        strategies = []
        for strategy_name, strategy_function in strategy_options:
            if trial.suggest_categorical(strategy_name, [True, False]):
                selected_strategies.append(strategy_name)
                strategies.append(strategy_function)
        
        min_signals = trial.suggest_int("min_signals", 1, 5 if len(strategies) > 5 else len(strategies))
        backtester = OptimzeBacktesting(
            strategy=strategies, 
            data=optimize_data,
            initial_balance=2e9, 
            cost=cost, 
            slippage=0.25, 
            TP=TP, 
            SL=SL, 
            max_pos=1e10, 
            min_signals=min_signals
        )
        
        try:
            backtester.backtest()
            
            pnl_series: pd.Series = backtester._history['pnl']
            mean_pnl = pnl_series.mean()
            winrate = (pnl_series > 0).astype(int).sum() / len(pnl_series)
            
            print("Mean PnL: ", mean_pnl, "Winrate: ", winrate)
            
            # BaseLine
            sl, tp = SL + cost, TP - cost
            hitting_prob = sl / (sl + tp)
            expected_pnl = tp * hitting_prob
            
            loss = float((mean_pnl / expected_pnl - 1) + (winrate / hitting_prob - 1)) / 2
            
            subdir_name = f"{trial.number}"
            os.makedirs(os.path.join("searching", subdir_name), exist_ok=True)
            backtester._history.to_csv(os.path.join("searching", subdir_name, "history.csv"))
            
            file_path = os.path.join("searching", subdir_name, "params.txt")
            with open(file_path, "w") as file:
                file.write(
                    f"""
    TP: {TP}, SL: {SL}
    Expected PnL: {expected_pnl}
    Hitting Probability: {hitting_prob}
    Strategies: {selected_strategies}
    Winrate: {winrate}, Mean Pnl: {mean_pnl}
    Loss: {loss}
                    """
                    )
            
            logging.info(f"-- {trial.number} --")
            logging.info(f"Strategy: {selected_strategies}, TP: {TP}, SL: {SL}")
            logging.info(f"Winrate: {winrate}")
            logging.info(f"Mean PnL: {mean_pnl}")
            logging.info(f"Loss: {loss}")
            logging.info(f"Number Of Trade: {len(pnl_series)}")
            logging.info(f"=============================\n\n")
            
            return loss
        except Exception as e:
            print(f"Error: {e}")
            return -1

    print("Start Searching...")

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=number_of_trials, n_jobs=100)

        df_trials = study.trials_dataframe()
        df_trials.to_csv("optuna_trials.csv", index=False)

        best_params = study.best_params
        best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

        with open("best_params.log", "w") as file:
            file.write("Best Trial: " + str(study.best_trial.number) + "\n")
            file.write("Best Loss: " + str(study.best_value) + "\n")
            file.write("Best Parameters:\n" + best_params_str)
            
    except KeyboardInterrupt:
        df_trials = study.trials_dataframe()
        df_trials.to_csv("optuna_trials.csv", index=False)

        best_params = study.best_params
        best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

        with open("best_params.log", "w") as file:
            file.write("Best Sharpe Ratio: " + str(study.best_value) + "\n")
            file.write("Best Parameters:\n" + best_params_str)
            
    except Exception as e:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(dir, "optuna_trials.csv"), index=False)

        best_params = study.best_params
        best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

        with open(os.path.join(dir, "best_params.log"), "w") as file:
            file.write("Best Sharpe Ratio: " + str(study.best_value) + "\n")
            file.write("Best Parameters:\n" + best_params_str)

if __name__ == "__main__":
    searching()
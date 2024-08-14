from utils import *
from alpha import *
import optuna
from typing import List
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(filename='optimization_results.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')



downloader = Downloader()

print("Downloading Data...")
data = downloader.get_historical_data(start_date="2022-01-01", end_date="2024-08-12")

print("Train test split")
# Train test  Split
ratio = 0.8
train_size = int(ratio * len(data))

train = data.iloc[:train_size]
test = data.iloc[train_size:]

# train validate
train_ratio = 0.2
train_size = int(train_ratio * len(train))

train_data = train.iloc[:train_size]
validate_data = train.iloc[train_size:]


def objective(trial):
    # max_pos = trial.suggest_int("max_pos", 1, 5)
    TP = trial.suggest_float("TP", 0.5, 15.0, step=0.1)
    SL = trial.suggest_float("SL", 0.1, 15.0, step=0.1)
    interval = trial.sugget_int("interval", 1, 60, step=1)

    strategy_options = ['MACD', 'RSI', 'BBL', 'VWAP', 'MA5', "MA20"]

    selected_strategies = []
    for strategy in strategy_options:
        if trial.suggest_categorical(f"use_{strategy}", [True, False]):
            selected_strategies.append(strategy)
    
    if not selected_strategies:
        selected_strategies.append(trial.suggest_categorical("fallback_strategy", strategy_options))

    strategies = []
    if 'MACD' in selected_strategies:
        strategies.append(MACD)
    if 'RSI' in selected_strategies:
        strategies.append(RSI)
    if 'BBL' in selected_strategies:
        strategies.append(BBL)
    if 'VWAP' in selected_strategies:
        strategies.append(VWAP)
    if 'MA5' in selected_strategies:
        strategies.append(MA5)
    if 'MA20' in selected_strategies:
        strategies.append(MA20)
    
    print(f"""
          Strategy: {selected_strategies}
          TP, SL: {TP, SL}
          interval: {interval}
          """)
    # Initialize the backtesting with the suggested parameters
    backtester = Backtesting(
        strategy=strategies, 
        interval=interval,
        data=train_data,  # Your dataset
        initial_balance=10000, 
        cost=0.07, 
        slippage=0.25, 
        TP=TP, 
        SL=SL, 
        max_pos=1, 
        position_size=0.3, 
        model=None, 
        window=50, 
        MP=False
    )
    
    try:
       # Run the backtest
        backtester.backtest()
        daily_returns = backtester.data.Equity[backtester.data.Equity != 10000].pct_change().dropna()
        std = daily_returns.std()
        sharpe_ratio = float((daily_returns.mean() / std) * np.sqrt(252))
        cum_returns = backtester._histor["pnl"].cumsum()
        drawdown = ((cum_returns / cum_returns.cummax()) - 1).min()
        
        logging.info("Sharpe Ratio:", sharpe_ratio)
        logging.info(f"intercal: {interval} Strategy: {selected_strategies}, TP: {TP}, SL: {SL}")        

        return sharpe_ratio * 0.9 + drawdown * 0.1  # The objective is to maximize the Sharpe ratio
    except Exception as e:
        print(e)
        return -1
        


print("Optimizing")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)

print("Best parameters:", study.best_params)
print("Best Sharpe Ratio:", study.best_value)

# Get the best parameters
best_params = study.best_params

# Convert best_params dictionary to a string representation
best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

with open("best_params.log", "w") as file:
    file.write("Best Sharpe Ratio: " + str(study.best_value) + "\n")
    file.write("Best Parameters:\n" + best_params_str)


# Test the best parameters on the validation set
print("Testing the best parameters on the validation set")
backtester = Backtesting(
    strategy=best_params['strategy'], 
    data=validate_data, 
    initial_balance=10000, 
    cost=0.07, 
    slippage=0.25, 
    TP=best_params['TP'], 
    SL=best_params['SL'], 
    max_pos=best_params['max_pos'], 
    position_size=0.3, 
    model=None, 
    window=50, 
    MP=False
)
try:
    backtester.backtest()
    print("Validation results:", backtester.evaluate())
    backtester.data.to_csv("validation.csv")
    backtester._history.to_csv("validation_history.csv")
except Exception as e:
    print("Error", e)
    backtester.data.to_csv("validation.csv")
    backtester._history.to_csv("validation_history.csv")


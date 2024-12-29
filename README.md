# Scalping Strategy Search

# Project Overview

This project focuses on searching various trading strategies using historical stock market data and the Optuna hyperparameter optimization framework. The optimization process selects the best trading strategy based on several technical indicators and parameters like Take Profit (TP) and Stop Loss (SL). The final results include optimized strategy configurations, along with key metrics such as mean PnL (Profit and Loss), win rate, and loss.

# Workflow Overview

1. Searching for Combinations: Identify optimal combinations of technical indicators, TP, and SL levels.
2. Strategy Optimization:
   - Refine the strategy by adding/removing technical indicators.
   - Tune parameters like SL, TP, and the optimal number of positions.
3. Testing:
   - Test on the in-sample dataset (training data).
   - Test on the out-of-sample dataset (testing data).
4. Paper Trading: Simulate trades based on the optimized strategy to verify performance.

# Data Overview

Data Description: The input data consists of historical stock market price data, including tickdata and volume.

Data Source: The historical data is downloaded from a data provider through the Downloader class.
Date Range: The data used in this project spans from January 1, 2022, to August 12, 2024.

## Input Data:

The input data is retrieved using the Downloader.get_historical_data function in the main script. The script automatically downloads data based on the specified date range (start_date="2022-01-01", end_date="2024-08-12").

```{python}
tickdata = Downloader.get_historical_data(ticker='VN30F1M', start_date="2022-01-01", end_date="2024-08-12")
```

## Output Data:

Search Results: For each trial, backtesting results are saved in the searching/ directory, with detailed logs, parameter configurations, and historical performance metrics in CSV format.

Best Results: The best performing parameters and loss are saved in a file named best_params.log.

# Method Overview

The main components of this project include:

Historical Data Downloading: The project retrieves stock market data using the Downloader class, which provides the raw data needed for backtesting.

Strategy Searching: Using Optuna's hyperparameter optimization framework, we run multiple trials to search for potential combinations of technical indicators and trading parameters.

Strategy Optimizing: Using Optuna's hyperparameter optimization framework, we run multiple trials on the potential conbinations to optimize trading parameters.

Backtesting: Each trial performs backtesting on historical data to compute the performance of the selected strategies.
Logging and Saving Results: Logs are stored for each trial run, and the best-performing strategies are saved for future use.

# Optimization Process

The optimization is performed in two stages:

Stage 1: Initial Search for Strategies
In this stage, the code searches for the best combination of technical indicators (e.g., RSI, MACD, Bollinger Bands) along with optimal TP and SL values. The parameters are defined as follows:

- TP (Take Profit): Suggests an optimal TP level.
- SL (Stop Loss): Optimizes the SL level to minimize risk.
- Technical Indicators: The project includes 24 different technical indicators for strategy generation.

Stage 2: Fine-tuning and Testing
In this stage, further optimization is carried out:

- Adjust SL and TP: Fine-tune previously optimized SL and TP values.
- Maximize Position Holding: Adjust the maximum number of open positions.
- Min Signals: The minimum number of signals that must be met before entering a trade.
- The final result is saved and analyzed using backtesting over both in-sample and out-of-sample datasets.

# Running the Project

1. Environment Setup
   The project uses Python >= 3.9.

To install dependencies:

```{bash}
pip install -r requirements.txt
```

2. Running the Optimization
   You can modify the search space for the parameters folder strategy, then add the name as well as the strategy function in the strategy list.

The input data for each strategy in default is OHLCV with additional Technical Indicators, you can allso add more tech nocal indicators in the ./utils/processor.py

Strategy output:

- 1 for Long signal
- -1 for Short signal
- 0 for no action

Strategy example:

```{python}
def Strategy_1(data):
    close_price = data['close']
    max_in_20 = close.price.rolling(20).max()
    min_in_20 = close.price.rolling(20).min()

    if max_in_20[-1] > max_in_20[-2]:
        return 1
    elif min_in_20[-1] < min_in_20[-2]:
        return -1

    return 0

```

Step 1: Running the searching.py file to search for strategies.

```{bash}
python3 seaching.py
```

The script will prompt for the number of trials to run. The search results, including trial logs and performance data, will be saved in the searching/ directory.

Step 2: Fine-tuning the strategy using the optimizing.py file.

```{bash}
python3 seaching.py
```

This will further optimize the configuration by adjusting parameters like TP, SL, and technical indicators.

Parameter Adjustments
You can modify the search space for the parameters (e.g., TP, SL, strategies) within the Optuna trials in the main.py or optimizing.py scripts.

Example snippet to change TP and SL ranges:

```{python}
TP = trial.suggest_float("TP", 1, 5, step=0.5)
SL = trial.suggest_float("SL", 1, 5, step=0.5)
```

# Result Files

- Trial Logs: Stored in searching/{trial_number}/history.csv.
- Best Parameters: Saved in best_params.log and strategy_optimize_best_params.txt.
- Results Overview
  -Example Metrics from Optimization:
  -Mean PnL: Indicates the average profit or loss per trade.
  -Win Rate: Percentage of winning trades.
  -Sharpe Ratio: Measures risk-adjusted return.
  -Loss: The difference between expected and actual performance, used as the objective function for optimization.

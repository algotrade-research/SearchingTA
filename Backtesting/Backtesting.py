import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Callable

from Portfolio import Portfolio
from BacktestConfig import BacktestConfig


class Backtesting:
    def __init__(self, 
                 strategy: List[Callable], 
                 data: pd.DataFrame, 
                 config: BacktestConfig = BacktestConfig(), 
                 time_out: int=30, 
                 interval="1min"):
        self.strategy = strategy
        self.data = data
        self.config = config
        self.time_out = time_out
        self.order_book = pd.DataFrame(columns=["date", "price", "signal", "timeout"])
        self.portfolio = Portfolio(config.initial_balance)
        self.process_data = self._process_data(interval)
        self.data["Equity"] = config.initial_balance

    def _process_data(self, interval):
        """Preprocess and resample data."""
        ohlcv = self.data.resample(interval).agg({
            "price": "ohlc", 
            "volume": "sum"
        }).dropna()
        ohlcv.columns = ohlcv.columns.droplevel(0)
        ohlcv.index = ohlcv.index.shift(1)
        self.data = self.data.loc[ohlcv.index[20]:ohlcv.index[-1]]
        return ohlcv

    def place_order(self, order_price, signal, date):
        """Place a new order in the order book."""
        order = {
            "date": date,
            "price": order_price,
            "signal": signal,
            "timeout": date + pd.Timedelta(minutes=self.time_out)
        }
        self.order_book = pd.concat([self.order_book, pd.DataFrame([order])], ignore_index=True)

    def check_orders(self, curr_price, bid_price, ask_price, date):
        """
            Check orders for execution or timeout.
            Buy at Ask price, exit at Bid price
            Sell at Bid price, Exit at Ask price
        """
        to_remove = []
        for i, row in self.order_book.iterrows():
            if date >= row["timeout"]:
                to_remove.append(i)
            elif (
                (row["signal"] == 1 and ask_price >= row["price"]) or 
                (row["signal"] == -1 and bid_price <= row["price"])
                ):
                self.portfolio.add_position({
                    "date": date,
                    "price": curr_price,
                    "signal": "buy" if row["signal"] == 1 else "sell",
                    "position_size": self.config.position_size,
                    "position": curr_price * self.config.position_size,
                    "TP": curr_price + self.config.TP if row["signal"] == 1 else curr_price - self.config.TP,
                    "SL": curr_price - self.config.SL if row["signal"] == 1 else curr_price + self.config.SL,
                    "close_price": np.nan,
                    "close_time": np.nan,
                    "pnl": np.nan
                })
                to_remove.append(i)
        self.order_book.drop(to_remove, inplace=True)

    def generate_signals(self, datetime):
        """Generate trading signals."""
        signals = [
            strategy(self.process_data.loc[self.process_data.index <= datetime]).tail(20) 
            for strategy in self.strategy
        ]

        if np.abs(sum(signals)) < self.config.min_signals:
            return 0
        
        # Validate conflicted signals
        signals = set(signals)
                
        return 

    def run_backtest(self):
        """Run the backtesting simulation."""
        with tqdm(total=len(self.data), desc="Backtesting Progress") as pbar:
            for i in range(len(self.data)):
                datetime = self.data.index[i]
                
                # check if in continuos trading hours
                time = datetime.time()
                if time >= pd.to_datetime('09:15').time() and time <= pd.to_datetime('14:30').time():
                    self.check_margin(price)
                    curr_price = self.data["close"].iloc[i]
                    bid_price = self.data["bid_price"].iloc[i+1] if (self.data["bid_price"].iloc[i+1] is not np.nan) else self.data["bid_price"].iloc[i]
                    ask_price = self.data["ask_price"].iloc[i+1] if (self.data["ask_price"].iloc[i+1] is not np.nan) else self.data["ask_price"].iloc[i]
                    buying_power = self.portfolio.buying_power(curr_price, self.config)
                    
                    self.check_orders(curr_price=curr_price, bid_price=bid_price, ask_price=ask_price,date=datetime)
                    
                    signal = self.generate_signals(datetime) if len(self.portfolio.holdings) < self.config.max_pos else 0
                    if signal != 0:
                        self.place_order(curr_price, signal, datetime)

                    if not self.portfolio.holdings.empty:
                        self.portfolio.force_liquidate(curr_price, self.config, datetime)
                        
                # if time is greater than 2:29 PM, close all positions
                if time >= pd.to_datetime('14:29').time():
                    pnl = self.portfolio._close_all(price, bid_price, ask_price, datetime)
                
                self.data.loc[datetime, "Equity"] = self.portfolio.balance
                pbar.update(1)
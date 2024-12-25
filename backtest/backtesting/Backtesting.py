import pandas as pd
import numpy as np
from utils import processor
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Callable

from ..portfolio import Portfolio
from ..backtest_config import BacktestConfig


class Backtesting:
    """Backtesting Environment"""
    def __init__(self,      
                 strategy: List[Callable], 
                 data: pd.DataFrame, 
                 config: BacktestConfig = None, 
                 interval=1):
        
        assert config is not None, "Config must be provided"
        self.strategy = strategy
        self.data = data
        self.config = config
        self.order_book = pd.DataFrame(columns=["date", "price", "signal", "timeout"])
        self.portfolio = Portfolio(config.initial_balance)
        self.process_data = self._process_data(interval)
        self.data["equity"] = config.initial_balance
        self.data["balance"] = config.initial_balance
        self.prevdate = 0

    def _process_data(self, min):
        """Preprocess and resample data."""
        interval = f"{min}T"
        ohlcv = self.data.resample(interval).agg({
            "price": "ohlc", 
            "volume": "sum"
        }).dropna()
        ohlcv.columns = ohlcv.columns.droplevel(0)
        # #print(ohlcv)
        ohlcv = processor(ohlcv)
        ohlcv = ohlcv.shift(1).dropna().astype(float)

        #print(ohlcv)

        self.data = self.data.loc[ohlcv.index[20]:ohlcv.index[-1]]
        return ohlcv

    def place_order(self, order_price, signal, date):
        """Place a new order in the order book."""
        order = {
            "date": date,
            "price": order_price,
            "signal": signal,
            "timeout": date + pd.Timedelta(minutes=self.config.timeout)
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
        process_data = self.process_data.loc[self.process_data.index <= datetime].tail(20)

        if self.prevdate == process_data.index[-1]:
            return 0
        else:
            self.prevdate = process_data.index[-1]
        
        signals = [
            strategy(process_data)
            for strategy in self.strategy
        ]

        if np.abs(sum(signals)) < self.config.min_signals:
            return 0
        
        # Validate conflicted signals
        signals = set(signals)
                
        return signals

    def run_backtest(self):
        """
            Run the backtesting simulation throught the data.
            Buy at Ask price, exit at Bid price
            Sell at Bid price, Exit at Ask price
        """
        with tqdm(total=len(self.data), desc="Backtesting Progress") as pbar:
            for i in range(len(self.data) - 20):
                datetime = self.data.index[i]
                
                # check if in continuos trading hours
                #print(self.data)
                time = datetime.time()
                curr_price = self.data["price"].iloc[i]
                bid_price = self.data["bid_price"].iloc[i+1] if (self.data["bid_price"].iloc[i+1] is not np.nan) else self.data["bid_price"].iloc[i]
                ask_price = self.data["ask_price"].iloc[i+1] if (self.data["ask_price"].iloc[i+1] is not np.nan) else self.data["ask_price"].iloc[i]
                if time >= pd.to_datetime('09:15').time() and time <= pd.to_datetime('14:30').time():
                    buying_power = self.portfolio.buying_power(curr_price, self.config)
                    
                    self.portfolio.force_liquidate(curr_price, bid_price, ask_price, self.config, datetime)
                    
                    self.check_orders(curr_price=curr_price, bid_price=bid_price, ask_price=ask_price,date=datetime)

                    if buying_power >= 1:
                        signal = self.generate_signals(datetime) if len(self.portfolio.holdings) < self.config.max_pos else 0
                    else:
                        signal = 0
                        
                    if signal != 0:
                        self.place_order(curr_price, signal, datetime)

                    if not self.portfolio.holdings.empty:
                        self.portfolio.force_liquidate(curr_price, self.config, datetime)
                        
                # if time is greater than 2:29 PM, close all positions
                if time >= pd.to_datetime('14:29').time():
                    pnl = self.portfolio._close_all(curr_price, bid_price, ask_price, datetime, self.config)
                
                self.data.loc[datetime, "balance"] = self.portfolio.balance
                self.data.loc[datetime, "equity"] = self.portfolio.balance + self.portfolio._unrealized_pnl(curr_price)
                pbar.update(1)

import pandas as pd
import numpy as np
from utils import processor
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Callable
import logging

from ..portfolio import Portfolio
from ..backtest_config import BacktestConfig


class Backtesting:
    """Backtesting Environment"""
    def __init__(self,      
                 strategy: List[Callable], 
                 data: pd.DataFrame, 
                 config: BacktestConfig = None,
                 search: bool = False
                 ):
        
        assert config is not None, "Config must be provided"
        self.strategy = strategy
        self.data = data
        self.config = config

        self.order_book = pd.DataFrame(columns=["date", "price", "signal", "timeout"])
        self.portfolio = Portfolio(config.initial_balance, config, search=search)
        
        self.process_data = self._process_data(config.interval)
        self.data["equity"] = config.initial_balance
        self.data["balance"] = config.initial_balance
        
        self.prevdate = 0
        self.position_size = 1

    def _process_data(self, min):
        """Preprocess and resample data."""
        interval = f"{min}T"
        ohlcv = self.data.resample(interval).agg({
            "price": "ohlc", 
            "volume": "sum"
        }).dropna()

        ohlcv.columns = ohlcv.columns.droplevel(0)
        # print(ohlcv.head())
        logging.info(f"Resampled data to {min} minutes interval")
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
                    "position_size": self.position_size,
                    "position": curr_price * self.config.margin * self.position_size,
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
        # print(sum(signals))
        signals = sum(signals)

        if self.config.side == 'long':
            return 1 if signals > 0 else 0
        elif self.config.side == 'short':
            return -1 if signals < 0 else 0
        else:
            return signals

    def run_backtest(self, name=''):
        """
            Run the backtesting simulation throught the data.
            Buy at Ask price, exit at Bid price
            Sell at Bid price, Exit at Ask price
        """
        # Extract data as NumPy arrays to speed up access
        data_len = len(self.data)
        prices = self.data["price"].values
        bid_prices = self.data["bid_price"].fillna(method="bfill").values
        ask_prices = self.data["ask_price"].fillna(method="bfill").values
        datetimes = self.data.index
        times = datetimes.time  # Extract time separately

        balance_updates = []
        equity_updates = []

        with tqdm(total=data_len - 20, desc=f"{name}-Progress") as pbar:
            for i in range(data_len - 20):
                datetime = datetimes[i]
                time = times[i]
                curr_price = prices[i]
                bid_price = bid_prices[i + 1] if i + 1 < data_len else bid_prices[i]
                ask_price = ask_prices[i + 1] if i + 1 < data_len else ask_prices[i]

                if self.config.position_size != 1:
                    self.position_size = self.portfolio.position_sizing(curr_price)

                if pd.to_datetime('09:15').time() <= time <= pd.to_datetime('14:30').time():
                    self.portfolio.check_position(curr_price, bid_price, ask_price, datetime)
                    buying_power = self.portfolio.buying_power(curr_price)
                    
                    self.check_orders(curr_price=curr_price, bid_price=bid_price, ask_price=ask_price, date=datetime)

                    if buying_power >= 1 and len(self.portfolio.holdings) < self.config.max_pos:
                        signal = self.generate_signals(datetime)
                    else:
                        signal = 0
                    
                    if signal != 0:
                        self.place_order(curr_price, signal, datetime)

                    if not self.portfolio.holdings.empty:
                        self.portfolio.force_liquidate(curr_price, bid_price, ask_price, datetime)

                # Close all positions after 2:29 PM
                if time >= pd.to_datetime('14:29').time():
                    self.portfolio._close_all(curr_price, bid_price, ask_price, datetime)

                # Store balance and equity updates for bulk assignment
                balance_updates.append((datetime, self.portfolio.balance))
                equity_updates.append((datetime, self.portfolio.balance + self.portfolio._unrealized_pnl(curr_price)))

                if self.portfolio.holdings.empty and self.portfolio.balance < (curr_price * self.config.margin):
                    logging.info("Out of buying power")
                    return

                pbar.update(1)

        # Apply batch updates to the DataFrame **after** the loop
        for x in balance_updates:
            self.data.loc[x[0]:, "balance"] = x[1]
        
        for x in equity_updates:
            self.data.loc[x[0]:, "equity"] = x[1]
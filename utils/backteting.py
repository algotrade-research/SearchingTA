from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

from .processor import processor

class AbstractBacktesting(ABC):
    """
    Abstract class for backtesting

    Attributes:
        data (pd.DataFrame): The historical data used for backtesting.
        initial_balance (float): The initial balance for the backtesting account.
        cost (float): The transaction cost in percentage.
        slippage (float): The slippage in percentage.
        TP (float): The take profit level in percentage.
        SL (float): The stop loss level in percentage.
        max_pos (int): The maximum number of positions allowed.
        position_size (float): The position size as a percentage of the account balance.
        model (object): The model used for generating trading signals.
        window (int): The window size for generating signals.
        MP (bool): Flag indicating whether to use market price or close price for backtesting.
        balance (float): The current balance of the backtesting account.
        _holdings (pd.DataFrame): The dataframe to track open positions.
        _history (pd.DataFrame): The dataframe to track trade history.

    Methods:
        open_order(curr_price, signal, date):
            Abstract method for opening a new order.
        close_order(curr_price, date):
            Abstract method for closing an existing order.
        _close_all(curr_price, date):
            Abstract method for closing all open positions.
        generate_signals(i):
            Abstract method for generating trading signals.
        render():
            Method for rendering the backtesting results plot.
    """
    def __init__(self, 
                 data,
                 initial_balance=10000.0,
                 cost: float=0.07,
                 slippage: float=0.25,
                 TP: float=0.1, 
                 SL: float=-0.1, 
                 max_pos=2,
                 position_size=0.3,
                 model=None, 
                 window=50,
                 MP=True):
        
        self.data = data
        self.position_size = position_size
        self.TP = TP
        self.SL = SL
        self.cost = cost
        self.slippage = slippage
        self.max_pos = max_pos
        self.model = model
        self.window = window
        self.MP = MP
        self.initial_balance = initial_balance
        self.balance = initial_balance  # Track balance

        self._holdings = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", "TP", "SL", "close_price", "close_time", "pnl"
            ])
        self._history = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", "TP", "SL", "close_price", "close_time", "pnl"
            ])

        self.data["Equity"] = self.initial_balance

    @abstractmethod
    def open_order(self, curr_price, signal, date):
        pass 

    @abstractmethod
    def close_order(self, curr_price, date):
        pass

    @abstractmethod
    def _close_all(self, curr_price, date):
        pass

    @abstractmethod
    def generate_signals(self, i):
        pass

    def render(self):
        # Prepare data for plotting
        dates = self.data.index
        prices = self.data['price'].resample("1T").ffill()
        signals = self._history['signal']
        
        # Create a figure
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
        
        fig.add_trace(go.Scatter(x=dates[signals == 'buy'], y=prices[signals == 'buy'], mode='markers', marker=dict(color='green'), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=dates[signals == 'sell'], y=prices[signals == 'sell'], mode='markers', marker=dict(color='red'), name='Sell Signal'))
        
        # Update layout for readability
        fig.update_layout(title='Backtesting Results', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
        
        # Render the plot
        fig.show()



class Backtesting(AbstractBacktesting):
    def __init__(self,
                 strategy: List[Callable], 
                 data, 
                 time_out: int=30,
                 interval="1min",
                 initial_balance=10000,
                 cost=0.5, 
                 slippage=0.25, 
                 TP=1, 
                 SL=1,
                 max_pos=1e19, 
                 position_size=0.3,
                 model=None,
                 window=50,
                 MP=False,
                 min_signals=0):
        super().__init__(data, initial_balance, cost, slippage, TP, SL, max_pos, position_size, model, window, MP)

        self.order_book = pd.DataFrame(columns=["date", "price", "signal", "timeout"])
        self.strategy = strategy
        self.time_out = time_out
        self.min_signals = min_signals
        self.process_data(interval)

    
    def process_data(self, interval):
        self.ohlcv = self.data.resample(interval).agg({
                    "price": "ohlc",
                    "volume": "sum"
                }).dropna()
        self.ohlcv.columns = self.ohlcv.columns.droplevel(0)

        self.data_ohlc = processor(self.ohlcv)

        self.data = self.data.loc[self.data_ohlc.index[0]:]
    
    def open_order(self, curr_price, signal, date) -> None:
        if len(self._holdings) < self.max_pos:
            side = "buy" if signal == 1 else "sell"
            tp = curr_price + self.TP if signal == 1 else curr_price - self.TP
            sl = curr_price - self.SL if signal == 1 else curr_price + self.SL
            new_order = {
                "date": date, 
                "price": curr_price, 
                "signal": side, 
                "position_size": self.position_size, 
                "position": curr_price * self.position_size, 
                "TP": tp, 
                "SL": sl, 
                "close_price": np.nan, "close_time": np.nan,
                "pnl": np.nan
            }
            self._holdings = pd.concat([self._holdings, pd.DataFrame([new_order])], ignore_index=True)

    def close_order(self, curr_price, bid_price, ask_price, date) -> None:
        # Long Position Exit at Bid price.
        # Short Position Exit at Ask price.
        closed_positions = []
        pnl = 0.0
        
        for i, row in self._holdings.iterrows():
            should_close = False
            if row["signal"] == "buy":
                if bid_price >= row["TP"] or bid_price <= row["SL"]:
                    pnl += (bid_price - row["price"])
                    should_close = True
                    row["close_price"] = bid_price

            else:  # sell
                if ask_price <= row["TP"] or ask_price >= row["SL"]:
                    pnl += (row["price"] - ask_price)
                    should_close = True
                    row["close_price"] = ask_price

            if should_close:
                pnl -= self.slippage * 2 if self.MP else 0.0
                pnl -= self.cost
                self.balance += pnl  # Update balance
                
                row["status"] = "closed"
                row["pnl"] = pnl
                
                row["close_time"] = date
                self._history = pd.concat([self._history, pd.DataFrame([row])], ignore_index=True)
                closed_positions.append(i)

        # Remove closed positions from holdings
        self._holdings.drop(closed_positions, inplace=True)

        return pnl
    
    def place_order(self, order_price, signal, date) -> None:
        order = {
            "date": date,
            "price": order_price,
            "signal": signal,
            "timeout": date + pd.Timedelta(minutes=self.time_out)
        }

        self.order_book = pd.concat([self.order_book, pd.DataFrame([order])], ignore_index=True)

    def check_order(self, curr_price, bid_price, ask_price, date) -> None:
        # Long Position: Enter at Ask price, Exit at Bid price.
        # Short Position: Enter at Bid price, Exit at Ask price.
        
        # Check timeout orders
        time_out = []
        for i, row in self.order_book.iterrows():
            if date >= row["timeout"]:
                time_out.append(i)
        
        self.order_book.drop(time_out, inplace=True)
        
        # check fillded orders
        filled = []
        self.order_book.sort_values("date", inplace=True, ascending=False)
        for i, row in self.order_book.iterrows():
            signal = row["signal"]
            if signal == 1 and ask_price >= row["price"]:
                self.open_order(ask_price, row["signal"], date)
                filled.append(i)
            elif signal == -1 and bid_price <= row["price"]:
                self.open_order(bid_price, row["signal"], date)
                filled.append(i)
        
        self.order_book.drop(filled, inplace=True)


    def _close_all(self, curr_price, bid_price, ask_price, date) -> None:
        pnl = 0
        for i, row in self._holdings.iterrows():
            row["close_price"] = bid_price if row["signal"] == "buy" else ask_price
            row["pnl"] = (bid_price - row["price"]) if row["signal"] == "buy" else (row["price"] - ask_price)
            row["close_time"] = date
            self.balance += row["pnl"]
            pnl += row["pnl"] - self.cost
            self._history = pd.concat([self._history, pd.DataFrame([row])], ignore_index=True)

        self._holdings = pd.DataFrame(columns=["date", "price", "signal", "position_size", "position", "TP", "SL", "close_price", "pnl"])
        
        return pnl

    def generate_signals(self, datetime) -> None:
        signals = []

        for strategy in self.strategy:
            signal = strategy(self.data_ohlc.loc[:datetime.round("min")])
            signals.append(signal)

        if len(signals) < self.min_signals:
            return 0
        # Validate conflicted signals
        signals = set(signals)
        
        return sum(signals)
    
    def backtest(self) -> None:
        # Initialize the progress bar
        with tqdm(total=len(self.data - self.window) - 10, desc="Backtesting Progress") as pbar:
            for i in range(self.window, len(self.data) - 10):
                datetime = self.data.index[i]
                # Use loc to update the Equity column
                self.data.loc[self.data.index[i], "Equity"] = self.data.loc[self.data.index[i-1], "Equity"]

                price = self.data["price"].iloc[i+1]
                bid_price = self.data["bid_price"].iloc[i+1] if (self.data["bid_price"].iloc[i+1] is not np.nan) else self.data["bid_price"].iloc[i]
                ask_price = self.data["ask_price"].iloc[i+1] if (self.data["ask_price"].iloc[i+1] is not np.nan) else self.data["ask_price"].iloc[i]
                close_price = self.data_ohlc.close[datetime.floor("min")]

                time = datetime.time()
                if time >= pd.to_datetime('09:15').time() and time <= pd.to_datetime('14:30').time():
                    signal = self.generate_signals(datetime) if time <= pd.to_datetime('14:20').time() else 0

                    if signal != 0:
                        self.place_order(close_price, signal, datetime)
    
                    if len(self._holdings) > 0:
                        pnl = self.close_order(price, bid_price, ask_price, datetime)
                        if pnl != 0:
                            # Update the Equity using loc and iloc for positional indexing
                            self.data.loc[self.data.index[i], "Equity"] = pnl + self.data.loc[self.data.index[i], "Equity"] 
                    
                    if len(self.order_book) > 0:
                        self.check_order(price, bid_price, ask_price, datetime)

                if time >= pd.to_datetime('14:29').time():
                    pnl = self._close_all(price, bid_price, ask_price, datetime)
                    self.data.loc[self.data.index[i], "Equity"] = pnl + self.data.loc[self.data.index[i], "Equity"] 


                Equity = self.data.loc[self.data.index[i], "Equity"]
                # Update the progress bar
                pbar.set_postfix({
                    'Equity': f'{Equity:.2f}',
                    'Open Positions': len(self._holdings),
                    'Orders': len(self.order_book)
                })

                # Update the progress bar
                pbar.update(1)
                
                if Equity < (self.data.loc[self.data.index[0], "Equity"] * 0.5):
                    break

        # self.render()

    
    def get_results(self) -> pd.DataFrame:
        return self._history
    
    def get_equity(self) -> pd.Series:
        return self.data.Equity[self.data.Equity != self.balance]
    
    def evaluate(self) -> pd.DataFrame:
        results = self.get_results()
        equity = self.get_equity().resample("1D").last().dropna()

        VN30F1 = self.data.price.resample("1D").last()
        vn30_result = pd.DataFrame({
            "daily_returns": VN30F1.pct_change().mean(),
            "mdd": (VN30F1 / VN30F1.cummax() - 1).min(),
            "sharpe_ratio": np.sqrt(252) * VN30F1.pct_change().mean() / VN30F1.pct_change().std(),
            "holding_return": (VN30F1[-1] - VN30F1[0]) / VN30F1[0],
            "annualized_return": (VN30F1.pct_change().mean()** 252) - 1
        }, index=["vn30"])

        strategy_result = pd.DataFrame({
            "daily_returns": equity.pct_change().mean(),
            "mdd": (equity / equity.cummax() - 1).min(),
            "sharpe_ratio": np.sqrt(252) * equity.pct_change().mean() / equity.pct_change().std(),
            "holding_return": (equity[-1] - equity[0]) / equity[0],
            "annualized_return": (equity.pct_change().mean() ** 252) - 1
        }, index=["strategy"])

        result = pd.concat([vn30_result, strategy_result], axis=0)

        return result.T

    
class OptimzeBacktesting(AbstractBacktesting):
    def __init__(self,
                 strategy: List[Callable], 
                 data, 
                 time_out: int=30,
                 interval="1min",
                 initial_balance=10000,
                 cost=0.5, 
                 slippage=0.25, 
                 TP=1, 
                 SL=1,
                 max_pos=1e19, 
                 position_size=0.3,
                 model=None,
                 window=50,
                 MP=False,
                 min_signals=2):
        super().__init__(data, initial_balance, cost, slippage, TP, SL, max_pos, position_size, model, window, MP)

        self.order_book = pd.DataFrame(columns=["date", "price", "signal", "timeout"])
        self.strategy = strategy
        self.time_out = time_out
        self.min_signals = min_signals
        self.process_data(interval)

    
    def process_data(self, interval):
        self.ohlcv = self.data.resample(interval).agg({
                    "price": "ohlc",
                    "volume": "sum"
                }).dropna()
        self.ohlcv.columns = self.ohlcv.columns.droplevel(0)

        self.data_ohlc = processor(self.ohlcv)

        self.data = self.data.loc[self.data_ohlc.index[0]:]
    
    def open_order(self, curr_price, signal, date) -> None:
        if len(self._holdings) < self.max_pos:
            side = "buy" if signal == 1 else "sell"
            tp = curr_price + self.TP if signal == 1 else curr_price - self.TP
            sl = curr_price - self.SL if signal == 1 else curr_price + self.SL
            new_order = {
                "date": date, 
                "price": curr_price, 
                "signal": side, 
                "position_size": self.position_size, 
                "position": curr_price * self.position_size, 
                "TP": tp, 
                "SL": sl, 
                "close_price": np.nan, "close_time": np.nan,
                "pnl": np.nan
            }
            self._holdings = pd.concat([self._holdings, pd.DataFrame([new_order])], ignore_index=True)

    def close_order(self, curr_price, bid_price, ask_price, date) -> None:
        # Long Position Exit at Bid price.
        # Short Position Exit at Ask price.
        closed_positions = []
        pnl = 0.0
        
        for i, row in self._holdings.iterrows():
            should_close = False
            if row["signal"] == "buy":
                if bid_price >= row["TP"] or bid_price <= row["SL"]:
                    pnl += (bid_price - row["price"])
                    should_close = True
                    row["close_price"] = bid_price

            else:  # sell
                if ask_price <= row["TP"] or ask_price >= row["SL"]:
                    pnl += (row["price"] - ask_price)
                    should_close = True
                    row["close_price"] = ask_price

            if should_close:
                pnl -= self.slippage * 2 if self.MP else 0.0
                pnl -= self.cost
                self.balance += pnl  # Update balance
                
                row["status"] = "closed"
                row["pnl"] = pnl
                
                row["close_time"] = date
                self._history = pd.concat([self._history, pd.DataFrame([row])], ignore_index=True)
                closed_positions.append(i)

        # Remove closed positions from holdings
        self._holdings.drop(closed_positions, inplace=True)

        return pnl
    
    def place_order(self, order_price, signal, date) -> None:
        order = {
            "date": date,
            "price": order_price,
            "signal": signal,
            "timeout": date + pd.Timedelta(minutes=self.time_out)
        }

        self.order_book = pd.concat([self.order_book, pd.DataFrame([order])], ignore_index=True)

    def check_order(self, curr_price, bid_price, ask_price, date) -> None:
        # Long Position: Enter at Ask price, Exit at Bid price.
        # Short Position: Enter at Bid price, Exit at Ask price.
        
        # Check timeout orders
        time_out = []
        for i, row in self.order_book.iterrows():
            if date >= row["timeout"]:
                time_out.append(i)
        
        self.order_book.drop(time_out, inplace=True)
        
        # check fillded orders
        filled = []
        self.order_book.sort_values("date", inplace=True, ascending=False)
        for i, row in self.order_book.iterrows():
            signal = row["signal"]
            if signal == 1 and ask_price >= row["price"]:
                self.open_order(ask_price, row["signal"], date)
                filled.append(i)
            elif signal == -1 and bid_price <= row["price"]:
                self.open_order(bid_price, row["signal"], date)
                filled.append(i)
        
        self.order_book.drop(filled, inplace=True)


    def _close_all(self, curr_price, bid_price, ask_price, date) -> None:
        pnl = 0
        for i, row in self._holdings.iterrows():
            row["close_price"] = bid_price if row["signal"] == "buy" else ask_price
            row["pnl"] = (bid_price - row["price"]) if row["signal"] == "buy" else (row["price"] - ask_price)
            row["close_time"] = date
            self.balance += row["pnl"]
            pnl += row["pnl"] - self.cost
            self._history = pd.concat([self._history, pd.DataFrame([row])], ignore_index=True)

        self._holdings = pd.DataFrame(columns=["date", "price", "signal", "position_size", "position", "TP", "SL", "close_price", "pnl"])
        
        return pnl

    def generate_signals(self, datetime) -> None:
        signals = []

        for strategy in self.strategy:
            signal = strategy(self.data_ohlc.loc[:datetime.round("min")])
            signals.append(signal)

        # Validate conflicted signals
        if len(signals) < self.min_signals:
            return 0
        signals = set(signals)
        
        return sum(signals)
    
    def backtest(self) -> None:
        # Initialize the progress bar
        with tqdm(total=len((self.data - self.window)) - 10, desc="Backtesting Progress") as pbar:
            for i in range(self.window, len(self.data) - 10):
                datetime = self.data.index[i]
                # Use loc to update the Equity column
                self.data.loc[self.data.index[i], "Equity"] = self.data.loc[self.data.index[i-1], "Equity"]

                price = self.data["price"].iloc[i+1]
                bid_price = self.data["bid_price"].iloc[i+1] if (self.data["bid_price"].iloc[i+1] is not np.nan) else self.data["bid_price"].iloc[i]
                ask_price = self.data["ask_price"].iloc[i+1] if (self.data["ask_price"].iloc[i+1] is not np.nan) else self.data["ask_price"].iloc[i]
                close_price = self.data_ohlc.close[datetime.floor("min")]

                time = datetime.time()
                if time >= pd.to_datetime('09:15').time() and time <= pd.to_datetime('14:30').time():
                    signal = self.generate_signals(datetime) if time <= pd.to_datetime('14:20').time() else 0

                    if signal != 0:
                        self.place_order(close_price, signal, datetime)
    
                    if len(self._holdings) > 0:
                        pnl = self.close_order(price, bid_price, ask_price, datetime)
                        if pnl != 0:
                            # Update the Equity using loc and iloc for positional indexing
                            self.data.loc[self.data.index[i], "Equity"] = pnl + self.data.loc[self.data.index[i], "Equity"] 
                    
                    if len(self.order_book) > 0:
                        self.check_order(price, bid_price, ask_price, datetime)

                if time >= pd.to_datetime('14:29').time():
                    pnl = self._close_all(price, bid_price, ask_price, datetime)
                    self.data.loc[self.data.index[i], "Equity"] = pnl + self.data.loc[self.data.index[i], "Equity"] 

                try:
                    winrate = (len(self._history["pnl"][self._history["pnl"] > 0]) / len(self._history["pnl"])) * 100
                    mean_pnl = self._history["pnl"].mean()
                except Exception as e:
                    winrate = 0
                    mean_pnl = 0
                    
                # Update the progress bar
                pbar.set_postfix({
                    'winrate': f'{winrate:.2f}%',
                    "mean_pnl": f'{mean_pnl:.2f}',
                    'holdings': len(self._holdings),
                    'orders': len(self.order_book)
                })
                pbar.update(1)                                    

                # Update the progress bar
                
                # if Equity < (self.data.loc[self.data.index[0], "Equity"] * 0.5):
                #     break

        # self.render()

    
    def get_results(self) -> pd.DataFrame:
        return self._history
    
    def get_equity(self) -> pd.Series:
        return self.data.Equity[self.data.Equity != self.balance]
    
    def evaluate(self) -> pd.DataFrame:
        results = self.get_results()
        equity = self.get_equity().resample("1D").last().dropna()

        VN30F1 = self.data.price.resample("1D").last()
        vn30_result = pd.DataFrame({
            "daily_returns": VN30F1.pct_change().mean(),
            "mdd": (VN30F1 / VN30F1.cummax() - 1).min(),
            "sharpe_ratio": np.sqrt(252) * VN30F1.pct_change().mean() / VN30F1.pct_change().std(),
            "holding_return": (VN30F1[-1] - VN30F1[0]) / VN30F1[0],
            "annualized_return": (VN30F1.pct_change().mean()** 252) - 1
        }, index=["vn30"])

        strategy_result = pd.DataFrame({
            "daily_returns": equity.pct_change().mean(),
            "mdd": (equity / equity.cummax() - 1).min(),
            "sharpe_ratio": np.sqrt(252) * equity.pct_change().mean() / equity.pct_change().std(),
            "holding_return": (equity[-1] - equity[0]) / equity[0],
            "annualized_return": (equity.pct_change().mean() ** 252) - 1
        }, index=["strategy"])

        result = pd.concat([vn30_result, strategy_result], axis=0)

        return result.T

import pandas as pd
from ..backtest_config import BacktestConfig

class Portfolio:
    """
        Portfolio class to manage the positions and balance of the trading account.
    """
    def __init__(self, initial_balance: float, config: BacktestConfig, search: bool = False):
        self.balance = initial_balance
        self.config = config
        self.search = search
        self.holdings = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", 
            "TP", "SL", "close_price", "close_time", "pnl"
        ])
        self.history = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", 
            "TP", "SL", "close_price", "close_time", "pnl"
        ])

    def add_position(self, position):
        """Add a new position to the portfolio."""
        self.holdings = pd.concat([self.holdings, pd.DataFrame([position])], ignore_index=True)
    
    def buying_power(self, curr_price):
        required_margin = self.config.margin * curr_price * len(self.holdings)
        equity = self.balance + self._unrealized_pnl(curr_price)
        available = equity - required_margin
        return int(available / (curr_price * self.config.margin))
            
    def close_position(self, index, bid_price, ask_price, date):
        """Close a position and update the portfolio."""
        # print('Start Function')
        # print(self.holdings)
        row = self.holdings.iloc[index]
        close_price = bid_price if row["signal"] == "buy" else ask_price
        pnl = self._calculate_pnl(row, bid_price, ask_price)
        # print(f"""
        #     Close Price: {close_price}
        #     {row["signal"]} Price: {row["price"]}
        #     pnl: {(bid_price - row["price"]) if row["signal"] == "buy" else (row["price"] - ask_price)}
        #     PnL: {pnl}
        # """)   

        # print(f"Calculated pnl: {pnl} for position {index}")
        # print(f"Balance before update: {self.balance}")
        self.balance += pnl
        # print(f"Balance after update: {self.balance}")

        row["close_price"] = close_price
        row["close_time"] = date
        row["pnl"] = pnl
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)

    def _calculate_pnl(self, row,  bid_price, ask_price):
        """Calculate the profit or loss for a position."""
        pnl = (bid_price - row["price"]) if row["signal"] == "buy" else (row["price"] - ask_price)
        # print(row["position_size"])

        pnl -= self.config.slippage
        pnl -= self.config.cost * 2
        
        if self.search:
            return pnl 
        else:
            pnl *= row["position_size"]
        return pnl

    def force_liquidate(self, curr_price, bid_price, ask_price, date):
        """Force liquidation to meet margin requirements."""
        while not self.holdings.empty and not self._meets_margin(curr_price):
            self.holdings["unrealized_pnl"] = self.holdings.apply(
                lambda row: self._calculate_pnl(row, bid_price, ask_price), axis=1
            )
            to_liquidate = self.holdings["unrealized_pnl"].idxmin()
            self.close_position(to_liquidate,  bid_price, ask_price, date)
            self.holdings.drop(index=to_liquidate, inplace=True)          

    def _meets_margin(self, curr_price):
        """Check if the portfolio meets margin requirements."""
        required_margin = self.config.margin * curr_price * len(self.holdings)
        equity = self.balance + self._unrealized_pnl(curr_price)
        return equity >= required_margin

    def _unrealized_pnl(self, curr_price):
        """Calculate the unrealized PnL."""
        return sum(
            (curr_price - row["price"]) * row["position_size"] if row["signal"] == "buy" else
            (row["price"] - curr_price) * row["position_size"]
            for _, row in self.holdings.iterrows()
        )
    
    def _close_all(self, curr_price, bid_price, ask_price, date) -> float:
        """Close all positions and update the portfolio."""
        pnl = 0
        for i, row in self.holdings.iterrows():
            row["close_price"] = bid_price if row["signal"] == "buy" else ask_price
            row["pnl"] = self._calculate_pnl(row, bid_price, ask_price)
            
            # print(self.balance)
            self.balance += row["pnl"] - self.config.cost * 2
            # print(self.balance)
            
            row["close_time"] = date
            pnl += row["pnl"] - self.config.cost * 2
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)

        self.holdings = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", 
            "TP", "SL", "close_price", "close_time", "pnl"
        ])

        return pnl
    
    def check_position(self, curr_price, bid_price, ask_price, date):
        """Check if the portfolio has reached the maximum number of positions."""
        to_close = []
        for i, row in self.holdings.iterrows():
            if (
                (row["signal"] == "buy" and (bid_price >= row["TP"] or bid_price <= row['SL'])) or
                (row["signal"] == "sell" and (ask_price <= row['TP'] or ask_price >= row['SL']))
            ):
                to_close.append(i)

        for index in to_close:
            self.close_position(index, bid_price, ask_price, date)

        # Drop positions that were closed
        self.holdings.drop(index=to_close, inplace=True)
        self.holdings.reset_index(drop=True, inplace=True)
    
    def position_sizing(self, curr_price):
        """Calculate the position size based on the current balance."""
        new_size = int((self.balance * self.config.position_size) / (curr_price * self.config.margin))
        
        if new_size < 1 and self.balance > (curr_price * self.config.margin):
            return 1
        
        return new_size
import pandas as pd

class Portfolio:
    """
        Portfolio class to manage the positions and balance of the trading account.
    """
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
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
    
    def buying_power(self, curr_price, config):
        required_margin = config.margin * curr_price * len(self.holdings)
        equity = self.balance + self._unrealized_pnl(curr_price)
        available = equity - required_margin
        
        return int(available / (curr_price * config.margin))
        
    def close_position(self, index, bid_price, ask_price, date, config):
        """Close a position and update the portfolio."""
        row = self.holdings.iloc[index]
        close_price = bid_price if row["signal"] == "buy" else ask_price
        pnl = self._calculate_pnl(row, bid_price, ask_price, config)

        print(self.balance)
        self.balance += pnl
        print(self.balance)

        row["close_price"] = close_price
        row["close_time"] = date
        row["pnl"] = pnl
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)

    def _calculate_pnl(self, row,  bid_price, ask_price, config):
        """Calculate the profit or loss for a position."""
        pnl = (bid_price - row["price"]) if row["signal"] == "buy" else (row["price"] - ask_price)
        pnl *= row["position_size"]
        pnl -= config.slippage * 2
        pnl -= config.cost
        return pnl

    def force_liquidate(self, curr_price, bid_price, ask_price, config, date):
        """Force liquidation to meet margin requirements."""
        while not self.holdings.empty and not self._meets_margin(curr_price, config):
            self.holdings["unrealized_pnl"] = self.holdings.apply(
                lambda row: self._calculate_pnl(row, bid_price, ask_price, config), axis=1
            )
            to_liquidate = self.holdings["unrealized_pnl"].idxmin()
            self.close_position(to_liquidate,  bid_price, ask_price, date, config)
            self.holdings.drop(index=to_liquidate, inplace=True)          

    def _meets_margin(self, curr_price, config):
        """Check if the portfolio meets margin requirements."""
        required_margin = config.margin * curr_price * len(self.holdings)
        equity = self.balance + self._unrealized_pnl(curr_price)
        return equity >= required_margin

    def _unrealized_pnl(self, curr_price):
        """Calculate the unrealized PnL."""
        return sum(
            (curr_price - row["price"]) * row["position_size"] if row["signal"] == "buy" else
            (row["price"] - curr_price) * row["position_size"]
            for _, row in self.holdings.iterrows()
        )
    
    def _close_all(self, curr_price, bid_price, ask_price, date, config) -> float:
        """Close all positions and update the portfolio."""
        pnl = 0
        for i, row in self.holdings.iterrows():
            row["close_price"] = bid_price if row["signal"] == "buy" else ask_price
            row["pnl"] = self._calculate_pnl(row, bid_price, ask_price, config)
            
            print(self.balance)
            self.balance += row["pnl"] - config.cost
            print(self.balance)
            
            row["close_time"] = date
            pnl += row["pnl"] - config.cost
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)

        self.holdings = pd.DataFrame(columns=[
            "date", "price", "signal", "position_size", "position", 
            "TP", "SL", "close_price", "close_time", "pnl"
        ])

        return pnl

    def check_position(self, curr_price, bid_price, ask_price, date, config):
        """Check if the portfolio has reached the maximum number of positions."""
        to_close = []
        for i, row in self.holdings.iterrows():
            if (
                (row["signal"] == "buy" and (bid_price >= row["TP"] or bid_price <= row['SL'])) or
                (row["signal"] == "sell" and (ask_price <= row['TP'] or ask_price >= row['SL']))
            ):
                to_close.append(i)
        for i in to_close:    
            self.close_position(i, bid_price, ask_price, date, config)
        self.holdings.drop(to_close, inplace=True)
            
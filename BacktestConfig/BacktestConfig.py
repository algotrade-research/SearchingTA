
class BacktestConfig:
    """
        Configuration for the backtesting simulation.
        - TP: Take Profit level
        - SL: Stop Loss level
        - max_pos: Maximum number of positions
        - position_size: Position size as a percentage of the total equity
        - margin: Margin requirement as a percentage of the total equity
        - min_signals: Minimum number of signals required to place a trade
    """
    def __init__(self, 
                 initial_balance: float=10000.0, 
                 cost: float=0.07, 
                 slippage: float | None =None, 
                 TP: float=None, 
                 SL: float=None, 
                 max_pos: int=5, 
                 position_size: int=1, 
                 margin: float=0.25,
                 min_signals = 2,
                 mode: 'one_way' | 'hedged'= 'one_way',
                 side: 'long' | 'short' = None):
        assert 1 <= position_size, f"Position size must be between 0 and 1. {position_size}"
        assert 0.2 <= margin <= 1, f"Margin must be between 0 and 1. {margin}"
        assert TP is not None and SL is not None, "TP and SL must be provided."
        assert (mode=='one_way' and side is not None) or mode=='hedged', "Side must be provided for One
        
        self.initial_balance = initial_balance
        self.cost = cost
        self.slippage = slippage
        self.TP = TP
        self.SL = SL
        self.max_pos = max_pos
        self.position_size = position_size
        self.margin = margin
        self.min_signals = 2
        self.mode = mode
        self.side = side
    
    def __str__(self):
        return f"""
            Initial Balance: {self.initial_balance}
            Cost: {self.cost}
            Slippage: {self.slippage}
            Take Profit: {self.TP}
            Stop Loss: {self.SL}
            Maximum Positions: {self.max_pos}
            Position Size: {self.position_size}
            Margin: {self.margin}
            Minimum Signals: {self.min_signals}
            Mode: {self.mode}
            {f"Side: {self.side}" if self.mode == 'one_way' else ""}
        """
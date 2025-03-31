import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Callable

class Report:
    def __init__(self,
                 data: pd.DataFrame,
                 history: pd.DataFrame,
                 equity: pd.Series,
                 balance: pd.Series,
                 benchmark: pd.DataFrame
                 ):
        self.data = data    
        self.history = history
        self.equity = equity    
        self.balance = balance
        self.benchmark = benchmark

    def plot_price_signal(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['price'], mode='lines', name='Price'))
        
        buy_signals = self.history[self.history['signal'] == 1]

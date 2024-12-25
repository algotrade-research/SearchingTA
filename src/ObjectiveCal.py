from backtest import BacktestConfig
import pandas as pd

class ObjectiveCalculator:
    @staticmethod
    def calculate(history: pd.DataFrame, config: BacktestConfig, TP: float, SL: float) -> float:
        break_even_prob = (SL + 2 * config.cost) / (SL + TP)
        expected_pnl = TP * break_even_prob

        pnl = history['pnl']
        mean_pnl = pnl.mean()
        winrate = (pnl > 0).astype(int).mean() * 100

        loss = (winrate / (break_even_prob * 100)) + (mean_pnl / expected_pnl)
        return loss
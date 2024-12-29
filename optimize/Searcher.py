from typing import List, Callable, Tuple
import warnings
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from utils import *
from strategy import *
from backtest import *
import optuna

class Searching:
    """
        Seaching Potential Strategies
    """
    def __init__(self, 
                 number_of_trials: int,
                 dir: str = 'searching',
                 data: pd.DataFrame = None,
                 SL: Tuple[float, float] = (1, 10),
                 TP: Tuple[float, float] = (1, 10),
                 side: ['long', 'short'] = None,
                 mode: ['one_way', 'hedged'] = 'one_way',
                 n_jobs: int = 2
                 ):
        
        assert data is not None, "Data must be provided"
        assert len(data) > 0, "Data must not be empty"
        assert len(data.columns) > 0, "Data must have columns"
        assert (mode == 'one_way' and side is not None) or mode == 'hedged', "Side must be provided for One way"
        assert side in ['long', 'short', None], "Side must be either 'long', 'short' or None"

        self._dir: str = dir
        self.TP = TP
        self.SL = SL
        self.side = side
        self.mode = mode
        self.n_jobs = n_jobs

        # initialize directory
        os.makedirs(dir, exist_ok=True)
        initialize_logging(dir)

        self.number_of_trials: int = number_of_trials
        self.data: pd.DataFrame = data
        self.bt = None

    def _configure(self,
                  strategies: List[Callable],
                  cost: float=0.25,
                  slippage: float = 0, 
                  TP: float=None, 
                  SL: float=None, 
                  position_size: int=1, 
                  margin: float=0.25,
                  min_signals = 2,
                  interval: int=1,
                  mode: ['one_way', 'hedged']= 'one_way',
                  side: ['long', 'short'] = None):
        """
            _Configure the backtesting environment
        """

        self.bt_config = BacktestConfig(
            cost=cost,
            slippage=slippage,
            max_pos=10e9,
            TP=TP,
            SL=SL,
            position_size=position_size,
            margin=margin,
            side=side,
            min_signals=min_signals,
            initial_balance=10e19,
            mode=mode,
            interval=interval
        )
        
        self.bt = Backtesting(
            strategy=strategies,
            data=self.data,
            config= self.bt_config
        )
        return f"""
            Strategies: {[strategy.__name__ for strategy in strategies]}
        """ + str(self.bt_config)
    
    def objective(self, history: pd.DataFrame, TP: float, SL: float) -> float:
        """
            Objective function to optimize 
            
            The goal here is to find a profitable strategy which has a high winrate
            and a high mean PnL

            The objective function is defined as the loss, which is calculated as follow:
                - break_even_prob = (SL + 2*cost) / (SL + TP)
                - expected_pnl = TP * break_even_prob
                - mean_pn: which is the profit and loss of the strategy
                - winrate: the percentage of winning trades

                objective = (winrate / break_even_prob - 1) + (mean_pnl / expected_pnl - 1)
                
            The find the optimal strategy, we by maximizing the objective function
        """
        break_even_prob = (SL + 2*self.bt_config.cost) / (SL + TP)
        expected_pnl = TP * break_even_prob
        
        pnl = history['pnl']
        mean_pnl = pnl.mean()
        winrate = (pnl > 0).astype(int).mean() * 100

        loss = (winrate / (break_even_prob * 100)) + (mean_pnl / expected_pnl)

        return loss

    def seaching_objective(self, trial):
        TP = trial.suggest_float("TP", self.TP[0], self.TP[1], step=0.5)
        SL = trial.suggest_float("SL", self.SL[0], self.SL[1], step=0.5)
        interval = trial.suggest_int("interval", 1, 60)

        selected_strategies = []
        for strategy_name, strategy_function in strategy_options:
            if trial.suggest_categorical(strategy_name, [True, False]):
                selected_strategies.append(strategy_function)

        self._configure(
            strategies=selected_strategies,
            TP=TP,
            SL=SL,
            side=self.side,
            mode=self.mode,
            interval=interval
        )

        assert self.bt is not None, "Backtesting environment must be _configured"

        try:
            self.bt.run_backtest(name=trial.number)
            history = self.bt.portfolio.history

            if len(history) == 0:
                return float('-inf')

            balance = self.bt.data['balance']
            equity = self.bt.data['equity']
            
            os.makedirs(os.path.join(self._dir, str(trial.number)), exist_ok=True)
            
            # save the history, nav and equity
            history.to_csv(os.path.join(self._dir, str(trial.number), "history.csv"))
            balance.to_csv(os.path.join(self._dir + '/' + str(trial.number), "balance.csv"))
            equity.to_csv(os.path.join(self._dir + '/' + str(trial.number), "equity.csv"))
            params = {
                "TP": TP,
                "SL": SL,
                "strategies": [strategy.__name__ for strategy in selected_strategies],
                "interval": interval
            }

            loss = self.objective(history, TP, SL)

            logging.info(f"Trial {trial.number} - Strategies: {selected_strategies}, TP: {TP}, SL: {SL} - Loss: {loss}")

            with open(os.path.join(self._dir + '/' + str(trial.number), "params.py"), "w") as file:
                file.write(f'# Parameters {loss}\n')
                file.write(f'params = {str(params)}')
            return loss
        
        except Exception as e:
            logging.error(f"Error: {e}")
            return -float('inf') 

    def run(self, name: str='') -> optuna.study.Study:

        try:
            study = optuna.create_study(direction="maximize",
                                        study_name=f"searching_{name}",
                                        storage="sqlite:///searching.db", 
                                        load_if_exists=True)    

            study.optimize(self.seaching_objective, n_trials=self.number_of_trials, n_jobs=self.n_jobs)

            best_params = study.best_params
            best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

            with open(f"{self._dir}/best_params.log", "w") as file:
                file.write("Best Trial: " + str(study.best_trial.number) + "\n")
                file.write("Best Loss: " + str(study.best_value) + "\n")
                file.write("Best Parameters:\n" + best_params_str)
            
            return study
        except Exception as e:
            logging.error(f"Error: {e}")
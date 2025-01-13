from typing import List, Callable, Tuple
import warnings
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from utils import *
from strategy import *
from backtest import *
import optuna
import random

class Optimizer:
    """
        Seaching Potential Strategies
    """
    def __init__(self,
                 trial: int,
                 path: str,
                 number_of_trials: int,
                 dir: str = 'optimizing',
                 data: pd.DataFrame = None,
                 SL: Tuple[float, float] = (-3, 3),
                 TP: Tuple[float, float] = (-3, 3),
                 side: ['long', 'short'] = None,
                 cost: float = 0.25,
                 slippage: float = 0.47,
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
        self.slippage = slippage

        # initialize directory
        os.makedirs(dir, exist_ok=True)
        initialize_logging(dir)

        self.number_of_trials: int = number_of_trials
        self.data: pd.DataFrame = data
        self.cost = cost
        self.bt = None

        self._read_params(trial, path)
        self._validate()

        np.random.seed(42)
        random.seed(42)

    def _validate(self):
        self.TP = (-(self.params['TP'] - 1) if self.params['TP'] + self.TP[0] < 0 else self.TP[0], self.TP[1])
        self.SL = (-(self.params['SL'] - 1) if self.params['SL'] + self.SL[0] < 0 else self.SL[0], self.SL[1])

    def _read_params(self, trial: int, path: str):
        params_file_path = os.path.join(path, str(trial), "params.py")
        try:
            with open(params_file_path, "r") as file:
                exec_globals = {}
                exec(file.read(), exec_globals) 
                
                self.params = exec_globals.get("params", None)
                
                if self.params is None:
                    raise ValueError("The 'params' variable was not found in the provided file.")
                
                print(f"Successfully read params: {self.params}")
        except FileNotFoundError:
            print(f"Error: File not found at {params_file_path}")
            exit(1)
        except Exception as e:
            print(f"Error reading params: {e}")
            exit(2)

    def _configure(self,
                  strategies: List[Callable],
                  slippage: float = 0, 
                  TP: float=None, 
                  SL: float=None,
                  position_size: int=0.1, 
                  margin: float=0.25,
                  min_signals = 2,
                  max_pos: int=10,
                  mode: ['one_way', 'hedged']= 'one_way',
                  interval: int=1,
                  side: ['long', 'short'] = None):
        """
            _Configure the backtesting environment
        """
        min_balance = (1200 * margin) * 12
        balance = max_pos * (1200 * margin) * (1 / position_size) * 1.5

        balance = balance if balance > min_balance else min_balance

        self.bt_config = BacktestConfig(
            interval=interval,
            initial_balance=balance,
            cost=self.cost,
            slippage=slippage,
            max_pos=max_pos,
            TP=TP,
            SL=SL,
            position_size=position_size,
            margin=margin,
            side=side,
            min_signals=min_signals,
            mode=mode
        )
        
        self.bt = Backtesting(
            strategy=strategies,
            data=self.data,
            config= self.bt_config
        )

        return f"""
            Strategies: {[strategy.__name__ for strategy in strategies]}
        """ + str(self.bt_config)
    
    def objective(self, balance: pd.Series) -> float:
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

        p_returns = balance.resample('1D').last().pct_change().dropna()
        mean_p_returns = p_returns.mean()
        std_p_returns = p_returns.std()

        sharpe_ratio = np.sqrt(252) * mean_p_returns / std_p_returns
        
        alpha = 0.5
        beta = 0.5
        loss =  (sharpe_ratio / 2 - 1) * beta + (mean_p_returns * 252 / 0.15 - 1) * alpha

        return loss
    def seaching_objective(self, trial):
        TP = self.params['TP'] + trial.suggest_float("TP", self.TP[0], self.TP[1], step=0.1)
        SL = self.params['SL'] + trial.suggest_float("SL", self.SL[0], self.SL[1], step=0.5)
        pos_size = trial.suggest_float("position_size", 0.05, 0.5, step=0.05)
        max_pos = trial.suggest_int("max_pos", 1, 10)
        min_signals = trial.suggest_int("min_signals", 2, 5)
    
        
        selected_strategies = []
        for strategy_name, strategy_function in strategy_options:
            if strategy_name in self.params['strategies']:
                if trial.suggest_categorical(strategy_name, [True, False]):
                    selected_strategies.append(strategy_function)

        self._configure(
            max_pos=max_pos,
            min_signals=min_signals,
            position_size=pos_size,
            TP=TP,
            SL=SL,
            interval=self.params['interval'],
            
            # Base Parameters
            side=self.side,
            mode=self.mode,
            slippage=self.slippage,
            strategies=selected_strategies,
        )

        assert self.bt is not None, "Backtesting environment must be _configured"

        # try:
        self.bt.run_backtest(name=trial.number)
        history = self.bt.portfolio.history

        if len(history) == 0:
            return float('-inf')

        balance = self.bt.data['balance'].fillna(method='ffill')
        equity = self.bt.data['equity'].fillna(method='ffill')

        # print('balance', balance)
        # print('equity', equity)
        
        os.makedirs(os.path.join(self._dir, str(trial.number)), exist_ok=True)
        
        # save the history, nav and equity
        history.to_csv(os.path.join(self._dir, str(trial.number), "history.csv"))
        balance.to_csv(os.path.join(self._dir + '/' + str(trial.number), "balance.csv"))
        equity.to_csv(os.path.join(self._dir + '/' + str(trial.number), "equity.csv"))

        params = {
            "TP": TP,
            "SL": SL,
            'position_size': pos_size,
            'max_pos': max_pos,
            'min_signals': min_signals,
            'interval': self.params['interval'],
            'side': self.side,
            'mode': self.mode,
            "strategies": [strategy.__name__ for strategy in selected_strategies]
        }

        loss = self.objective(balance)

        logging.info(f"Trial {trial.number} - Strategies: {selected_strategies}, TP: {TP}, SL: {SL} - Loss: {loss}")

        with open(os.path.join(self._dir + '/' + str(trial.number), "params.py"), "w") as file:
            file.write(f'# Parameters {loss}\n')
            file.write(f'params = {str(params)}')
            
        return loss
        
        # except Exception as e:
        #     logging.error(f"Error: {e}")
        #     return -float('inf') 

    def run(self, name: str=''):

        try:
            np.random.seed(42)
            random.seed(42)
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(sampler=sampler,
                                        direction="maximize",
                                        study_name=f"optimizing_{name}",
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
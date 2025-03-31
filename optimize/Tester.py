from typing import List, Callable, Tuple
import warnings
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from utils import *
from strategy import *
from backtest import *
import optuna

class Tester:
    """
        Testing Strategies
    """
    def __init__(self,
                 trial_num: int,
                 path: str,
                 dir: str = 'testing',
                 data: pd.DataFrame = None,
                 cost: float = 0.25
                 ):
        
        assert data is not None, "Data must be provided"
        assert len(data) > 0, "Data must not be empty"
        assert len(data.columns) > 0, "Data must have columns"
        self._dir: str = dir

        # initialize directory
        os.makedirs(dir, exist_ok=True)
        initialize_logging(dir)

        self.data: pd.DataFrame = data
        self.bt: Backtesting = None
        
        self.cost = cost
        
        self.trial_num = trial_num
        self._read_params(trial_num, path)

    def _read_params(self, trial: int, path: str):
        with open(os.path.join(path, str(trial), "params.py"), "r") as file:
            '''
            params.py
                params = { "TP": 1, "SL": 1, "position_size": 0.1, "max_pos": 1, "min_signals": 2, "strategies": ['strategy1', 'strategy2']}
            '''
            exec_globals = {}
            exec(file.read(), exec_globals)
            params = exec_globals.get("params", None)  # Safely retrieve 'params' from the executed 
            
        print(params)
        strrategy_func = []
        for strategy_name, strategy_function in strategy_options:
            if strategy_name in params['strategies']:
                strrategy_func.append(strategy_function)
        params['strategies'] = strrategy_func
        self.params = params
    
    def _configure(self,
                  strategies: List[Callable],
                  slippage: float = 0, 
                  TP: float=None, 
                  SL: float=None,
                  position_size: int=0.1, 
                  margin: float=0.25,
                  min_signals = 2,
                  max_pos: int=10,
                  interval: int=1,
                  mode: ['one_way', 'hedged']= 'one_way',
                  side: ['long', 'short'] = None):
        """
            _Configure the backtesting environment
        """
        min_balance = (1200 * margin) * 12
        balance = max_pos * (1200 * margin) * (1 / position_size) * 1.5

        balance = balance if balance > min_balance else min_balance

        self.bt_config = BacktestConfig(
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
    
    def evaluate(self, history: pd.Series, TP: float, SL: float, equity: pd.Series, balance: pd.Series,Benchmark: pd.DataFrame = None) -> float:
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

    def run(self, name=''):
        TP = self.params['TP'] 
        SL = self.params['SL']
        pos_size = self.params['position_size'] 
        max_pos = self.params['max_pos']
        min_signals = self.params['min_signals'] 
        interval = self.params['interval']
        side = self.params['side']
        strategy = self.params['strategies']

    
        self._configure(
            max_pos=2,
            min_signals=min_signals,
            position_size=pos_size,
            TP=TP,
            SL=SL,
            interval=interval,
            slippage=0.47,

            # Base Parameters
            side=side,
            mode='one_way',
            strategies=strategy,
        )

        assert self.bt is not None, "Backtesting environment must be _configured"

        try:
            self.bt.run_backtest(name=name)
            history = self.bt.portfolio.history
            path = '/' + str(self.trial_num) + '/'
            # print(path)
            # self.bt.data.to_csv("test_data.csv")
            # if len(history) == 0:
            #     return 0

            balance = self.bt.data['balance']
            equity = self.bt.data['equity']
            
            os.makedirs(os.path.join(self._dir, str(self.trial_num)), exist_ok=True)
            
            # save the history, nav and equity
            history.to_csv(os.path.join(self._dir + '/' + str(self.trial_num), "history.csv"))
            balance.to_csv(os.path.join(self._dir + '/' + str(self.trial_num), "balance.csv"))
            equity.to_csv(os.path.join(self._dir + '/' + str(self.trial_num), "equity.csv"))
            print(balance)
        
        except Exception as e:
            logging.error(f"Error: {e}")
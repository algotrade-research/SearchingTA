from optimize import Searching, Optimizer, Tester
from utils import *
import pandas as pd
import numpy as np
import optuna
import os

np.random.seed(42)

# Number of Trials
num_trials = 500 #int(input('Enter the number of trials: '))

# Configuration for the optimization
config = {
    'number_of_trials': num_trials,
    'side': 'short',
    'n_jobs': 1,
    'cost': 0.25,
    'slippage': 0.47,
}

# main directory
main_dir = 'result_short'

# Data Collection
start_date = '2018-01-01'
end_date = '2025-01-10'

if os.path.exists(f'{start_date}-{end_date}.csv'):
    data = pd.read_csv(f'{start_date}-{end_date}.csv')
    data.set_index('datetime', inplace=True)
    data.index = pd.to_datetime(data.index)
else :
    downloader = Downloader()
    print('Download Data')
    data = downloader.get_historical_data(start_date=start_date, end_date=end_date)
    data.to_csv(f'{start_date}-{end_date}.csv')




print(data.head())
# Train Test Split 
ratio = (0.3, 0.7)
insample = data.iloc[:int(len(data) * ratio[1])]
outsample = data.iloc[int(len(data) * ratio[1]):]

search_data = insample.iloc[:int(len(insample) * ratio[0])]

# Searching
search_dir = os.path.join(main_dir, 'searching')
search = Searching(data=search_data, TP=(1, 10), SL=(1, 10), dir=search_dir, **config)
search_study = search.run('_short')

# search_study = optuna.load_study(study_name='searching__short', storage=f'sqlite:///searching.db')
best_search_trial = search_study.best_trial.number

# Optimizing
optimize_dir = os.path.join(main_dir, 'optimizing')
optimizer = Optimizer(trial=best_search_trial, path=search_dir, data=insample, dir=optimize_dir, **config)
optimize_study = optimizer.run(name=str(best_search_trial) + '_short')

# optimize_study = optuna.load_study(study_name=f'optimizing_{best_search_trial}_short', storage=f'sqlite:///searching.db')
best_optimize_trial = optimize_study.best_trial.number

# history = pd.read_csv(os.path.join(optimize_dir, str(best_optimize_trial), 'history.csv'))
# plot_price_and_signals(insample, history).show()

# Testing
test_dir = os.path.join(main_dir, 'testing')
tester = Tester(trial_num=best_optimize_trial, path=optimize_dir, data=outsample, dir=test_dir)
tester.run()
from optimize import Searching, Optimizer, Tester
from utils import *
import pandas as pd
import plotly.graph_objects as go

num_trials = int(input('Enter the number of trials: '))

config = {
    'number_of_trials': num_trials,
    'side': 'long',
    'n_jobs': 2
}

main_dir = 'result'
downloader = Downloader()
start_date = '2019-01-01'
end_date = '2024-11-01'

print('Download Data')
data = downloader.get_historical_data(start_date=start_date, end_date=end_date)

# Train Test Split 
ratio = (0.3, 0.7)
search_data = data.iloc[:int(len(data) * ratio[0])]
optimize_data = data.iloc[:int(len(data) * ratio[1])]
test_data = data.iloc[int(len(data) * ratio[1]):]


search_dir = os.path.join(main_dir, 'searching')
search = Searching(data=search_data, TP=(1, 10), SL=(1, 10), dir=search_dir, **config)
search_study = search.run()
best_search_trial = search_study.best_trial.number

optimize_dir = os.path.join(main_dir, 'optimizing')
optimizer = Optimizer(trial=best_search_trial, path=search_dir, data=optimize_data, dir=optimize_dir, **config)
optimize_study = optimizer.run(name=str(best_search_trial))

best_optimize_trial = optimize_study.best_trial.number

history = pd.read_csv(os.path.join(optimize_dir, str(best_optimize_trial), 'history.csv'))

plot_price_and_signals(data, history).show()

test_dir = os.path.join(main_dir, 'testing')
tester = Tester(trial_num=best_optimize_trial, path=optimize_dir, data=test_data, side='long', dir=test_dir)
tester.run()
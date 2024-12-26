from search_ta import Searching, Optimizer, Tester
from utils import *
import pandas as pd
import plotly.graph_objects as go

num_trials = int(input('Enter the number of trials: '))

main_dir = 'search_ta_result'
downloader = Downloader()
start_date = '2023-01-01'
end_date = '2024-11-01'

print('Download Data')
data = downloader.get_historical_data(start_date=start_date, end_date=end_date)

# Train Test Split 
ratio = (0.3, 0.7)
search_data = data.iloc[:int(len(data) * ratio[0])]
optimize_data = data.iloc[:int(len(data) * ratio[1])]
test_data = data.iloc[int(len(data) * ratio[1]):]


search_dir = os.path.join(main_dir, 'searching')
search = Searching(data=search_data, TP=(1, 10), SL=(1, 10), number_of_trials=2, side='long', n_jobs=1, dir=search_dir)
search_study = search.run()

optimize_dir = os.path.join(main_dir, 'optimizing')
optimizer = Optimizer(trial=2, path=search_dir, number_of_trials=1, data=optimize_data, side='long', n_jobs=1, dir=optimize_dir, benchmark=optimize_data)
optimize_study = optimizer.run()

# test_dir = os.path.join(main_dir, 'testing')
# tester = Tester(trial_num=2, path=optimize_dir, data=test_data, side='long', dir=test_dir)

history = pd.read_csv(os.path.join(optimize_dir, '20', 'history.csv'))

print(history.head())
open = history.loc[:, ['date', 'price']]
close = history.loc[:, ['close_time', 'close_price']]

open['date'] = pd.to_datetime(open['date'])
close['close_time'] = pd.to_datetime(close['close_time'])

# Create a DataFrame for candlestick plotting
candlestick_data = optimize_data.resample('15min').agg({
    'price': 'ohlc',
    'volume': 'sum'
})

candlestick_data.columns = candlestick_data.columns.droplevel(0)

# Create the candlestick figure
fig = go.Figure(data=[go.Candlestick(x=candlestick_data.index,
                                     open=candlestick_data['open'],
                                     high=candlestick_data['high'],
                                     low=candlestick_data['low'],
                                     close=candlestick_data['close'])])


fig.add_trace(go.Scatter(x=open['date'], y=open['price'], mode='markers', name='Open Price', marker=dict(symbol='triangle-up', color='green')))
fig.add_trace(go.Scatter(x=close['close_time'], y=close['close_price'], mode='markers', name='Close Price', marker=dict(symbol='triangle-down', color='red')))

# Update layout for better visualization
fig.update_layout(title='Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

# Export the figure to an HTML file
fig.write_html('candlestick_chart.html')

print('Candlestick chart has been saved to candlestick_chart.html')
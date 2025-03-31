import plotly.graph_objects as go
import pandas as pd

def highlight_max_second_max(s):
    is_max = s == s.max()
    is_second_max = s == s.nlargest(2).iloc[-1]  # Get the second largest value

    return ['background-color: green' if m else 'background-color: lightgreen' if sm else '' for m, sm in zip(is_max, is_second_max)]

def highlight_table(df):
    return df.style.apply(highlight_max_second_max, subset=df.columns)

def plotly_candlestick(data, title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price'):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['open'],
                                         high=data['high'],
                                         low=data['low'],
                                         close=data['close'])])

    fig.update_layout(title=title,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title,
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')
    
    return fig


def plot_price_and_signals(data: pd.DataFrame, history: pd.DataFrame) -> None:
    open = history.loc[:, ['date', 'price']]
    close = history.loc[:, ['close_time', 'close_price']]

    open['date'] = pd.to_datetime(open['date'])
    close['close_time'] = pd.to_datetime(close['close_time'])

    # Create a DataFrame for candlestick plotting
    candlestick_data = data.resample('5min').agg({
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
    return fig
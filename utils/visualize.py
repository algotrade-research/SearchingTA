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
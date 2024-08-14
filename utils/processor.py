import pandas as pd
import ta
from abc import ABC, abstractmethod

def processor(df) -> pd.DataFrame:
    data = df.copy()

    # Relative Strength Index
    for i in [5, 14, 30]:
        data[f'rsi_{i}'] = ta.momentum.rsi(data['close'], window=i)

    # Bollinger Bands
    data['upper_band'] = ta.volatility.bollinger_hband(data['close'], window=20)
    data['lower_band'] = ta.volatility.bollinger_lband(data['close'], window=20)
    data['ma20'] = ta.volatility.bollinger_mavg(data['close'], window=20)

    # Moving Average and Exponential Moving Average
    for i in [5, 10, 15, 20, 25, 50]:
        data[f'ma{i}'] = data['close'].rolling(window=i).mean()
        data[f'ema{i}'] = data["close"].ewm(span=i, adjust=False).mean()
    
    # MACD
    data['macd'] = ta.trend.macd(data['close'])
    data['macd_hist'] = ta.trend.macd_diff(data['close'])

    # VWAP
    data['vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'], window=14)

    return data.dropna()
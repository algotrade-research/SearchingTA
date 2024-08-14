import pandas as pd
import numpy as np

# Relative Strength Index
def RSI(df) -> int:
    rsi_5 = df['rsi_5'].iloc[-1]
    rsi_14 = df['rsi_14'].iloc[-1]
    rsi_30 = df['rsi_30'].iloc[-1]

    # Signal
    rsi_5_cross = (rsi_5 > rsi_14).astype(int)
    rsi_14_cross = (rsi_14 > rsi_30).astype(int)
    rsi_cross = (rsi_14 > 50).astype(int)

    if (rsi_5_cross + rsi_14_cross + rsi_cross == 3):
        return 1
    
    rsi_5_cross = (rsi_5 < rsi_14).astype(int)
    rsi_14_cross = (rsi_14 < rsi_30).astype(int)
    rsi_cross = (rsi_14 < 50).astype(int)

    if (rsi_5_cross + rsi_14_cross + rsi_cross == 3):
        return -1

    return 0

# Bollinger Bands
def BBL(df) -> int:
    upper_bands = df['upper_band'].iloc[-1]
    lower_bands = df['lower_band'].iloc[-1]
    close = df['close'].iloc[-1]
    mean = df['ma20'].iloc[-1]

    # Signal
    upper_bands_cross = (close > upper_bands).astype(int)
    lower_bands_cross = (close < lower_bands).astype(int)

    return upper_bands_cross - lower_bands_cross

# MACD
def MACD(df) -> int:
    histogram = df['macd_hist'].iloc[-1]
    prev_histogram = df['macd_hist'].iloc[-2]

    # Signal 
    histogram_cross = (histogram > 0 and prev_histogram < 0).astype(int)
    histogram_cross_down = (histogram < 0 and prev_histogram > 0).astype(int)

    return histogram_cross - histogram_cross_down

def VWAP(df) -> int:
    vwap = df['vwap'].iloc[-1]
    close = df['close'].iloc[-1]

    prev_vwap = df['vwap'].iloc[-2]
    prev_close = df['close'].iloc[-2]

    # Signal
    vwap_cross = (close > vwap and prev_close < prev_vwap).astype(int)
    vwap_cross_down = (close < vwap and prev_close > prev_vwap).astype(int)

    return vwap_cross - vwap_cross_down

def MA5(df):
    ma5= df['ma5'].iloc[-1]
    ma20 = df['ma20'].iloc[-1]

    prev_ma5 = df['ma5'].iloc[-2]
    prev_ma20 = df['ma20'].iloc[-2]

    # Signal
    ma5_cross = (ma5 > ma20).astype(int)
    prev_cross = (prev_ma5 < prev_ma20).astype(int)

    if (ma5_cross + prev_cross == 2):
        return 1
    
    ma5_cross = (ma5 < ma20).astype(int)
    prev_cross = (prev_ma5 > prev_ma20).astype(int)

    if (ma5_cross + prev_cross == 2):
        return -1
    
    return 0

def MA20(df):
    ma20 = df['ma20'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]

    prev_ma20 = df['ma20'].iloc[-2]
    prev_ma50 = df['ma50'].iloc[-2]

    # Signal
    ma20_cross = (ma20 > ma50).astype(int)
    prev_cross = (prev_ma20 < prev_ma50).astype(int)

    if (ma20_cross + prev_cross == 2):
        return 1
    
    ma20_cross = (ma20 < ma50).astype(int)
    prev_cross = (prev_ma20 > prev_ma50).astype(int)

    if (ma20_cross + prev_cross == 2):
        return -1
    
    return 0


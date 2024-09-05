import pandas as pd
import ta
from abc import ABC, abstractmethod

import ta.trend

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
    
    data['ppo'] = ta.momentum.ppo_hist(data['close'])
    
    data["tsi"] = ta.momentum.tsi(data['close'])
    
    data['roc'] = ta.momentum.roc(data['close'])
    
    data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    
    data["adx"] = ta.trend.adx(data['high'], data['low'], data['close'])
    data['di_plus'] = ta.trend.adx_pos(data['high'], data['low'], data['close'])
    data['di_minus'] = ta.trend.adx_neg(data['high'], data['low'], data['close'])
    
    data["cci"] = ta.trend.cci(data['high'], data['low'], data['close'])
    
    data["volume_ma5"] = data['volume'].rolling(window=5).mean()
    data["volume_ma10"] = data['volume'].rolling(window=10).mean()
    
    data["stoch_k"] = ta.momentum.stochrsi_k(data['close'])
    data["stoch_d"] = ta.momentum.stochrsi_d(data['close'])
    
    data["williams_r"] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
    
    data["psar"] = ta.trend.PSARIndicator(data['high'], data['low'], data['close']).psar()
    
    data["obv"] = ta.volume.on_balance_volume(data['close'], data['volume'])
    
    data["donchian_hband"] = ta.volatility.donchian_channel_hband(data["high"], data["low"], data['close'], window=20)
    data["donchian_lband"] = ta.volatility.donchian_channel_lband(data["high"], data["low"], data['close'], window=20)
    
    data["uo"] = ta.momentum.ultimate_oscillator(data['high'], data['low'], data['close'])
    
    data['force_index'] = ta.volume.force_index(data['close'], data['volume'])
    
    data["keltner_hband"] = ta.volatility.keltner_channel_hband(data["high"], data["low"], data['close'], window=20)
    data["keltner_lband"] = ta.volatility.keltner_channel_lband(data["high"], data["low"], data['close'], window=20)
    
    data['vi_plus'] = ta.trend.vortex_indicator_pos(data['high'], data['low'], data['close'])
    data['vi_minus'] = ta.trend.vortex_indicator_neg(data['high'], data['low'], data['close'])
    
    return data.dropna()
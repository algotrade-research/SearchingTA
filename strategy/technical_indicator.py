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

# 1 MACD
def MACD(df) -> int:
    histogram = df['macd_hist'].iloc[-1]
    prev_histogram = df['macd_hist'].iloc[-2]

    # Signal 
    histogram_cross = (histogram > 0 and prev_histogram < 0).astype(int)
    histogram_cross_down = (histogram < 0 and prev_histogram > 0).astype(int)

    return histogram_cross - histogram_cross_down

# 2 VWAP
def VWAP(df) -> int:
    vwap = df['vwap'].iloc[-1]
    close = df['close'].iloc[-1]

    prev_vwap = df['vwap'].iloc[-2]
    prev_close = df['close'].iloc[-2]

    # Signal
    vwap_cross = (close > vwap and prev_close < prev_vwap).astype(int)
    vwap_cross_down = (close < vwap and prev_close > prev_vwap).astype(int)

    return vwap_cross - vwap_cross_down

# 3 MA5
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

# 4 MA20
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

# 5 PPO
def PPO(df) -> int:
    ppo = df['ppo'].iloc[-1]

    prev_ppo = df['ppo'].iloc[-2]

    # Signal
    long = (ppo > 0 and prev_ppo < 0).astype(int)
    short = (ppo < 0 and prev_ppo > 0).astype(int)
    
    return long - short

# 6 ROC
def ROC(df) -> int:
    roc = df['roc'].iloc[-1]
    prev_roc = df['roc'].iloc[-2]

    # Signal
    roc_cross = (roc > 0).astype(int)
    prev_cross = (prev_roc < 0).astype(int)

    if (roc_cross + prev_cross == 2):
        return 1
    
    roc_cross = (roc < 0).astype(int)
    prev_cross = (prev_roc > 0).astype(int)

    if (roc_cross + prev_cross == 2):
        return -1
    
    return 0

# 7 TSI
def TSI(df) -> int:
    tsi = df['tsi'].iloc[-1]
    prev_tsi = df['tsi'].iloc[-2]
    
    # Signal
    tsi_cross = (tsi > 0).astype(int)
    prev_cross = (prev_tsi < 0).astype(int)
    
    if (tsi_cross + prev_cross == 2):
        return 1
    
    tsi_cross = (tsi < 0).astype(int)
    prev_cross = (prev_tsi > 0).astype(int)
    
    if (tsi_cross + prev_cross == 2):
        return -1
    
    return 0

# 8 ATR
def ATR(df) -> int:
    atr = df['atr'].iloc[-1]
    close = df['close'].iloc[-1]

    prev_atr = df['atr'].iloc[-2]
    prev_close = df['close'].iloc[-2]

    # Signal
    atr_cross = (close > atr).astype(int)
    atr_cross_down = (close < atr).astype(int)

    return atr_cross - atr_cross_down

# 9 ADX
def ADX(df) -> int:
    adx = df['adx'].iloc[-1]
    di_plus = df['di_plus'].iloc[-1]
    di_minus = df['di_minus'].iloc[-1]

    prev_adx = df['adx'].iloc[-2]
    prev_di_plus = df['di_plus'].iloc[-2]
    prev_di_minus = df['di_minus'].iloc[-2]

    # Signal
    adx_cross = (adx > 25).astype(int)
    di_plus_cross = (di_plus > di_minus).astype(int)
    di_minus_cross = (di_minus > di_plus).astype(int)

    if (adx_cross + di_plus_cross + di_minus_cross == 3):
        return 1
    
    adx_cross = (adx < 25).astype(int)
    di_plus_cross = (di_plus < di_minus).astype(int)
    di_minus_cross = (di_minus < di_plus).astype(int)

    if (adx_cross + di_plus_cross + di_minus_cross == 3):
        return -1

    return 0

# 10 CCI
def CCI(df) -> int:
    cci = df['cci'].iloc[-1]

    # Signal
    cci_cross = (cci > 100).astype(int)
    cci_cross_down = (cci < -100).astype(int)

    return cci_cross - cci_cross_down

# 11 Momentum
def Momentum(df) -> int:
    
    ma20 = df['ma20'].iloc[-1]
    ma50 = df['ma50'].iloc[-1]
    
    up_macd = df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]
    down_macd = df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2]
    
    trend = (ma20 > ma50).astype(int) - (ma20 < ma50).astype(int)

    signal = MA5(df)
    if trend == 1 and up_macd and signal == 1:
        return 1
    elif trend == -1 and down_macd and signal == -1:
        return -1
    
    return 0

# 12 Volume_MA
def Volume_MA(df) -> int:
    volume_ma5 = df['volume_ma5'].iloc[-1]
    volume_ma10 = df['volume_ma10'].iloc[-1]
    
    prev_volume_ma5 = df['volume_ma5'].iloc[-2]
    prev_volume_ma10 = df['volume_ma10'].iloc[-2]
    
    # Signal
    volume_ma5_cross = (volume_ma5 > volume_ma10 and prev_volume_ma5 < prev_volume_ma10).astype(int)
    volume_ma5_cross_down = (volume_ma5 < volume_ma10 and prev_volume_ma5 > prev_volume_ma10).astype(int)
    
    return volume_ma5_cross - volume_ma5_cross_down

# 13 MomentumBBL
def MomentumBBL(df) -> int:
    
    bl_up = df['upper_band'].iloc[-5:]
    bl_down = df['lower_band'].iloc[-5:]
    
    close = df['close'].iloc[-5:]
    
    bl = (close > bl_up).astype(int) - (close < bl_down).astype(int)
    
    up_macd = df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]
    down_macd = df['macd_hist'].iloc[-1] < df['macd_hist'].iloc[-2]
    
    if bl.sum() > 0 and up_macd:
        return 1
    elif bl.sum() < 0 and down_macd:
        return -1
    
    return 0

# 14 CHOP
def SO(df) -> int:
    k = df['stoch_k'].iloc[-1]
    d = df['stoch_d'].iloc[-1]

    # Signal
    so_cross = (k > d).astype(int)
    so_cross_down = (k < d).astype(int)

    return so_cross - so_cross_down

# 15 Williams R
def W_R(df) -> int:
    wr = df['williams_r'].iloc[-1]

    # Signal
    wr_cross = (wr < -20).astype(int)
    wr_cross_down = (wr > -80).astype(int)

    return wr_cross - wr_cross_down

# 16 PSAR
def PSAR(df) -> int:
    psar = df['psar'].iloc[-1]
    close = df['close'].iloc[-1]

    # Signal
    psar_cross = (close > psar).astype(int)
    psar_cross_down = (close < psar).astype(int)

    return psar_cross - psar_cross_down

# 17 OBV
def OBV(df) -> int:
    obv = df['obv'].iloc[-1]
    prev_obv = df['obv'].iloc[-2]

    # Signal
    obv_cross = (obv > prev_obv).astype(int)
    obv_cross_down = (obv < prev_obv).astype(int)

    return obv_cross - obv_cross_down

# 18 Donchian
def Donchian(df) -> int:
    upper = df['donchian_hband'].iloc[-1]
    lower = df['donchian_lband'].iloc[-1]
    close = df['close'].iloc[-1]

    # Signal
    donchian_cross = (close > upper).astype(int)
    donchian_cross_down = (close < lower).astype(int)

    return donchian_cross - donchian_cross_down

# 19 Keltner
def Keltner(df) -> int:
    upper = df['keltner_hband'].iloc[-1]
    lower = df['keltner_lband'].iloc[-1]
    close = df['close'].iloc[-1]

    # Signal
    keltner_cross = (close > upper).astype(int)
    keltner_cross_down = (close < lower).astype(int)

    return keltner_cross - keltner_cross_down

# 22 UO
def UO(df) -> int:
    uo = df['uo'].iloc[-1]

    # Signal
    uo_cross = (uo > 50).astype(int)
    uo_cross_down = (uo < 50).astype(int)

    return uo_cross - uo_cross_down

# 23 Force Index
def FI(df) -> int:
    fi = df['force_index'].iloc[-1]
    prev_fi = df['force_index'].iloc[-2]

    # Signal
    fi_cross = (fi > 0 and prev_fi < 0).astype(int)
    fi_cross_down = (fi < 0 and prev_fi > 0).astype(int)

    return fi_cross - fi_cross_down

# 24 Vortex
def Vortex(df) -> int:
    vi_plus = df['vi_plus'].iloc[-1]
    vi_minus = df['vi_minus'].iloc[-1]

    # Signal
    vortex_cross = (vi_plus > vi_minus).astype(int)
    vortex_cross_down = (vi_plus < vi_minus).astype(int)

    return vortex_cross - vortex_cross_down




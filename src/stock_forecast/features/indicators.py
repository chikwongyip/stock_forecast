"""纯 pandas 自实现技术指标 (~20 个).

全部为简单 rolling/ewm 公式, 不依赖 TA-Lib/pandas-ta.
所有指标在第 T 行的含义是"用截至第 T 日(含)的行情算出的指标值"——
防前视的对齐(shift)由 build.py 统一执行, 本模块不管时间平移.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, n: int) -> pd.Series:
    """经典 Wilder RSI: 前 n 个变化量用简单均值作种子, 之后按
    avg = (avg*(n-1) + x)/n 递推. ewm(alpha=1/n) 与之不等价(种子不同), 故显式递推.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0).to_numpy(dtype=float)
    down = (-delta).clip(lower=0.0).to_numpy(dtype=float)
    m = len(delta)
    out = np.full(m, np.nan)
    if m <= n:
        return pd.Series(out, index=close.index)
    # up[0]=down[0]=NaN, 有效变化量从下标 1 开始; 种子用第 1..n 个变化量
    ag = np.nanmean(up[1 : n + 1])
    ad = np.nanmean(down[1 : n + 1])
    out[n] = _rsi_value(ag, ad)
    for i in range(n + 1, m):
        ag = (ag * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + down[i]) / n
        out[i] = _rsi_value(ag, ad)
    return pd.Series(out, index=close.index)


def _rsi_value(avg_up: float, avg_down: float) -> float:
    if avg_down == 0:
        return 100.0 if avg_up > 0 else 50.0
    return 100 - 100 / (1 + avg_up / avg_down)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9):
    lowest = low.rolling(n, min_periods=1).min()
    highest = high.rolling(n, min_periods=1).max()
    rsv = (close - lowest) / (highest - lowest).replace(0.0, np.nan) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def boll(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0.0).apply(np.sign)
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        n: int = 14) -> pd.Series:
    """资金流量指标 (0~100)."""
    typical = (high + low + close) / 3
    raw_money = typical * volume
    delta = typical.diff().fillna(0.0)
    pos = raw_money.where(delta > 0, 0.0)
    neg = raw_money.where(delta < 0, 0.0)
    pos_sum = pos.rolling(n).sum()
    neg_sum = neg.rolling(n).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(100.0).clip(0, 100)


def compute_all(kline: pd.DataFrame) -> pd.DataFrame:
    """输入英文列名 K线(date/open/close/high/low/volume/amount...), 输出指标特征.

    输出与输入按 date 索引对齐; 第 T 行含 T 日行情信息(防前视 shift 在 build.py).
    """
    c, h, lo = kline["close"], kline["high"], kline["low"]
    v = kline["volume"].astype(float)
    o = kline["open"]
    f = pd.DataFrame(index=kline.index)

    # 趋势
    for n in (5, 10, 20, 60):
        f[f"ma{n}"] = sma(c, n)
    f["ema12"], f["ema26"] = ema(c, 12), ema(c, 26)
    f["macd"], f["macd_signal"], f["macd_hist"] = macd(c)
    # 价格相对均线的位置(无量纲, 跨股票可比)
    for n in (5, 10, 20, 60):
        f[f"close_ma{n}_ratio"] = c / f[f"ma{n}"] - 1

    # 动量
    f["rsi6"], f["rsi14"], f["rsi24"] = rsi(c, 6), rsi(c, 14), rsi(c, 24)
    f["kdj_k"], f["kdj_d"], f["kdj_j"] = kdj(h, lo, c)
    f["roc10"] = c.pct_change(10)

    # 波动
    upper, mid, lower = boll(c)
    f["boll_width"] = (upper - lower) / mid
    f["boll_position"] = ((c - lower) / (upper - lower)).clip(-0.5, 1.5)
    f["atr_ratio"] = atr(h, lo, c) / c

    # 量价
    f["obv_slope5"] = obv(c, v).diff(5)
    f["volume_ratio"] = v / sma(v, 20)
    f["mfi14"] = mfi(h, lo, c, v)

    # 收益率与波动率
    for n in (1, 3, 5, 10, 20):
        f[f"ret{n}"] = c.pct_change(n)
    f["volatility5"] = c.pct_change().rolling(5).std()
    f["volatility20"] = c.pct_change().rolling(20).std()

    # K线形态(相对值)
    f["amplitude"] = (h - lo) / c
    f["close_open_ratio"] = (c - o) / o
    f["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / c
    f["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - lo) / c

    # 隔日跳空
    f["gap"] = o / c.shift(1) - 1

    return f

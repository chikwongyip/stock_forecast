"""指标数值正确性: 手算固定小序列 + 独立朴素递推实现交叉对照."""

import numpy as np
import pandas as pd

from stock_forecast.features import indicators as ind

CLOSE = pd.Series([10, 11, 12, 11, 12, 13, 14, 13, 12, 13], dtype=float)
HIGH = CLOSE + 0.5
LOW = CLOSE - 0.5
VOL = pd.Series([100, 110, 120, 105, 115, 130, 140, 125, 110, 135], dtype=float)
OPEN = pd.Series([10, 10.5, 11.5, 12, 11, 12, 13, 14, 13, 12.5], dtype=float)


def test_sma_hand_computed():
    s = ind.sma(CLOSE, 3)
    assert np.isclose(s.iloc[2], (10 + 11 + 12) / 3)
    assert np.isclose(s.iloc[4], (12 + 11 + 12) / 3)
    assert pd.isna(s.iloc[1])


def test_ema_hand_computed():
    s = ind.ema(CLOSE, 3)  # span=3 -> alpha = 2/(3+1) = 0.5, adjust=False
    a = 2 / (3 + 1)
    e0 = 10.0
    e1 = e0 + a * (11 - e0)
    e2 = e1 + a * (12 - e1)
    assert np.isclose(s.iloc[0], e0)
    assert np.isclose(s.iloc[1], e1)
    assert np.isclose(s.iloc[2], e2)


def _naive_rsi(close: list[float], n: int) -> list[float]:
    """经典 Wilder 递推的朴素逐日实现, 与向量化实现相互独立."""
    out = [np.nan] * len(close)
    gains = [max(close[i] - close[i - 1], 0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0) for i in range(1, len(close))]
    if len(gains) < n:
        return out
    ag, ad = sum(gains[:n]) / n, sum(losses[:n]) / n
    for i in range(n, len(gains) + 1):
        if i > n:
            ag = (ag * (n - 1) + gains[i - 1]) / n
            ad = (ad * (n - 1) + losses[i - 1]) / n
        if ad == 0:
            out[i] = 100.0 if ag > 0 else 50.0
        else:
            out[i] = 100 - 100 / (1 + ag / ad)
    return out


def test_rsi_matches_wilder_recursion():
    n = 3
    got = ind.rsi(CLOSE, n).tolist()
    want = _naive_rsi(CLOSE.tolist(), n)
    for g, w in zip(got, want):
        assert np.isclose(g, w, atol=1e-9, equal_nan=True), (g, w)


def test_rsi_monotonic_up_is_100():
    up = pd.Series(np.linspace(10, 20, 15))
    vals = ind.rsi(up, 14).iloc[14:]
    assert vals.notna().all()
    assert (vals > 99).all()


def test_rsi_flat_is_50():
    flat = pd.Series([5.0] * 20)
    vals = ind.rsi(flat, 14).iloc[14:]
    assert vals.notna().all()
    assert np.allclose(vals, 50.0)


def test_macd_constant_series_is_zero():
    flat = pd.Series([5.0] * 40)
    m, sig, hist = ind.macd(flat)
    assert np.allclose(m.iloc[25:], 0, atol=1e-12)
    assert np.allclose(hist.iloc[25:], 0, atol=1e-12)


def test_boll_symmetry_and_position():
    upper, mid, lower = ind.boll(CLOSE, 5)
    assert np.allclose(upper - mid, mid - lower, equal_nan=True)  # 暖机期 NaN 对齐
    pos = (CLOSE - lower) / (upper - lower)
    assert (pos.dropna().between(-0.1, 1.1)).all()


def test_kdj_identity():
    k, d, j = ind.kdj(HIGH, LOW, CLOSE)
    assert np.allclose(j, 3 * k - 2 * d, equal_nan=True)


def test_obv_hand_computed():
    # 涨日加成交量, 跌日减, 平日不动
    got = ind.obv(CLOSE, VOL).tolist()
    want = [0.0]
    for i in range(1, len(CLOSE)):
        if CLOSE.iloc[i] > CLOSE.iloc[i - 1]:
            want.append(want[-1] + VOL.iloc[i])
        elif CLOSE.iloc[i] < CLOSE.iloc[i - 1]:
            want.append(want[-1] - VOL.iloc[i])
        else:
            want.append(want[-1])
    assert np.allclose(got, want)


def test_atr_positive():
    assert (ind.atr(HIGH, LOW, CLOSE, 5).iloc[1:] > 0).all()


def test_mfi_in_range():
    m = ind.mfi(HIGH, LOW, CLOSE, VOL, 5)
    assert m.dropna().between(0, 100).all()


def test_compute_all_shapes_and_no_inf():
    kline = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(CLOSE)),
        "open": OPEN, "close": CLOSE, "high": HIGH, "low": LOW,
        "volume": VOL,
    })
    f = ind.compute_all(kline)
    assert f.shape[0] == len(CLOSE)
    assert f.shape[1] > 30
    assert not np.isinf(f.to_numpy()).any()

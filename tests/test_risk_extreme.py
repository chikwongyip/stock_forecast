"""极端场景扩充(Phase 5 / 5.3)单测: 连续一字板/振幅异常/量能枯竭. 全离线合成."""

import numpy as np
import pandas as pd

from stock_forecast.config import get_settings
from stock_forecast.risk import extreme

S = get_settings()


def _frame(n, close=10.0, high=None, low=None, vol=1e6):
    """默认帧: 每日有 2% 正常振幅(high!=low), 等量 -> 不触发任何极端场景."""
    c = np.full(n, float(close))
    h = c * 1.02 if high is None else np.asarray(high, float)
    lo = c * 0.98 if low is None else np.asarray(low, float)
    v = np.full(n, float(vol)) if np.isscalar(vol) else np.asarray(vol, float)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": c, "close": c, "high": h, "low": lo, "volume": v,
    })


def test_normal_frame_no_notes():
    assert extreme.detect_extreme_scenarios(_frame(61), S) == []


def test_consecutive_yizi_streak():
    n = 10
    high = [10.5] * 7 + [10.0] * 3   # 最后 3 日一字板
    low = [9.5] * 7 + [10.0] * 3
    notes = extreme.detect_extreme_scenarios(_frame(n, high=high, low=low), S)
    assert any("连续 3 日一字板" in x for x in notes)


def test_single_yizi_no_streak_note():
    """只有最后 1 日一字板 -> 不触发'连续'(阈值 2)."""
    n = 10
    high = [10.5] * 9 + [10.0]
    low = [9.5] * 9 + [10.0]
    notes = extreme.detect_extreme_scenarios(_frame(n, high=high, low=low), S)
    assert not any("连续" in x and "一字板" in x for x in notes)


def test_amplitude_abnormal():
    n = 20
    high = np.linspace(10, 20, n)   # 区间最高 20
    low = np.linspace(5, 8, n)      # 区间最低 5
    close = np.linspace(8, 12, n)   # 中位 ~10 -> 振幅 ~1.5 >= 0.4
    f = pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": close, "close": close, "high": high, "low": low,
        "volume": np.full(n, 1e6),
    })
    notes = extreme.detect_extreme_scenarios(f, S)
    assert any("振幅" in x for x in notes)


def test_amplitude_normal_no_note():
    n = 20
    close = np.full(n, 10.0)
    f = pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": close, "close": close, "high": close * 1.02, "low": close * 0.98,
        "volume": np.full(n, 1e6),
    })
    assert not any("振幅" in x for x in extreme.detect_extreme_scenarios(f, S))


def test_volume_dry():
    n = 61
    vol = np.full(n, 1e6)
    vol[-1] = 1e3   # 最后一日量能骤降到底部
    notes = extreme.detect_extreme_scenarios(_frame(n, vol=vol), S)
    assert any("量能枯竭" in x for x in notes)


def test_volume_constant_not_dry():
    """等量序列: 分位=0 但量并不低于中位, 中位数守卫避免误报."""
    assert not any("量能枯竭" in x for x in extreme.detect_extreme_scenarios(_frame(61), S))


def test_insufficient_data_no_crash():
    assert extreme.detect_extreme_scenarios(_frame(5), S) == []   # 窗口不足, 静默
    assert extreme.detect_extreme_scenarios(None, S) == []
    assert extreme.detect_extreme_scenarios(pd.DataFrame(), S) == []

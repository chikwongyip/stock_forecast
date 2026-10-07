"""预测分布区间(Phase 6 / 6.1)单测: 三点混合分位数 + 防前视硬断言. 全离线合成."""

import numpy as np
import pandas as pd

from stock_forecast.config import get_settings
from stock_forecast.output import scenarios as sc

S = get_settings()
PROBS = {"p_up": 0.4, "p_flat": 0.3, "p_down": 0.3}


def _kline(n=150, seed=7, start=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=n)
    close = start * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    return pd.DataFrame({"date": dates, "close": close})


def test_basic_structure():
    out = sc.scenario_quantiles(PROBS, _kline(), S.label_threshold, S.scenario_lookback)
    assert out is not None
    for key in ("q25", "q50", "q75", "q25_scenario", "q50_scenario", "q75_scenario",
                "scenarios", "lookback_days", "note"):
        assert key in out
    assert out["q25"] <= out["q50"] <= out["q75"]
    assert [s["name"] for s in out["scenarios"]] == ["悲观", "中性", "乐观"]
    assert np.isclose(sum(s["prob"] for s in out["scenarios"]), 1.0)


def test_quantile_follows_probability_weighting():
    """强偏涨时 p50 落在乐观原子; 强偏跌时落在悲观原子."""
    k = _kline()
    up = sc.scenario_quantiles({"p_up": 0.8, "p_flat": 0.1, "p_down": 0.1}, k, S.label_threshold)
    down = sc.scenario_quantiles({"p_up": 0.1, "p_flat": 0.1, "p_down": 0.8}, k, S.label_threshold)
    assert up["q50_scenario"] == "乐观"
    assert down["q50_scenario"] == "悲观"


def test_no_lookahead_last_row():
    """硬断言: 最后一行收益 = 本次预测的标签, 篡改最后一行 close 输出必须一字不变."""
    base = sc.scenario_quantiles(PROBS, _kline(), S.label_threshold)
    k2 = _kline()
    k2.loc[k2.index[-1], "close"] *= 1.5  # 篡改最后一日收盘(标签本身)
    tampered = sc.scenario_quantiles(PROBS, k2, S.label_threshold)
    assert base == tampered


def test_insufficient_data_returns_none():
    assert sc.scenario_quantiles(PROBS, _kline(n=20), S.label_threshold) is None
    assert sc.scenario_quantiles(PROBS, None, S.label_threshold) is None
    assert sc.scenario_quantiles(PROBS, pd.DataFrame(), S.label_threshold) is None


def test_format_scenario_lines():
    out = sc.scenario_quantiles(PROBS, _kline(), S.label_threshold)
    lines = sc.format_scenario_lines(out)
    assert len(lines) == 3
    assert "p25" in lines[0] and "p50" in lines[0] and "p75" in lines[0]
    assert sc.format_scenario_lines(None) == []

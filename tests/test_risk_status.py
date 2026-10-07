"""交易状态识别(Phase 5 / 5.1)单测: 停牌/涨跌停/ST → 适用性降级. 全离线合成."""

import numpy as np
import pandas as pd

from stock_forecast.config import get_settings
from stock_forecast.risk import status

S = get_settings()


def _kline(closes, start="2025-01-06", pct=None, high=None, low=None, vol=None):
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "open": c, "close": c,
        "high": np.asarray(high, dtype=float) if high is not None else c,
        "low": np.asarray(low, dtype=float) if low is not None else c,
        "volume": np.asarray(vol, dtype=float) if vol is not None else np.full(n, 1e6),
    })
    if pct is not None:
        df["pct_change"] = pct
    return df


def _index(n, start="2025-01-06"):
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n), "close": np.full(n, 3000.0),
    })


# ---------- price_limit_for: 板别限幅 ----------

def test_price_limit_by_board():
    assert status.price_limit_for("600519") == 0.10          # 沪主板
    assert status.price_limit_for("000001") == 0.10          # 深主板
    assert status.price_limit_for("300750") == 0.20          # 创业板
    assert status.price_limit_for("301001") == 0.20          # 创业板(301)
    assert status.price_limit_for("688111") == 0.20          # 科创板
    assert status.price_limit_for("830799") == 0.30          # 北交所(8 开头)
    assert status.price_limit_for("600519", is_st=True) == 0.05   # 主板 ST
    assert status.price_limit_for("300750", is_st=True) == 0.20   # 创业板 ST 仍 20%


# ---------- detect_st ----------

def test_detect_st():
    assert status.detect_st("*ST海润") is True
    assert status.detect_st("ST康美") is True
    assert status.detect_st("贵州茅台") is False
    assert status.detect_st(None) is False
    assert status.detect_st("") is False


# ---------- detect_limit_move ----------

def test_limit_up_main_board():
    k = _kline([10.0, 11.0], pct=[0.0, 10.0])
    assert status.detect_limit_move(k, "600519", False, S) == "涨停"


def test_limit_down_main_board():
    k = _kline([10.0, 9.0], pct=[0.0, -10.0])
    assert status.detect_limit_move(k, "600519", False, S) == "跌停"


def test_limit_up_star_board_20pct():
    k = _kline([10.0, 12.0], pct=[0.0, 20.0])
    assert status.detect_limit_move(k, "300750", False, S) == "涨停"


def test_10pct_not_limit_on_star_board():
    k = _kline([10.0, 11.0], pct=[0.0, 10.0])  # 创业板 +10% 非涨停(限幅 20%)
    assert status.detect_limit_move(k, "300750", False, S) is None


def test_limit_move_fallback_close_ratio():
    """无 pct_change 列 -> 用 close-to-close 收益判定."""
    k = _kline([10.0, 11.0])
    assert status.detect_limit_move(k, "600519", False, S) == "涨停"


def test_limit_move_st_5pct():
    k = _kline([10.0, 10.5], pct=[0.0, 5.0])  # ST 主板 +5% 涨停
    assert status.detect_limit_move(k, "600519", True, S) == "涨停"


def test_limit_move_insufficient_data():
    assert status.detect_limit_move(_kline([10.0]), "600519", False, S) is None
    assert status.detect_limit_move(None, "600519", False, S) is None


def test_limit_move_normal_day_none():
    k = _kline([10.0, 10.2, 10.3], pct=[0.0, 2.0, 0.98])
    assert status.detect_limit_move(k, "600519", False, S) is None


# ---------- detect_suspension ----------

def test_suspension_current():
    """个股最新日落后于大盘 -> 当前停牌."""
    k = _kline([10.0] * 5)   # 01-06 .. 01-10
    idx = _index(6)          # 01-06 .. 01-13
    current, recent, gap, sl, il = status.detect_suspension(k, idx, S)
    assert current is True and recent is False


def test_suspension_recent_gap():
    """有当日行情但窗口内中间缺 2 天 -> 近期停牌(复牌)."""
    dates = pd.bdate_range("2025-01-06", periods=10)
    idx = pd.DataFrame({"date": dates, "close": np.full(10, 3000.0)})
    keep = [0, 1, 2, 3, 4, 7, 8, 9]  # 缺 index 5,6
    k = pd.DataFrame({
        "date": dates[keep], "open": np.full(8, 10.0), "close": np.full(8, 10.0),
        "high": np.full(8, 10.0), "low": np.full(8, 10.0), "volume": np.full(8, 1e6),
    })
    current, recent, gap, sl, il = status.detect_suspension(k, idx, S)
    assert current is False and recent is True and gap >= 2


def test_suspension_index_missing_graceful():
    k = _kline([10.0] * 5)
    current, recent, gap, sl, il = status.detect_suspension(k, None, S)
    assert current is False and recent is False and il is None


def test_no_suspension_aligned():
    k = _kline([10.0] * 6)
    idx = _index(6)
    current, recent, gap, sl, il = status.detect_suspension(k, idx, S)
    assert current is False and recent is False


# ---------- assess_trading_status ----------

def test_assess_normal():
    st = status.assess_trading_status("600519", _kline([10.0, 10.1, 10.2]),
                                      _index(3), "贵州茅台", S)
    assert st["applicable"] is True and st["reference_only"] is False
    assert st["label"] == "正常" and st["warnings"] == []


def test_assess_suspended_not_applicable():
    st = status.assess_trading_status("600519", _kline([10.0] * 5), _index(6),
                                      "贵州茅台", S)
    assert st["applicable"] is False and st["label"] == "停牌"
    assert any("停牌" in w for w in st["warnings"])


def test_assess_st_reference_only():
    st = status.assess_trading_status("600519", _kline([10.0, 10.1, 10.2]), _index(3),
                                      "*ST测试", S)
    assert st["applicable"] is True and st["reference_only"] is True
    assert st["label"] == "ST"


def test_assess_limit_up_reference_only():
    k = _kline([10.0, 11.0], pct=[0.0, 10.0])
    st = status.assess_trading_status("600519", k, _index(2), "某公司", S)
    assert st["reference_only"] is True and st["label"] == "涨停"


def test_assess_severity_st_over_limit():
    """ST + 涨停同时命中: 主标签取严重度更高的涨停(severity 1 < ST 3)."""
    k = _kline([10.0, 10.5], pct=[0.0, 5.0])  # ST 主板 5% 涨停
    st = status.assess_trading_status("600519", k, _index(2), "*ST测试", S)
    assert set(st["statuses"]) >= {"ST", "涨停"}
    assert st["label"] == "涨停"


def test_assess_name_missing_skips_st():
    st = status.assess_trading_status("600519", _kline([10.0, 10.1, 10.2]), _index(3),
                                      None, S)
    assert st["applicable"] is True and st["reference_only"] is False
    assert st["label"] == "正常"

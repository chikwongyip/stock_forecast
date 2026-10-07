"""Phase 2 扩展源特征函数专项单测: 手算对照 + 时间语义边界, 全部离线合成."""

import numpy as np
import pandas as pd

from stock_forecast.features import sources as src

N = 30
KD = pd.bdate_range("2024-01-01", periods=N)


def _kline(dates=KD, closes=None):
    c = np.asarray(closes if closes is not None else np.linspace(10, 12, len(dates)), float)
    return pd.DataFrame({"date": dates, "close": c})


# ---------------------------------------------------------------------------
# 事件衰减(龙虎榜/增减持/解禁共用)
# ---------------------------------------------------------------------------

def test_event_decay_hand_computed():
    """单事件: 第 T 行 = value * 0.5^(days/halflife), 严格后顾."""
    ev_date = KD[5]
    events = pd.DataFrame({"date": [ev_date], "value": [100.0]})
    f = src.event_decay_features(events, _kline(), "value", "ev",
                                 halflife=14.0, window=60.0)
    # 事件日之前: 0(无事件), 不是 NaN(源可用)
    assert (f["ev_decay"].iloc[:5] == 0).all()
    assert (f["ev_count"].iloc[:5] == 0).all()
    # 事件当日: 权重 0.5^0 = 1
    assert np.isclose(f["ev_decay"].iloc[5], 100.0)
    assert f["ev_count"].iloc[5] == 1
    # 5 个日历日后: 0.5^(5/14)
    d5 = (KD[10] - ev_date).days
    assert np.isclose(f["ev_decay"].iloc[10], 100.0 * 0.5 ** (d5 / 14))
    # 未来事件绝不影响之前行(防前视的核心): 事件在第 20 行, 第 19 行必须仍为 0
    ev2 = pd.DataFrame({"date": [KD[20]], "value": [50.0]})
    f2 = src.event_decay_features(ev2, _kline(), "value", "ev")
    assert (f2["ev_decay"].iloc[:20] == 0).all()


def test_event_decay_window_and_aggregation():
    """窗口外事件不计; 同日多事件求和."""
    events = pd.DataFrame({
        "date": [KD[3], KD[3], KD[0] - pd.Timedelta(days=200)],
        "value": [10.0, 5.0, 999.0],
    })
    f = src.event_decay_features(events, _kline(), "value", "ev", window=60.0)
    assert np.isclose(f["ev_decay"].iloc[3], 15.0)  # 同日聚合, 权重 1
    assert f["ev_count"].iloc[3] == 2
    # 200 天前的事件在窗口外: 第 25 行仍只见 KD[3] 的残余衰减
    days = (KD[25] - KD[3]).days
    assert np.isclose(f["ev_decay"].iloc[25], 15.0 * 0.5 ** (days / 14.0))


def test_event_decay_none_vs_empty():
    k = _kline()
    f_none = src.event_decay_features(None, k, "value", "ev")
    assert f_none["ev_decay"].isna().all() and f_none["ev_count"].isna().all()
    f_empty = src.event_decay_features(pd.DataFrame(columns=["date", "value"]), k, "value", "ev")
    assert (f_empty["ev_decay"] == 0).all() and (f_empty["ev_count"] == 0).all()
    assert list(f_none.columns) == list(f_empty.columns) == ["ev_decay", "ev_count"]


# ---------------------------------------------------------------------------
# 宏观: 公布滞后是防前视的关键边界
# ---------------------------------------------------------------------------

def test_macro_publish_lag():
    """M 月值必须自 M+1 月 15 日(含)起才可见, 之前一律 NaN."""
    macro = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        "pmi": [50.1, 50.4],
        "cpi_yoy": [0.2, 0.3],
        "m2_yoy": [7.1, 7.0],
    })
    dates = pd.bdate_range("2024-01-02", "2024-04-30")
    f = src.macro_features(macro, _kline(dates=dates))
    eff_jan = pd.Timestamp("2024-02-15")  # 1月值生效日 = 2/15
    eff_feb = pd.Timestamp("2024-03-15")
    before = f[dates < eff_jan]
    assert before["macro_pmi"].isna().all()
    mid = f[(dates >= eff_jan) & (dates < eff_feb)]
    assert np.allclose(mid["macro_pmi"], 50.1)
    after = f[dates >= eff_feb]
    assert np.allclose(after["macro_pmi"], 50.4)
    # PMI 环比: 2 月生效起 = 50.4 - 50.1
    assert np.allclose(after["macro_pmi_mom"].dropna(), 0.3, atol=1e-9)


def test_macro_none_and_missing_cols():
    f = src.macro_features(None, _kline())
    assert list(f.columns) == src.MACRO_COLS and f.isna().all().all()
    # 只有 pmi 列时其余列仍必须存在(全 NaN), schema 恒定
    m = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "pmi": [50.0]})
    f2 = src.macro_features(m, _kline())
    assert list(f2.columns) == src.MACRO_COLS
    assert f2["macro_cpi_yoy"].isna().all()


# ---------------------------------------------------------------------------
# 外盘: 日历缺口只向后填充
# ---------------------------------------------------------------------------

def test_global_ffill_and_overnight_ret():
    dates = KD[:10]
    # 挖掉第 3 行模拟美股假日
    glob = pd.DataFrame({
        "date": dates.delete(3),
        "nasdaq": [100, 101, 102, 103, 104, 105, 106, 107, 108],
        "spx": [50.0] * 9,
        "vix": [15.0] * 9,
    })
    f = src.global_features(glob, _kline(dates=dates))
    # 第 3 行(ffill 自第 2 行 102): ret = 102/102 - 1? 第 2 行值 102, 第 3 行填充 102 -> ret 0
    assert np.isclose(f["us_nasdaq_ret1"].iloc[2], 102 / 101 - 1)
    assert np.isclose(f["us_nasdaq_ret1"].iloc[3], 0.0)  # ffill 日
    assert np.isclose(f["us_nasdaq_ret1"].iloc[4], 103 / 102 - 1)
    assert (f["us_spx_ret1"].iloc[1:] == 0).all()
    assert np.allclose(f["vix_level"].dropna(), 15.0)


def test_global_none():
    f = src.global_features(None, _kline())
    assert list(f.columns) == src.GLOBAL_COLS and f.isna().all().all()


# ---------------------------------------------------------------------------
# 行业 / 北向 / 两融
# ---------------------------------------------------------------------------

def test_industry_excess_ret():
    dates = KD[:6]
    ind = pd.DataFrame({"date": dates, "close": [1000, 1010, 1005, 1020, 1030, 1025]})
    k = _kline(dates=dates, closes=[10, 10.2, 10.1, 10.3, 10.4, 10.35])
    f = src.industry_features(ind, k)
    sec_r1 = 1010 / 1000 - 1
    stk_r1 = 10.2 / 10 - 1
    assert np.isclose(f["sec_ret1"].iloc[1], sec_r1)
    assert np.isclose(f["excess_ret1"].iloc[1], stk_r1 - sec_r1)
    assert np.isclose(f["sec_ret5"].iloc[5], 1025 / 1000 - 1)


def test_industry_none():
    f = src.industry_features(None, _kline())
    assert list(f.columns) == src.INDUSTRY_COLS and f.isna().all().all()


def test_north_sums_and_divergence():
    dates = KD[:6]
    north = pd.DataFrame({"date": dates, "value": [10.0, -5.0, 20.0, 1.0, 2.0, 3.0]})
    index = pd.DataFrame({"date": dates, "close": [3000, 3010, 3005, 3000, 2990, 2980]})
    f = src.north_features(north, _kline(dates=dates), index)
    assert np.isclose(f["north_net1"].iloc[2], 20.0)
    assert np.isclose(f["north_net3"].iloc[1], 5.0)   # min_periods=2
    assert np.isclose(f["north_net3"].iloc[2], 25.0)
    assert pd.isna(f["north_net5"].iloc[1])            # min_periods=3
    assert np.isclose(f["north_net5"].iloc[4], 10 - 5 + 20 + 1 + 2)
    # 第 5 行: 北向 5 日累计 > 0, 指数 5 日收益 2980/3000-1 < 0 -> 背离 = -1
    assert f["north_div5"].iloc[5] == -1


def test_margin_chg():
    dates = KD[:8]
    bal = [100, 102, 101, 103, 104, 105, 106, 107]
    margin = pd.DataFrame({"date": dates, "margin_balance": [float(b) for b in bal]})
    f = src.margin_features(margin, _kline(dates=dates))
    assert np.isclose(f["margin_chg5"].iloc[5], 105 / 100 - 1)
    assert pd.isna(f["margin_chg5"].iloc[4])
    assert pd.isna(f["margin_chg20"].iloc[7])


# ---------------------------------------------------------------------------
# 通用: 输出长度与 kline 对齐
# ---------------------------------------------------------------------------

def test_all_outputs_aligned_to_kline():
    k = _kline()
    for f in (
        src.industry_features(None, k),
        src.north_features(None, k),
        src.margin_features(None, k),
        src.event_decay_features(None, k, "value", "ev"),
        src.global_features(None, k),
        src.macro_features(None, k),
    ):
        assert len(f) == N
        assert list(f.index) == list(range(N))

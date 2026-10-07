"""防前视偏差硬断言 + 标签定义正确性.

最核心的测试: 修改第 T 日及之后的任何原始数据, X[T] 特征行必须一字不变.
"""

import numpy as np
import pandas as pd

from stock_forecast.features import labeling, sources
from stock_forecast.features.build import EXTRA_KEYS, build_feature_matrix

THRESHOLD = 0.005
N_DAYS = 120

# 事件衰减类源的输出列(前缀拼接口径, 与 sources.event_decay_features 一致)
EVENT_COLS = ["lhb_decay", "lhb_count", "holder_decay", "holder_count",
              "unlock_decay", "unlock_count"]


def _make_synthetic(n: int = N_DAYS, seed: int = 7):
    """构造确定性的合成 K线/指数/资金流 + Phase 2 扩展源, 离线可复现."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    kline = pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })
    idx_close = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    index = pd.DataFrame({
        "date": dates,
        "open": idx_close, "close": idx_close,
        "high": idx_close * 1.005, "low": idx_close * 0.995,
        "volume": np.full(n, 1e8),
    })
    fund_flow = pd.DataFrame({
        "date": dates[-60:],  # 资金流只覆盖近期 60 天
        "main_net_pct": rng.normal(0, 3, 60),
        "super_net_pct": rng.normal(0, 2, 60),
        "big_net_pct": rng.normal(0, 2, 60),
    })

    # ---- Phase 2 扩展源(合成): 覆盖日频/事件/月频/含未来行/含日历缺口各形态 ----
    extras = {
        "industry": pd.DataFrame({
            "date": dates,
            "close": 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        }),
        "north": pd.DataFrame({"date": dates, "value": rng.normal(0, 30, n)}),
        "margin": pd.DataFrame({
            "date": dates,
            "margin_balance": 15000 + np.cumsum(rng.normal(0, 20, n)),
        }),
        "lhb": pd.DataFrame({
            "date": dates[[5, 20, 50, 90, 110]],
            "net_buy": rng.normal(0, 5e7, 5),
        }),
        "holder": pd.DataFrame({
            "date": dates[[3, 40, 100]], "amount": [5e6, -3e6, 2e6],
        }),
        # 解禁日历含"未来行"(公开信息, 但特征只用后顾事件 -> 不得影响未来行之前)
        "unlock": pd.DataFrame({
            "date": [dates[30], dates[100], dates[-1] + pd.Timedelta(days=45)],
            "ratio": [0.5, 1.2, 0.8],
        }),
        # 外盘: 挖掉几天模拟美股假日缺口(ffill 只向后填充)
        "global": pd.DataFrame({
            "date": dates.delete([3, 25, 60]),
            "nasdaq": 15000 * np.exp(np.cumsum(rng.normal(0, 0.01, n - 3))),
            "spx": 5000 * np.exp(np.cumsum(rng.normal(0, 0.008, n - 3))),
            "vix": 15 + np.abs(rng.normal(0, 2, n - 3)),
        }),
        "macro": pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "pmi": 49 + rng.normal(0, 0.5, 12),
            "cpi_yoy": rng.normal(0, 0.3, 12),
            "m2_yoy": 7 + rng.normal(0, 0.2, 12),
        }),
    }
    return kline, index, fund_flow, extras


def test_label_thresholds():
    closes = pd.Series([100.0, 101.0, 101.4, 101.6, 101.0])
    y = labeling.make_labels(closes, 0.005)
    # 变化: +1.0% -> 涨; +0.396% -> 平; +0.197% -> 平; -0.59% -> 跌
    assert pd.isna(y.iloc[0])  # 首行无前收 -> NaN (NaN 不能用 == 比较)
    assert y.iloc[1:].tolist() == [1, 0, 0, -1]


def test_label_boundary_exclusive():
    assert labeling.pct_to_label(0.005, 0.005) == 0  # 恰好等于阈值 -> 平
    assert labeling.pct_to_label(0.0051, 0.005) == 1
    assert labeling.pct_to_label(-0.0051, 0.005) == -1


def test_no_lookahead_in_features():
    """防前视硬断言: 篡改 T 日及之后的所有原始数据(含 Phase 2 全部扩展源),
    X[:T+1] 必须一字不变.

    扩展源覆盖各时间形态: 日频对齐(行业/北向/两融/外盘)、后顾事件衰减
    (龙虎榜/增减持/解禁, 含未来日历行)、公布滞后的月频宏观。
    """
    kline, index, ff, extras = _make_synthetic()
    X1, y1, _ = build_feature_matrix(kline, index, ff, THRESHOLD, extras=extras)

    t = 80  # 篡改第 80 行及之后的全部原始数据
    k2 = kline.copy()
    k2.loc[t:, ["open", "close", "high", "low", "volume"]] *= 1.5
    i2 = index.copy()
    i2.loc[t:, "close"] *= 0.5
    f2 = ff.copy()
    # ff 覆盖 kline 第 60..119 行(ff 行 j <-> kline 行 60+j);
    # 只篡改对应 kline 行号 >= t 的部分(ff 行 20 起), 与"篡改第 t 行及之后"一致
    f2.loc[20:, "main_net_pct"] = 999.0

    # 扩展源统一规则: 篡改 date >= dates[t] 的所有行(与"信息时点 >= t"一致;
    # 宏观月频行以月初日期为信息时点, 其生效日更晚, 该口径偏保守仍然安全)
    cut = pd.Timestamp(kline["date"].iloc[t])
    ex2 = {}
    for key, df in extras.items():
        d2 = df.copy()
        mask = pd.to_datetime(d2["date"]) >= cut
        num_cols = [c for c in d2.columns if c != "date"]
        d2.loc[mask, num_cols] = d2.loc[mask, num_cols] * 3.0 + 999.0
        ex2[key] = d2

    X2, y2, _ = build_feature_matrix(k2, i2, f2, THRESHOLD, extras=ex2)

    # X 的第 0..t 行(含 t, 即"用 ≤T-1 数据"的部分)必须完全一致
    pd.testing.assert_frame_equal(X1.iloc[: t + 1], X2.iloc[: t + 1])
    # 标签 y[T] 只依赖 close_T/close_{T-1}: t 之前的行必须不变。
    # (y[t] 本身随 close_t 被篡改而合法地改变 —— 标签描述当日走势, 不属前视)
    pd.testing.assert_series_equal(y1.iloc[:t], y2.iloc[:t])
    # 而篡改必须确实影响了后续特征(否则测试退化): 整体 + 逐源抽查。
    # 注: macro 因公布滞后(月值 M+1 月 15 日生效), 被篡改月份的响应落在序列
    # 更后段, 不在 tail 抽查内 —— 其"列是活的"由下方 notna 断言保证,
    # 响应性/滞后边界由 test_sources.py::test_macro_publish_lag 专项验证。
    assert not X1.iloc[t + 1].equals(X2.iloc[t + 1])
    tail1, tail2 = X1.iloc[t + 1:], X2.iloc[t + 1:]
    for col in ["ret1", "sec_ret1", "north_net1", "lhb_decay", "holder_decay",
                "unlock_decay", "us_nasdaq_ret1"]:
        assert col in tail1.columns, f"缺少扩展特征列 {col}"
        assert not tail1[col].equals(tail2[col]), f"{col} 未受篡改影响, 测试退化"
    # 扩展特征列必须"是活的"(合成源确实产生了非 NaN 信号, 防止空列让断言退化)
    for col in ["macro_pmi", "margin_chg5", "vix_level", "excess_ret1"]:
        assert X1[col].notna().any(), f"{col} 全为 NaN, 合成数据或特征函数失效"


def test_features_lag_one_day():
    """X[T] 的收益率特征应等于 T-1 日的单日涨跌幅(而非 T 日)."""
    kline, index, ff, _extras = _make_synthetic()
    X, _, _ = build_feature_matrix(kline, index, ff, THRESHOLD)
    day_t = kline["date"].iloc[30]
    prev_ret = kline["close"].iloc[29] / kline["close"].iloc[28] - 1
    assert np.isclose(X.loc[day_t, "ret1"], prev_ret)
    # X[T] 不应等于 T 日自身涨跌幅(前视)
    same_day_ret = kline["close"].iloc[30] / kline["close"].iloc[29] - 1
    assert not np.isclose(X.loc[day_t, "ret1"], same_day_ret)


def test_fund_flow_nan_preserved_in_early_rows():
    """资金流覆盖期外的行保留 NaN(而非被 dropna 整行丢弃)."""
    kline, index, ff, _extras = _make_synthetic()
    X, y, optional = build_feature_matrix(kline, index, ff, THRESHOLD)
    assert "main_net_pct" in optional
    # 资金流覆盖第 60.. 行, shift(1) 后 X 第 61.. 行才有值
    assert X["main_net_pct"].iloc[:61].isna().all()
    assert X["main_net_pct"].iloc[61:].notna().all()


# ---------------------------------------------------------------------------
# Phase 2 扩展源: schema 稳定性与 optional 口径
# ---------------------------------------------------------------------------

PHASE2_COLS = (
    sources.INDUSTRY_COLS + sources.NORTH_COLS + sources.MARGIN_COLS
    + EVENT_COLS + sources.GLOBAL_COLS + sources.MACRO_COLS
)


def test_phase2_columns_all_optional():
    """扩展源特征列必须全部进 optional_cols(不参与 dropna, NaN 交给 LightGBM)."""
    kline, index, ff, extras = _make_synthetic()
    X, _y, optional = build_feature_matrix(kline, index, ff, THRESHOLD, extras=extras)
    for col in PHASE2_COLS:
        assert col in X.columns, f"缺少扩展特征列 {col}"
        assert col in optional, f"{col} 未进 optional_cols, 会被 dropna 误删行"


def test_phase2_schema_stable_without_sources():
    """源全缺失时列集合不变(全 NaN), 保证票池 concat 与模型特征清单稳定."""
    kline, index, ff, _ = _make_synthetic()
    X_none, _, opt_none = build_feature_matrix(kline, index, ff, THRESHOLD, extras=None)
    X_all_null, _, opt_null = build_feature_matrix(
        kline, index, ff, THRESHOLD, extras={k: None for k in EXTRA_KEYS}
    )
    assert list(X_none.columns) == list(X_all_null.columns)
    assert opt_none == opt_null
    for col in PHASE2_COLS:
        assert X_none[col].isna().all()


def test_phase2_event_sources_empty_vs_missing():
    """事件源: None(源不可用)->NaN 列; 空帧(源可用无事件)->0, 二者语义不同."""
    kline, index, ff, _ = _make_synthetic()
    base = {k: None for k in EXTRA_KEYS}
    # None(源不可用) -> 全 NaN
    X_none, _, _ = build_feature_matrix(kline, index, ff, THRESHOLD, extras=base)
    assert X_none["lhb_decay"].isna().all()
    # 空帧(源可用但无事件) -> shift 前全 0, shift(1) 后首行 NaN 其余 0
    base2 = dict(base, lhb=pd.DataFrame(columns=["date", "net_buy"]))
    X_empty, _, _ = build_feature_matrix(kline, index, ff, THRESHOLD, extras=base2)
    assert pd.isna(X_empty["lhb_decay"].iloc[0])
    assert (X_empty["lhb_decay"].iloc[1:] == 0).all()
    assert (X_empty["lhb_count"].iloc[1:] == 0).all()

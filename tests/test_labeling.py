"""防前视偏差硬断言 + 标签定义正确性.

最核心的测试: 修改第 T 日及之后的任何原始数据, X[T] 特征行必须一字不变.
"""

import numpy as np
import pandas as pd

from stock_forecast.features import labeling
from stock_forecast.features.build import build_feature_matrix

THRESHOLD = 0.005
N_DAYS = 120


def _make_synthetic(n: int = N_DAYS, seed: int = 7):
    """构造确定性的合成 K线/指数/资金流, 离线可复现."""
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
    return kline, index, fund_flow


def test_label_thresholds():
    closes = pd.Series([100.0, 101.0, 101.4, 101.6, 101.0])
    y = labeling.make_labels(closes, 0.005)
    # 变化: +1.0% -> 涨; +0.396% -> 平; +0.197% -> 平; -0.59% -> 跌
    assert y.tolist() == [np.nan, 1, 0, 0, -1]


def test_label_boundary_exclusive():
    assert labeling.pct_to_label(0.005, 0.005) == 0  # 恰好等于阈值 -> 平
    assert labeling.pct_to_label(0.0051, 0.005) == 1
    assert labeling.pct_to_label(-0.0051, 0.005) == -1


def test_no_lookahead_in_features():
    """防前视硬断言: 篡改 T 日及之后的所有原始数据, X[:T+1] 不变."""
    kline, index, ff = _make_synthetic()
    X1, y1, _ = build_feature_matrix(kline, index, ff, THRESHOLD)

    t = 80  # 篡改第 80 行及之后的全部原始数据
    k2 = kline.copy()
    k2.loc[t:, ["open", "close", "high", "low", "volume"]] *= 1.5
    i2 = index.copy()
    i2.loc[t:, "close"] *= 0.5
    f2 = ff.copy()
    f2.loc[:, "main_net_pct"] = 999.0

    X2, y2, _ = build_feature_matrix(k2, i2, f2, THRESHOLD)

    # X 的第 0..t 行(含 t, 即"用 ≤T-1 数据"的部分)必须完全一致
    pd.testing.assert_frame_equal(X1.iloc[: t + 1], X2.iloc[: t + 1])
    # 标签基于涨跌幅, 对整体缩放不变 -> 全序列标签不变
    pd.testing.assert_series_equal(y1, y2)
    # 而篡改必须确实影响了后续特征(否则测试退化)
    assert not X1.iloc[t + 1].equals(X2.iloc[t + 1])


def test_features_lag_one_day():
    """X[T] 的收益率特征应等于 T-1 日的单日涨跌幅(而非 T 日)."""
    kline, index, ff = _make_synthetic()
    X, _, _ = build_feature_matrix(kline, index, ff, THRESHOLD)
    day_t = kline["date"].iloc[30]
    prev_ret = kline["close"].iloc[29] / kline["close"].iloc[28] - 1
    assert np.isclose(X.loc[day_t, "ret1"], prev_ret)
    # X[T] 不应等于 T 日自身涨跌幅(前视)
    same_day_ret = kline["close"].iloc[30] / kline["close"].iloc[29] - 1
    assert not np.isclose(X.loc[day_t, "ret1"], same_day_ret)


def test_fund_flow_nan_preserved_in_early_rows():
    """资金流覆盖期外的行保留 NaN(而非被 dropna 整行丢弃)."""
    kline, index, ff = _make_synthetic()
    X, y, optional = build_feature_matrix(kline, index, ff, THRESHOLD)
    assert "main_net_pct" in optional
    # 资金流覆盖第 60.. 行, shift(1) 后 X 第 61.. 行才有值
    assert X["main_net_pct"].iloc[:61].isna().all()
    assert X["main_net_pct"].iloc[61:].notna().all()

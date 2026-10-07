"""L4 市场状态识别(规则引擎 v1)单测.

覆盖: 合成牛/熊/震荡/高波动覆盖的确定性判定、暖机期与降级路径、
防前视契约(篡改 T 日及之后指数数据, <T 行状态一字不变)、
预测口径对齐(regime_for_prediction 取 T-1 行)、回测分桶报表。
"""

import numpy as np
import pandas as pd

from stock_forecast.config import get_settings
from stock_forecast.features import regime as rg
from stock_forecast.model.backtest import regime_bucket_report

S = get_settings()


def _index(closes) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    n = len(c)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "open": c, "close": c, "high": c * 1.001, "low": c * 0.999,
        "volume": np.full(n, 1e8),
    })


# ---------------------------------------------------------------------------
# 确定性状态判定(手工构造均线排列)
# ---------------------------------------------------------------------------

def test_bull_alignment():
    """60 天横盘后单边拉升: 收盘 > MA20 > MA60 -> 牛市."""
    closes = np.concatenate([np.full(60, 3000.0), np.linspace(3000, 3600, 20)[1:]])
    r = rg.classify_regimes(_index(closes), S)
    assert r["regime"].iloc[-1] == rg.REGIME_BULL
    # 暖机期(MA60 无值)标记为未知
    assert (r["regime"].iloc[: S.regime_ma_slow - 1] == rg.REGIME_UNKNOWN).all()


def test_bear_alignment():
    """60 天横盘后单边下跌: 收盘 < MA20 < MA60 -> 熊市."""
    closes = np.concatenate([np.full(60, 3000.0), np.linspace(3000, 2400, 20)[1:]])
    r = rg.classify_regimes(_index(closes), S)
    assert r["regime"].iloc[-1] == rg.REGIME_BEAR


def test_mixed_alignment_is_range():
    """横盘 -> 拉升 -> 回落: 收盘 < MA20 但 MA20 > MA60(排列混杂) -> 震荡."""
    closes = np.concatenate([
        np.full(60, 3000.0),
        np.linspace(3000, 3300, 11)[1:],   # 10 天拉升
        np.linspace(3300, 3000, 11)[1:],   # 10 天回落, 收盘回到 3000
    ])
    r = rg.classify_regimes(_index(closes), S)
    row = r.iloc[-1]
    # 前置条件: 确实是"收盘<MA20 且 MA20>MA60"的混杂排列(而非其他路径进震荡)
    assert row["close"] < row["ma_fast"] and row["ma_fast"] > row["ma_slow"]
    assert row["regime"] == rg.REGIME_RANGE


def test_high_vol_overrides_bull_to_range():
    """长牛尾部波动率放大到历史极值: 多头排列但分位 >= 0.85 -> 强制震荡."""
    t = np.arange(130)
    calm = 3000 * np.exp(0.0008 * t) * (1 + 0.001 * np.sin(t))  # 低波动缓涨
    wild = [calm[-1]]
    for i in range(71):  # 高波动强拉升: +2.5% / -1% 交替, 净上行
        wild.append(wild[-1] * (1.025 if i % 2 == 0 else 0.99))
    closes = np.concatenate([calm, wild[1:]])
    r = rg.classify_regimes(_index(closes), S)
    row = r.iloc[-1]
    # 前置条件: 均线仍为多头排列, 但波动率分位打到高位
    assert row["close"] > row["ma_fast"] > row["ma_slow"]
    assert row["vol_quantile"] >= S.regime_high_vol_q
    assert row["regime"] == rg.REGIME_RANGE


# ---------------------------------------------------------------------------
# 降级路径: 数据不可用 / 暖机期
# ---------------------------------------------------------------------------

def test_warmup_unknown_and_prediction_unavailable():
    closes = np.full(40, 3000.0)
    r = rg.classify_regimes(_index(closes), S)
    assert (r["regime"] == rg.REGIME_UNKNOWN).all()
    pred = rg.regime_for_prediction(_index(closes), S)
    assert pred["available"] is False
    assert pred["regime"] == rg.REGIME_RANGE  # 展示兜底值
    assert "历史不足" in pred["reason"]


def test_prediction_unavailable_on_none_or_tiny_index():
    for bad in (None, _index([3000.0])):
        pred = rg.regime_for_prediction(bad, S)
        assert pred["available"] is False
        assert pred["as_of"] is None


# ---------------------------------------------------------------------------
# 防前视契约与预测口径(T-1)对齐
# ---------------------------------------------------------------------------

def test_classify_no_lookahead():
    """篡改第 t 行及之后的指数收盘, < t 行的状态与数值必须一字不变."""
    rng = np.random.default_rng(11)
    closes = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    r1 = rg.classify_regimes(_index(closes), S)
    c2 = closes.copy()
    t = 200
    c2[t:] *= 1.3
    r2 = rg.classify_regimes(_index(c2), S)
    pd.testing.assert_frame_equal(r1.iloc[:t], r2.iloc[:t])
    # 篡改必须影响后续行(否则测试退化)
    assert not r1.iloc[t:].equals(r2.iloc[t:])


def test_prediction_uses_previous_close():
    """末日暴跌只进入最后一行状态; 预测口径取 T-1 行, 必须不受影响.

    这是 regime 与 X[T](统一 shift(1))信息时点对齐的核心断言: 若预测直接用
    T 行状态, 当日收盘信息会经融合调整泄露进对 y[T] 的预测。
    """
    rng = np.random.default_rng(5)
    closes = 3000 * np.exp(np.cumsum(rng.normal(0.001, 0.008, 300)))
    idx = _index(closes)
    r_orig = rg.classify_regimes(idx, S)

    crash = np.concatenate([closes, [closes[-1] * 0.7]])
    pred = rg.regime_for_prediction(_index(crash), S)
    assert pred["available"] is True
    # 预测暴跌日走势所用状态 = 暴跌前一日(T-1)收盘状态, 与原序列末行一致
    assert pred["regime"] == r_orig["regime"].iloc[-1]
    assert pred["as_of"] == str(r_orig.index[-1].date())
    # 而暴跌确实进入了"截至最新收盘"的状态(两口径不同, 证明对齐生效)
    r_ext = rg.classify_regimes(_index(crash), S)
    assert r_ext["close"].iloc[-1] < r_orig["close"].iloc[-1]


def test_shifted_regimes_alignment():
    rng = np.random.default_rng(3)
    closes = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    idx = _index(closes)
    r = rg.classify_regimes(idx, S)["regime"]
    s = rg.shifted_regimes(idx, S)
    assert pd.isna(s.iloc[0])
    pd.testing.assert_series_equal(
        s.iloc[1:].reset_index(drop=True), r.iloc[:-1].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 回测分桶报表
# ---------------------------------------------------------------------------

def test_regime_bucket_report_bear():
    """稳定下跌市: 明细日期全部落入"熊市"桶, 方向准确率手算核对."""
    closes = 3000 * 0.995 ** np.arange(120)
    idx = _index(closes)
    dates = pd.DatetimeIndex(idx["date"])
    detail = pd.DataFrame({
        "date": dates[-30:],
        "y_true": [1] * 15 + [-1] * 15,
        "y_pred": [1] * 10 + [0] * 5 + [-1] * 15,
        "fold": 0,
    })
    buckets = regime_bucket_report(detail, idx, S)
    assert buckets is not None
    assert set(buckets) == {"熊市"}
    b = buckets["熊市"]
    assert b["n"] == 30
    # 方向口径(本例无真实"平"): 命中 10 + 15 = 25
    assert np.isclose(b["direction_accuracy"], 25 / 30)


def test_regime_bucket_report_unmapped_and_none():
    """首个交易日(无前日)与指数未覆盖的日期 -> "未知"桶; index 缺失 -> None."""
    closes = 3000 * 0.995 ** np.arange(120)
    idx = _index(closes)
    dates = pd.DatetimeIndex(idx["date"])
    detail = pd.DataFrame({
        "date": [dates[0], pd.Timestamp("2020-01-01")],
        "y_true": [1, -1], "y_pred": [1, -1], "fold": 0,
    })
    buckets = regime_bucket_report(detail, idx, S)
    assert buckets["未知"]["n"] == 2
    assert regime_bucket_report(detail, None, S) is None

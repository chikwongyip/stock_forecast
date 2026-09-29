"""模型与回测框架测试: 时间有序性、泄露灵敏度、保存/加载."""

import numpy as np
import pandas as pd
import pytest

from stock_forecast.model import metrics, train
from stock_forecast.model.backtest import run_backtest, walk_forward_folds


def _synthetic_pool(n_days: int = 150, n_symbols: int = 3, seed: int = 0):
    """合成票池: 特征为有信息量特征 + 纯噪声特征, 标签由有信息量特征决定."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    for k in range(n_symbols):
        informative = rng.normal(0, 1, n_days)
        noise = rng.normal(0, 1, (n_days, 5))
        df = pd.DataFrame(noise, index=dates)
        df["informative"] = informative
        # 标签由 informative 的符号决定(阈值化的涨/跌/平)
        y = pd.Series(
            np.select(
                [informative > 0.5, informative < -0.5], [1, -1], default=0
            ),
            index=dates,
        )
        Xs.append(df.assign(symbol=f"S{k}"))
        ys.append(y)
    X = pd.concat(Xs).sort_index()
    y = pd.concat(ys).sort_index()
    return X, y


def test_walk_forward_folds_time_ordered():
    dates = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=100, freq="B")
    )
    folds = walk_forward_folds(dates, train_ratio=0.6, test_window=20)
    assert len(folds) == 2
    for train_dates, test_dates in folds:
        assert train_dates.max() < test_dates.min()  # 严格时间先后
        assert len(set(train_dates) & set(test_dates)) == 0


def test_backtest_detects_leaky_signal():
    """informative 特征直接决定标签 -> 方向准确率应接近 100%.

    这是回测口径的'泄露灵敏度'自检: 管道能捕捉强信号, 评估不是摆设.
    """
    X, y = _synthetic_pool()
    detail, summary = run_backtest(X.drop(columns=["symbol"]), y, 0.6, 20)
    assert summary["direction_accuracy"] > 0.95
    assert summary["possible_leakage"] is True


def test_backtest_pure_noise_is_not_significant():
    """纯噪声特征 -> 方向准确率应远离 95%, 不产生泄露误报."""
    rng = np.random.default_rng(3)
    n = 150
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(0, 1, (n, 6)), index=dates)
    y = pd.Series(rng.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]), index=dates)
    _, summary = run_backtest(X, y, 0.6, 20)
    assert summary["direction_accuracy"] < 0.95
    assert summary["possible_leakage"] is False


def test_direction_accuracy_excludes_flat():
    y_true = pd.Series([1, -1, 0, 1])
    y_pred = pd.Series([1, 1, 1, -1])
    # 剔除真实"平"后: 命中 1/3
    assert np.isclose(metrics.direction_accuracy(y_true, y_pred), 1 / 3)


def test_model_save_load_roundtrip(tmp_path):
    X, y = _synthetic_pool()
    model = train.train_model(X.drop(columns=["symbol"]), y)
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    from lightgbm import Booster

    booster = Booster(model_file=str(path))
    row = X.drop(columns=["symbol"]).iloc[[-1]]
    p1 = train.predict_proba_df(model.booster_, row)
    p2 = train.predict_proba_df(booster, row)
    pd.testing.assert_frame_equal(p1, p2)
    # 概率归一
    assert np.isclose(p1.values.sum(), 1.0)

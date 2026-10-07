"""特征漂移 PSI 监控单测 (Phase 7.1): 全部离线合成数据, 不触网.

覆盖: PSI 纯函数口径(同分布≈0 / 平移分布高 / 仅非 NaN 样本)、缺失率
单边检查(恶化告警、改善不告警)、基线构造(退化列、期望比例归一)、
基线读写容错、分级阈值边界、告警落盘与读取、run_drift_check 无基线降级。
"""

import json

import numpy as np
import pandas as pd
import pytest

from stock_forecast.monitor import drift


def _train_frame(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ret1": rng.normal(0.0, 0.02, n),
        "vol_ratio": rng.normal(1.0, 0.3, n),
        "const_col": np.zeros(n),                       # 退化列(常量)
        "nan_col": rng.choice([np.nan, 1.0], n, p=[0.2, 0.8]),  # 20% 缺失
    })


# ---------- PSI 纯函数 ----------

def test_psi_identical_distribution_near_zero():
    X = _train_frame()
    base = drift.build_baseline(X)
    psi, _ = drift.feature_psi(X["ret1"], base["features"]["ret1"])
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_alert_level():
    X = _train_frame()
    base = drift.build_baseline(X)
    shifted = X["ret1"] + 5 * X["ret1"].std()  # 整体平移 -> 质量堆到最高箱
    psi, _ = drift.feature_psi(shifted, base["features"]["ret1"])
    assert psi >= 0.25  # alert 级


def test_nan_jump_alerts_and_improvement_does_not():
    """缺失率检查是单边的: 覆盖恶化 -> alert; 覆盖改善 -> 不告警."""
    X = _train_frame()
    base = drift.build_baseline(X)
    spec = base["features"]["nan_col"]
    assert spec["nan_ratio"] == pytest.approx(0.2, abs=0.05)

    # 恶化: 近端 90% 缺失(非缺失部分同分布) -> nan_jump -> alert
    rng = np.random.default_rng(0)
    worse = pd.Series(rng.choice([np.nan, 1.0], 200, p=[0.9, 0.1]))
    row = drift.evaluate_feature("nan_col", worse, spec, 0.10, 0.25, 0.20)
    assert row["nan_jump"] is True and row["level"] == "alert"

    # 全 NaN: PSI 不可算(None), 缺失率检查兜底 -> alert
    all_nan = drift.evaluate_feature("nan_col", pd.Series([np.nan] * 100),
                                     spec, 0.10, 0.25, 0.20)
    assert all_nan["psi"] is None and all_nan["level"] == "alert"

    # 改善: 近端 0% 缺失且数值同分布 -> 不告警(资金流覆盖期场景)
    # (常量列有 1 个空箱, 半样本平滑下 psi 有 ~0.5/n 量级的小底, 远低于 warn)
    better = pd.Series([1.0] * 100)
    row2 = drift.evaluate_feature("nan_col", better, spec, 0.10, 0.25, 0.20)
    assert row2["nan_jump"] is False and row2["level"] == "ok"
    assert row2["psi"] < 0.10


def test_psi_values_zero_when_equal():
    e = np.array([0.5, 0.3, 0.2])
    assert drift.psi_values(e, e) == pytest.approx(0.0, abs=1e-9)


def test_psi_empty_bin_penalty_scales_with_n():
    """半样本平滑: 小样本全挤一箱时, 空箱惩罚不应达到固定 eps 的量级(~100)."""
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    spec = drift.build_baseline(train)["features"]["x"]
    small = pd.Series([5.0] * 30)  # 30 样本全落最高箱, 其余箱为空
    psi, _ = drift.feature_psi(small, spec)
    assert psi >= 0.25          # 仍然是显著漂移
    assert psi < 5.0            # 但惩罚被 n 平滑封顶(固定 1e-6 时会 >50)


# ---------- 基线构造 ----------

def test_build_baseline_structure():
    X = _train_frame()
    base = drift.build_baseline(X, n_bins=10)
    assert base["baseline_version"] == drift.BASELINE_VERSION
    assert base["n_samples"] == len(X)
    assert set(base["features"]) == set(X.columns)
    for spec in base["features"].values():
        # 期望比例 = 非 NaN 样本内的分箱比例(归一)
        assert sum(spec["expected"]) == pytest.approx(1.0, abs=1e-9)
        assert len(spec["expected"]) == len(spec["edges"]) + 1


def test_build_baseline_constant_column_single_edge():
    # 常量列: 分位边界去重后剩 1 个(非退化), 可监控单边漂移(口径见模块 docstring)
    X = _train_frame()
    base = drift.build_baseline(X)
    spec = base["features"]["const_col"]
    assert spec["degenerate"] is False
    assert spec["edges"] == [0.0]
    psi, _ = drift.feature_psi(X["const_col"], spec)
    assert psi < 0.02  # 同分布: 只剩半样本平滑的空箱小底


def test_build_baseline_degenerate_column():
    # 真退化 = 非 NaN 样本 < 2: 无数值边界, 数值漂移不监控(只剩缺失率检查)
    X = pd.DataFrame({"sparse": [np.nan] * 99 + [1.0]})
    base = drift.build_baseline(X)
    spec = base["features"]["sparse"]
    assert spec["degenerate"] is True
    assert spec["edges"] == []
    assert len(spec["expected"]) == 1
    psi, _ = drift.feature_psi(pd.Series([np.nan] * 99 + [1.0]), spec)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_feature_psi_spec_mismatch_returns_none():
    # expected 长度必须 = len(edges)+1(非 NaN 分箱数)
    bad_spec = {"edges": [0.0, 1.0], "expected": [0.5, 0.5], "nan_ratio": 0.0}
    assert drift.feature_psi(pd.Series([0.1, 0.2, 0.3]), bad_spec) is None
    # 空输入不可评估
    assert drift.feature_psi(pd.Series([], dtype=float),
                             {"edges": [], "expected": [1.0]}) is None


# ---------- 分级 ----------

def test_classify_level_boundaries():
    assert drift.classify_level(0.05, 0.10, 0.25) == "ok"
    assert drift.classify_level(0.10, 0.10, 0.25) == "warn"   # 恰达阈值
    assert drift.classify_level(0.24, 0.10, 0.25) == "warn"
    assert drift.classify_level(0.25, 0.10, 0.25) == "alert"  # 恰达阈值
    assert drift.classify_level(1.5, 0.10, 0.25) == "alert"


# ---------- 基线与告警 IO ----------

def test_baseline_roundtrip(tmp_path):
    p = tmp_path / "drift_baseline.json"
    base = drift.build_baseline(_train_frame(100))
    drift.save_baseline(base, p)
    loaded = drift.load_baseline(p)
    assert loaded["features"].keys() == base["features"].keys()


def test_load_baseline_missing_or_bad(tmp_path):
    assert drift.load_baseline(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert drift.load_baseline(bad) is None
    not_feats = tmp_path / "nf.json"
    not_feats.write_text(json.dumps({"baseline_version": 1}), encoding="utf-8")
    assert drift.load_baseline(not_feats) is None


def test_append_alert_and_load(tmp_path):
    p = tmp_path / "alerts.csvl"
    drift.append_alert("feature_drift", "warn", path=p, feature="ret1", psi=0.12)
    drift.append_alert("retrain_rejected", "warn", path=p, reason="x")
    p.write_text(p.read_text(encoding="utf-8") + "{broken\n\n", encoding="utf-8")
    records = drift.load_alerts(p)
    assert len(records) == 2  # 坏行/空行跳过
    assert records[0]["kind"] == "feature_drift" and records[0]["psi"] == 0.12
    assert records[1]["kind"] == "retrain_rejected"
    assert all("ts" in r and "level" in r for r in records)


def test_load_alerts_missing_file(tmp_path):
    assert drift.load_alerts(tmp_path / "nope.csvl") == []


# ---------- run_drift_check 降级 ----------

def test_run_drift_check_baseline_missing(tmp_path):
    result = drift.run_drift_check(
        symbols=["600519"], baseline_path=tmp_path / "nope.json"
    )
    assert result["ok"] is False
    assert result["baseline_missing"] is True
    assert "backtest" in result["error"] or "retrain" in result["error"]


def test_run_drift_check_baseline_version_mismatch(tmp_path):
    """旧格式基线整体拒绝(版本闸门), 防止'全部评估失败 -> 假绿'."""
    p = tmp_path / "b.json"
    base = drift.build_baseline(_train_frame(50))
    base["baseline_version"] = 0
    drift.save_baseline(base, p)
    result = drift.run_drift_check(symbols=["600519"], baseline_path=p)
    assert result["ok"] is False and result["baseline_missing"] is True
    assert "版本" in result["error"]

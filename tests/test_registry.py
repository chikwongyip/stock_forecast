"""模型注册表(Phase 4 / L5)单测: 统一接口、元数据落盘、集成与分歧度、降级路径.

Stub 成员用于确定性验证集成数学(等权平均/TV 分歧度); 真实 LGBM/XGB 用于
fit->save->load 工件回环。全部离线合成数据。
"""

import json

import numpy as np
import pandas as pd
import pytest

from stock_forecast.model import registry, train

XGB_SKIP = pytest.mark.skipif(not registry.XGBOOST_AVAILABLE, reason="xgboost 未安装")


def _synthetic(n_days: int = 150, n_symbols: int = 2, seed: int = 0):
    """informative 特征决定标签 + 噪声特征(与 test_model 同构)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    for _k in range(n_symbols):
        informative = rng.normal(0, 1, n_days)
        df = pd.DataFrame(rng.normal(0, 1, (n_days, 4)), index=dates, columns=list("abcd"))
        df["informative"] = informative
        y = pd.Series(
            np.select([informative > 0.5, informative < -0.5], [1, -1], default=0),
            index=dates,
        )
        Xs.append(df)
        ys.append(y)
    return pd.concat(Xs).sort_index(), pd.concat(ys).sort_index()


class _StubModel(registry.BaseModel):
    """固定概率的桩成员: 确定性验证集成平均与 TV 分歧度."""

    def __init__(self, stub_name: str, probs: list[float]):
        super().__init__()
        self._stub_name = stub_name
        self._probs = probs

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._stub_name

    def fit(self, X, y) -> None:
        pass

    def predict_proba(self, X):
        return pd.DataFrame(
            [self._probs] * len(X), index=X.index,
            columns=["p_down", "p_flat", "p_up"],
        )

    def save(self, models_dir):
        raise NotImplementedError

    def load_artifact(self, path):
        raise NotImplementedError

    def feature_names(self):
        return ["a"]


# ---------------------------------------------------------------------------
# 统一接口: fit / predict_proba / save / load
# ---------------------------------------------------------------------------

def test_lgbm_fit_predict_save_load(tmp_path):
    X, y = _synthetic()
    m = registry.LGBMModel()
    m.fit(X, y)
    row = X.iloc[[-1]]
    p = m.predict_proba(row)
    assert list(p.columns) == ["p_down", "p_flat", "p_up"]  # 固定列序(不变量 7)
    assert p.index.equals(row.index)
    assert np.isclose(p.values.sum(), 1.0)
    assert m.feature_names() == list(X.columns)

    f = m.save(tmp_path)
    m2 = registry.LGBMModel()
    m2.load_artifact(tmp_path / f)
    pd.testing.assert_frame_equal(m2.predict_proba(row), p)
    assert m2.feature_names() == list(X.columns)


@XGB_SKIP
def test_xgb_fit_predict_save_load(tmp_path):
    X, y = _synthetic()
    m = registry.XGBModel()
    m.fit(X, y)
    row = X.iloc[[-1]]
    p = m.predict_proba(row)
    assert list(p.columns) == ["p_down", "p_flat", "p_up"]
    assert np.isclose(p.values.sum(), 1.0)

    f = m.save(tmp_path)
    m2 = registry.XGBModel()
    m2.load_artifact(tmp_path / f)
    p2 = m2.predict_proba(row)
    pd.testing.assert_frame_equal(p2, p, atol=1e-6, rtol=1e-6)
    assert m2.feature_names() == list(X.columns)


@XGB_SKIP
def test_xgb_missing_class_fixed_columns():
    """训练数据缺"平"类: 原生 num_class=3 保证输出仍为固定 3 列(schema 稳定)."""
    X, _y = _synthetic()
    y = pd.Series(np.where(np.arange(len(X)) % 2 == 0, 1, -1), index=X.index)
    m = registry.XGBModel()
    m.fit(X, y)
    p = m.predict_proba(X.iloc[[-1]])
    assert list(p.columns) == ["p_down", "p_flat", "p_up"]
    assert np.isclose(p.values.sum(), 1.0)


# ---------------------------------------------------------------------------
# 集成: 等权平均 + TV 分歧度(确定性桩)
# ---------------------------------------------------------------------------

def test_ensemble_mean_and_disagreement():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    ens = registry.EnsembleModel([
        _StubModel("m1", [0.1, 0.1, 0.8]),
        _StubModel("m2", [0.5, 0.3, 0.2]),
    ])
    p = ens.predict_proba(X)
    assert np.allclose(p["p_up"], 0.5)      # (0.8+0.2)/2
    assert np.allclose(p["p_down"], 0.3)    # (0.1+0.5)/2
    assert np.allclose(p["p_flat"], 0.2)
    # TV = 0.5*(|0.1-0.5| + |0.1-0.3| + |0.8-0.2|) = 0.5*1.2 = 0.6
    assert np.allclose(ens.disagreement(X), 0.6)
    assert ens.name == "m1+m2"


def test_ensemble_identical_members_zero_disagreement():
    X = pd.DataFrame({"a": [1.0]})
    ens = registry.EnsembleModel([
        _StubModel("m1", [0.2, 0.3, 0.5]),
        _StubModel("m2", [0.2, 0.3, 0.5]),
    ])
    assert float(ens.disagreement(X).iloc[0]) == 0.0


def test_ensemble_requires_members():
    with pytest.raises(ValueError):
        registry.EnsembleModel([])


# ---------------------------------------------------------------------------
# 注册表: 元数据落盘 / 加载 / 旧版回退 / 空目录
# ---------------------------------------------------------------------------

def test_save_load_predictor_single_lgbm(tmp_path):
    X, y = _synthetic()
    m = registry.LGBMModel()
    m.fit(X, y)
    registry.save_predictor(
        m, feature_names=list(X.columns), train_dates=X.index,
        n_samples=len(X), models_dir=tmp_path,
    )
    meta = json.loads((tmp_path / registry.META_FILE).read_text(encoding="utf-8"))
    assert meta["ensemble"] is False
    assert meta["members"] == [{"name": "lgbm", "file": "pool_model.txt"}]
    # 4.1 验收: 元数据带特征列清单与训练数据时间范围
    assert meta["feature_names"] == list(X.columns)
    assert meta["train_date_range"] == [str(X.index.min().date()), str(X.index.max().date())]
    assert meta["n_train_samples"] == len(X)

    loaded, lmeta = registry.load_predictor(tmp_path)
    assert isinstance(loaded, registry.LGBMModel)
    assert lmeta["skipped_members"] == []
    row = X.iloc[[-1]]
    pd.testing.assert_frame_equal(loaded.predict_proba(row), m.predict_proba(row))


@XGB_SKIP
def test_save_load_ensemble_roundtrip(tmp_path):
    X, y = _synthetic()
    ens = registry.EnsembleModel([registry.LGBMModel(), registry.XGBModel()])
    ens.fit(X, y)
    registry.save_predictor(ens, list(X.columns), X.index, len(X), models_dir=tmp_path)
    loaded, meta = registry.load_predictor(tmp_path)
    assert isinstance(loaded, registry.EnsembleModel)
    assert loaded.name == "lgbm+xgb"
    assert meta["ensemble"] is True and meta["skipped_members"] == []
    row = X.iloc[[-1]]
    pd.testing.assert_frame_equal(
        loaded.predict_proba(row), ens.predict_proba(row), atol=1e-6, rtol=1e-6
    )
    # 加载后的分歧度可计算(4.5 链路)
    assert float(loaded.disagreement(row).iloc[0]) >= 0.0
    assert (tmp_path / "pool_model.txt").exists() and (tmp_path / "pool_xgb.json").exists()


def test_load_predictor_legacy_no_meta(tmp_path):
    """旧版只有 pool_model.txt(无 meta): 回退单 LightGBM, 特征清单从 booster 恢复."""
    X, y = _synthetic()
    model = train.train_model(X, y)
    model.booster_.save_model(str(tmp_path / "pool_model.txt"))
    loaded, meta = registry.load_predictor(tmp_path)
    assert isinstance(loaded, registry.LGBMModel)
    assert meta["meta_version"] == 0
    assert loaded.feature_names() == list(X.columns)
    row = X.iloc[[-1]]
    pd.testing.assert_frame_equal(
        loaded.predict_proba(row), train.predict_proba_df(model.booster_, row)
    )


def test_load_predictor_empty_dir(tmp_path):
    assert registry.load_predictor(tmp_path) is None


def test_available_models_matches_env():
    names = [c.name for c in registry.available_models()]
    assert names[0] == "lgbm"  # LightGBM 必备
    assert ("xgb" in names) == registry.XGBOOST_AVAILABLE

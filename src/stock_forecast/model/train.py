"""LightGBM 三分类模型: 训练、持久化、推理.

标签映射: {-1 跌, 0 平, 1 涨} -> {0, 1, 2} (LightGBM 要求 0..K-1).
单票一年仅 ~240 样本, 参数朝欠拟合方向保守配置, 类权重均衡处理"平"多数类.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import Booster, LGBMClassifier, early_stopping

from stock_forecast.config import MODELS_DIR
from stock_forecast.features.labeling import DOWN, FLAT, UP

LABEL_MAP = {DOWN: 0, FLAT: 1, UP: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
MODEL_PATH = MODELS_DIR / "pool_model.txt"


def make_model(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        class_weight="balanced",
        importance_type="gain",
        random_state=seed,
        verbosity=-1,
    )


def _time_ordered_val_split(
    X: pd.DataFrame, y: pd.Series, val_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """按时间轴从训练窗尾部切验证集(严禁随机分割)."""
    n_val = max(20, int(len(X) * val_ratio))
    return X.iloc[:-n_val], y.iloc[:-n_val], X.iloc[-n_val:], y.iloc[-n_val:]


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> LGBMClassifier:
    """训练; 用尾部 20% 做早停验证集(时间有序)."""
    y_enc = y.map(LABEL_MAP)
    X_tr, y_tr, X_val, y_val = _time_ordered_val_split(X, y_enc)
    model = make_model(seed)
    # eval_X/eval_y 为 lightgbm>=4.7 的新式参数(替代已弃用的 eval_set, Phase 4.1 处理)
    model.fit(
        X_tr, y_tr,
        eval_X=X_val, eval_y=y_val,
        callbacks=[early_stopping(20, verbose=False)],
    )
    return model


def train_final_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> LGBMClassifier:
    """全量训练并落盘, 供 predict 使用."""
    model = train_model(X, y, seed)
    save_model(model)
    return model


def save_model(model: LGBMClassifier) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))


def load_booster() -> Booster | None:
    """加载已保存的模型; 不存在返回 None."""
    if not MODEL_PATH.exists():
        return None
    return Booster(model_file=str(MODEL_PATH))


def predict_proba_df(booster: Booster, X: pd.DataFrame) -> pd.DataFrame:
    """原生 Booster 推理, 输出按 [-1, 0, 1] 命名的概率: p_down/p_flat/p_up.

    LightGBM 类别顺序即训练时的编码 {0:跌, 1:平, 2:涨}.
    """
    probs = booster.predict(X)
    if probs.ndim == 1:  # 单行输入时 LightGBM 返回 (3,)
        probs = probs.reshape(1, -1)
    return pd.DataFrame(probs, index=X.index, columns=["p_down", "p_flat", "p_up"])


def top_features(model: LGBMClassifier, n: int = 10) -> pd.Series:
    return (
        pd.Series(model.feature_importances_, index=model.feature_name_)
        .sort_values(ascending=False)
        .head(n)
    )

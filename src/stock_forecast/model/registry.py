"""模型注册表 (Phase 4 / L5): 统一 fit/predict_proba/save/load 接口 + 元数据落盘.

设计:
- `BaseModel` 抽象接口: fit(X, y)(y 为原生三分类 {-1,0,1}, 内部做 LABEL_MAP 编码)、
  predict_proba(X) -> DataFrame[p_down/p_flat/p_up](列序固定, 不变量 7)、
  save/load_artifact、feature_names();
- `LGBMModel`: LightGBM(迁入 train.py 现有逻辑: 早停验证集 = 训练窗尾部 20%, 时间有序);
- `XGBModel`: XGBoost 第二模型(4.2), 同特征同标签、独立早停; **库缺失时自动移出
  注册表**(不变量 5, 集成退化为单模型不 crash);
- `EnsembleModel`: 等权概率平均(4.3); `disagreement(X)` = 成员概率两两全变差
  (TV)距离均值 ∈ [0,1](4.5 不确定性量化, 融合层据此降档);
- 落盘布局(data/models/): `pool_model.txt`(LightGBM 原生, 文件名与旧版一致)、
  `pool_xgb.json`(XGBoost 原生)、`pool_meta.json`(注册表元数据: 成员清单/
  **特征列清单/训练数据时间范围**/样本数, 任务 4.1 验收项)。
  旧版无 meta 的 pool_model.txt 自动回退单 LightGBM 路径(向后兼容)。
"""

from __future__ import annotations

import abc
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecast.config import MODELS_DIR
from stock_forecast.model import train

try:  # XGBoost 为可选依赖: 缺失时注册表只含 LightGBM, 全链路优雅降级
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    xgb = None
    XGBOOST_AVAILABLE = False

META_FILE = "pool_meta.json"


class BaseModel(abc.ABC):
    """统一模型接口: 所有成员必须输出固定列序 p_down/p_flat/p_up 的概率帧."""

    name: str = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """y 为原生三分类 {-1, 0, 1}; 编码(LABEL_MAP)在实现内部完成."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回列固定为 [p_down, p_flat, p_up]、index 与 X 一致的概率帧."""

    @abc.abstractmethod
    def save(self, models_dir: Path) -> str:
        """落盘模型工件, 返回文件名(相对 models_dir)."""

    @abc.abstractmethod
    def load_artifact(self, path: Path) -> None:
        """从工件文件加载(推理态)."""

    @abc.abstractmethod
    def feature_names(self) -> list[str]:
        """训练时的特征列清单(加载态从工件/元数据恢复)."""


class LGBMModel(BaseModel):
    """LightGBM 三分类(现有 train.py 逻辑的注册表封装)."""

    name = "lgbm"
    FILE = "pool_model.txt"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._model = None
        self._booster = None
        self._features: list[str] | None = None

    @property
    def booster(self):
        """原生 Booster(特征重要性归因等场景使用)."""
        return self._booster

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = train.train_model(X, y, seed=self.seed)
        self._booster = self._model.booster_
        self._features = list(X.columns)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            raise RuntimeError("LGBMModel 尚未训练/加载")
        return train.predict_proba_df(self._booster, X)

    def save(self, models_dir: Path) -> str:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(models_dir / self.FILE))
        return self.FILE

    def load_artifact(self, path: Path) -> None:
        from lightgbm import Booster

        self._booster = Booster(model_file=str(path))

    def feature_names(self) -> list[str]:
        if self._features is not None:
            return self._features
        return list(self._booster.feature_name())


def _balanced_weights(y_enc: pd.Series) -> pd.Series:
    """类均衡样本权重(等价于 LightGBM class_weight='balanced' 的口径)."""
    counts = y_enc.value_counts()
    k = len(train.LABEL_MAP)
    return y_enc.map({c: len(y_enc) / (k * n) for c, n in counts.items()}).astype(float)


class XGBModel(BaseModel):
    """XGBoost 三分类第二模型(4.2): 同特征同标签, 参数对齐 LightGBM 的保守口径.

    实现要点:
    - **原生 Booster API 而非 sklearn 包装器**: 显式 num_class=3, 训练窗缺类时
      输出仍恒为 3 列固定列序(schema 稳定, 与 LightGBM 口径一致); xgboost 3.x
      的 sklearn 包装器要求标签连续从 0 起(缺类会 ValueError), 原生 API 无此限制;
    - 早停: 训练窗尾部 20%(与 LightGBM 同一时间有序分割, 不变量 2),
      推理显式 iteration_range=(0, best_iteration+1);
    - 类均衡: DMatrix weight(等价 LightGBM class_weight="balanced");
    - min_child_weight 为 **hessian 和**口径(≈ LGBM min_child_samples × 平均
      hessian ~0.2), 取 10 对齐"50 样本"的保守度; 直接抄 50 会让小训练窗无法分裂。
    """

    name = "xgb"
    FILE = "pool_xgb.json"
    NUM_BOOST_ROUND = 300
    EARLY_STOPPING_ROUNDS = 20

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._booster = None
        self._features: list[str] | None = None

    def _params(self) -> dict:
        return {
            "objective": "multi:softprob",
            "num_class": 3,             # 显式声明: 缺类时仍输出固定 3 列
            "eta": 0.05,
            "max_depth": 6,             # ≈ LightGBM num_leaves=31 的容量口径
            "min_child_weight": 10,     # hessian 和口径, ≈ LGBM min_child_samples=50
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "verbosity": 0,
            "seed": self.seed,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost 未安装, XGBModel 不可用")
        y_enc = y.map(train.LABEL_MAP)
        X_tr, y_tr, X_val, y_val = train._time_ordered_val_split(X, y_enc)
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=_balanced_weights(y_tr))
        dval = xgb.DMatrix(X_val, label=y_val)
        self._booster = xgb.train(
            self._params(), dtr,
            num_boost_round=self.NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        self._features = list(X.columns)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._booster is None:
            raise RuntimeError("XGBModel 尚未训练/加载")
        # 早停训得的模型只用最优轮数推理(best_iteration 随 JSON 工件持久化)
        bi = getattr(self._booster, "best_iteration", None)
        iteration_range = (0, int(bi) + 1) if bi is not None else (0, 0)
        raw = np.asarray(
            self._booster.predict(xgb.DMatrix(X), iteration_range=iteration_range)
        )
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return pd.DataFrame(raw, index=X.index, columns=["p_down", "p_flat", "p_up"])

    def save(self, models_dir: Path) -> str:
        models_dir.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(models_dir / self.FILE))
        return self.FILE

    def load_artifact(self, path: Path) -> None:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost 未安装, 无法加载 XGBModel 工件")
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))

    def feature_names(self) -> list[str]:
        if self._features is not None:
            return self._features
        return list(self._booster.feature_names or [])


class EnsembleModel(BaseModel):
    """等权概率平均集成(4.3) + 成员分歧度(4.5).

    disagreement = 成员概率两两全变差(TV)距离的均值 ∈ [0, 1]:
    0 = 完全一致, 1 = 完全对立; 双成员时即 0.5*Σ|p_a - p_b|。
    """

    def __init__(self, members: list[BaseModel], seed: int = 42):
        super().__init__(seed)
        if not members:
            raise ValueError("EnsembleModel 至少需要一个成员")
        self.members = members

    @property
    def name(self) -> str:  # type: ignore[override]
        return "+".join(m.name for m in self.members)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for m in self.members:
            m.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        acc = None
        for m in self.members:
            p = m.predict_proba(X)
            acc = p if acc is None else acc + p
        return acc / len(self.members)

    def disagreement(self, X: pd.DataFrame) -> pd.Series:
        probs = [m.predict_proba(X).to_numpy() for m in self.members]
        if len(probs) < 2:
            return pd.Series(0.0, index=X.index)
        tvs = [0.5 * np.abs(a - b).sum(axis=1) for a, b in itertools.combinations(probs, 2)]
        return pd.Series(np.mean(tvs, axis=0), index=X.index)

    def save(self, models_dir: Path) -> str:
        raise NotImplementedError("EnsembleModel 由 save_predictor 逐成员落盘")

    def load_artifact(self, path: Path) -> None:
        raise NotImplementedError("EnsembleModel 由 load_predictor 逐成员加载")

    def feature_names(self) -> list[str]:
        return self.members[0].feature_names()


_MODEL_CLASSES: dict[str, type[BaseModel]] = {"lgbm": LGBMModel, "xgb": XGBModel}


def available_models() -> list[type[BaseModel]]:
    """当前环境可用的模型类(LightGBM 必备; XGBoost 缺库时自动移出)."""
    out: list[type[BaseModel]] = [LGBMModel]
    if XGBOOST_AVAILABLE:
        out.append(XGBModel)
    return out


def save_predictor(
    predictor: BaseModel,
    feature_names: list[str],
    train_dates: pd.DatetimeIndex | None = None,
    n_samples: int | None = None,
    models_dir: Path = MODELS_DIR,
    backtest_summary: dict | None = None,
) -> Path:
    """落盘 predictor(集成为逐成员工件)+ 注册表元数据 pool_meta.json.

    元数据含: 成员清单、**特征列清单、训练数据时间范围**(4.1 验收)、样本数、
    训练时刻、**回测指标 backtest**(Phase 7.2: 自动重训"不劣于旧模型"的对比基准;
    旧工件无该字段, 读取端按 None 降级)。predict 的特征一致性检查以元数据为准。
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    members = predictor.members if isinstance(predictor, EnsembleModel) else [predictor]
    entries = [{"name": m.name, "file": m.save(models_dir)} for m in members]
    meta = {
        "meta_version": 1,
        "ensemble": len(members) > 1,
        "members": entries,
        "feature_names": [str(c) for c in feature_names],
        "train_date_range": (
            [str(pd.Timestamp(train_dates.min()).date()), str(pd.Timestamp(train_dates.max()).date())]
            if train_dates is not None and len(train_dates) else None
        ),
        "n_train_samples": None if n_samples is None else int(n_samples),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "backtest": backtest_summary,
    }
    meta_path = models_dir / META_FILE
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def load_predictor(models_dir: Path = MODELS_DIR) -> tuple[BaseModel, dict] | None:
    """加载 predictor: 优先注册表 meta; 无 meta 的旧 pool_model.txt 回退单 LightGBM.

    返回 (predictor, meta); meta["skipped_members"] 列出加载失败/缺库而被剔除的
    成员(集成自动退化为剩余成员, 不 crash)。无任何可用模型返回 None。
    """
    models_dir = Path(models_dir)
    meta_path = models_dir / META_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = None
        if isinstance(meta, dict) and meta.get("members"):
            members: list[BaseModel] = []
            skipped: list[str] = []
            for entry in meta["members"]:
                cls = _MODEL_CLASSES.get(entry.get("name", ""))
                if cls is None or entry.get("name") == "xgb" and not XGBOOST_AVAILABLE:
                    skipped.append(entry.get("name", "?"))
                    continue
                try:
                    m = cls()
                    m.load_artifact(models_dir / entry["file"])
                    members.append(m)
                except Exception:  # noqa: BLE001 - 单成员工件损坏降级剔除, 不阻塞
                    skipped.append(entry.get("name", "?"))
            if members:
                meta["skipped_members"] = skipped
                predictor = EnsembleModel(members) if len(members) > 1 else members[0]
                return predictor, meta
    # 旧版回退: 只有 pool_model.txt(LightGBM 原生), 无注册表元数据
    legacy = models_dir / LGBMModel.FILE
    if legacy.exists():
        m = LGBMModel()
        m.load_artifact(legacy)
        return m, {
            "meta_version": 0, "ensemble": False,
            "members": [{"name": "lgbm", "file": LGBMModel.FILE}],
            "feature_names": None, "train_date_range": None,
            "n_train_samples": None, "trained_at": None, "skipped_members": [],
        }
    return None

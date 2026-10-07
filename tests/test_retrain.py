"""自动重训 + 归档回滚单测 (Phase 7.2): 判定纯函数 + 文件操作, 全部离线.

run_retrain 全链路(真实回测)由容器端到端演练覆盖(见 docs/experiments.md);
此处覆盖: decide_replacement 全分支、archive/restore/list 文件语义、
save_predictor 落盘回测指标(backtest 块)与读取回环。
"""

import json

import numpy as np
import pandas as pd

from stock_forecast.model import registry, retrain


# ---------- decide_replacement ----------

def _metrics(da: float) -> dict:
    return {"direction_accuracy": da, "overall_accuracy": da,
            "fold_accuracy_std": 0.05, "n_samples": 100, "possible_leakage": False}


def test_decide_force_overrides_everything():
    adopt, reason = retrain.decide_replacement(
        {"backtest": _metrics(0.9)}, _metrics(0.1), tolerance=0.0, force=True)
    assert adopt and "force" in reason


def test_decide_reject_without_new_metrics():
    adopt, reason = retrain.decide_replacement({"backtest": _metrics(0.55)}, None)
    assert not adopt and "回测失败" in reason


def test_decide_reject_nan_new_metrics():
    adopt, _ = retrain.decide_replacement(
        {"backtest": _metrics(0.55)}, _metrics(float("nan")))
    assert not adopt


def test_decide_first_time_adopt():
    adopt, reason = retrain.decide_replacement(None, _metrics(0.5))
    assert adopt and "首次训练" in reason


def test_decide_old_without_benchmark_adopt():
    # Phase 7 之前的旧工件: meta 无 backtest 块 -> 默认采纳并说明
    adopt, reason = retrain.decide_replacement({"members": []}, _metrics(0.5))
    assert adopt and "无可对比" in reason
    adopt2, _ = retrain.decide_replacement({"backtest": None}, _metrics(0.5))
    assert adopt2


def test_decide_not_worse_adopt():
    adopt, reason = retrain.decide_replacement(
        {"backtest": _metrics(0.55)}, _metrics(0.55))
    assert adopt and "采纳" in reason
    adopt2, _ = retrain.decide_replacement(
        {"backtest": _metrics(0.55)}, _metrics(0.58))
    assert adopt2


def test_decide_worse_reject():
    adopt, reason = retrain.decide_replacement(
        {"backtest": _metrics(0.58)}, _metrics(0.55))
    assert not adopt and "拒绝" in reason


def test_decide_within_tolerance_adopt():
    adopt, _ = retrain.decide_replacement(
        {"backtest": _metrics(0.55)}, _metrics(0.545), tolerance=0.01)
    assert adopt
    adopt2, _ = retrain.decide_replacement(
        {"backtest": _metrics(0.55)}, _metrics(0.53), tolerance=0.01)
    assert not adopt2


# ---------- 归档 / 回滚文件操作 ----------

def _make_artifacts(models_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / registry.META_FILE).write_text(
        json.dumps({"meta_version": 1, "trained_at": "2026-10-01T00:00:00",
                    "members": [{"name": "lgbm", "file": registry.LGBMModel.FILE}]}),
        encoding="utf-8")
    (models_dir / registry.LGBMModel.FILE).write_text("lgbm-artifact", encoding="utf-8")
    (models_dir / "drift_baseline.json").write_text("{}", encoding="utf-8")


def test_archive_current_moves_artifacts(tmp_path):
    models = tmp_path / "models"
    archive = tmp_path / "models/archive"
    _make_artifacts(models)
    dest = retrain.archive_current(models, archive)
    assert dest is not None and dest.exists()
    # 现役位置清空(等待新代落盘), 归档含全部工件(基线随代际)
    assert not (models / registry.META_FILE).exists()
    assert not (models / registry.LGBMModel.FILE).exists()
    assert (dest / registry.META_FILE).read_text(encoding="utf-8")
    assert (dest / "drift_baseline.json").exists()


def test_archive_current_empty_returns_none(tmp_path):
    assert retrain.archive_current(tmp_path / "models", tmp_path / "archive") is None


def test_list_and_restore_latest(tmp_path):
    models = tmp_path / "models"
    archive = tmp_path / "models/archive"
    _make_artifacts(models)
    dest = retrain.archive_current(models, archive)
    (dest / registry.LGBMModel.FILE).write_text("old-gen", encoding="utf-8")

    archives = retrain.list_archives(archive)
    assert len(archives) == 1
    assert archives[0]["trained_at"] == "2026-10-01T00:00:00"

    src = retrain.restore_latest(models, archive)
    assert src == dest
    # 复制回现役(非破坏式: 归档保留), 内容为归档代际
    assert (models / registry.LGBMModel.FILE).read_text(encoding="utf-8") == "old-gen"
    assert (models / "drift_baseline.json").exists()
    assert (dest / registry.LGBMModel.FILE).exists()
    # 重复回滚幂等
    assert retrain.restore_latest(models, archive) == dest


def test_restore_latest_no_archive(tmp_path):
    assert retrain.restore_latest(tmp_path / "models", tmp_path / "archive") is None
    assert retrain.list_archives(tmp_path / "archive") == []


def test_list_archives_newest_first(tmp_path):
    archive = tmp_path / "archive"
    for name in ("20260101_000000", "20260901_000000", "20260501_000000"):
        (archive / name).mkdir(parents=True)
    names = [a["name"] for a in retrain.list_archives(archive)]
    assert names == ["20260901_000000", "20260501_000000", "20260101_000000"]


# ---------- save_predictor 携带回测指标 ----------

def _synthetic(n_days: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    informative = rng.normal(0, 1, n_days)
    X = pd.DataFrame(rng.normal(0, 1, (n_days, 3)), index=dates, columns=list("abc"))
    X["informative"] = informative
    y = pd.Series(
        np.select([informative > 0.5, informative < -0.5], [1, -1], default=0),
        index=dates,
    )
    return X, y


def test_save_predictor_persists_backtest_metrics(tmp_path):
    X, y = _synthetic()
    m = registry.LGBMModel()
    m.fit(X, y)
    summary = _metrics(0.61)
    registry.save_predictor(m, list(X.columns), train_dates=X.index,
                            n_samples=len(X), models_dir=tmp_path,
                            backtest_summary=summary)
    meta = json.loads((tmp_path / registry.META_FILE).read_text(encoding="utf-8"))
    assert meta["backtest"]["direction_accuracy"] == 0.61
    # 读取回环: load_predictor 的 meta 带回 backtest 块(重训对比基准)
    loaded = registry.load_predictor(tmp_path)
    assert loaded is not None
    assert loaded[1]["backtest"]["direction_accuracy"] == 0.61


def test_save_predictor_backtest_default_none(tmp_path):
    X, y = _synthetic(80)
    m = registry.LGBMModel()
    m.fit(X, y)
    registry.save_predictor(m, list(X.columns), models_dir=tmp_path)
    meta = json.loads((tmp_path / registry.META_FILE).read_text(encoding="utf-8"))
    assert meta["backtest"] is None  # 旧调用口径不变(向后兼容)

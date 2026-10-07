"""自动重训 + 归档回滚 (Phase 7 / 7.2, L8).

流程(`run_retrain`, cmd_backtest 的受控变体):
1. 票池数据 -> walk-forward 多模型回测(与 backtest --pool 完全同口径);
2. **替换判定** `decide_replacement`: 新模型回测方向准确率 >= 旧模型 meta
   中的 backtest 指标 - 容差(`retrain_tolerance`, 默认 0 = 不劣于即替换)
   才采纳; 旧工件无对比基准(Phase 7 之前训练) -> 默认采纳并记录;
3. 采纳: 先把现役工件(meta/lgbm/xgb/**漂移基线**)整体归档到
   `data/models/archive/{时间戳}/`, 再落盘新代 —— 基线随模型代际同进退;
4. 拒绝: 现役模型不动, 拒绝原因落 `data/alerts.csvl`(kind=retrain_rejected);
5. `rollback`: 恢复**最近一次归档**(复制回现役位置, 归档保留不销毁 ——
   非破坏式, 重复回滚幂等)。

已知口径局限(记录于 docs/experiments.md): 新旧回测的数据窗口不同(重训时
历史更长), 指标对比是近似参考而非严格同分割对比; 采纳/拒绝/回滚全部落
alerts.csvl 形成运维审计线。dry_run 只做回测与判定, 不训练最终模型不写任何工件。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import (
    DRIFT_BASELINE_FILE,
    MODELS_ARCHIVE_DIR,
    MODELS_DIR,
    get_settings,
)
from stock_forecast.model import backtest as bt
from stock_forecast.model import registry
from stock_forecast.monitor.drift import append_alert

console = Console()

# 归档 = 一个模型代际的全部工件(基线与 meta 同进退, 见模块 docstring)
ARTIFACT_NAMES = (
    registry.META_FILE,
    registry.LGBMModel.FILE,
    registry.XGBModel.FILE,
    Path(DRIFT_BASELINE_FILE).name,
)


# ---------------------------------------------------------------------------
# 纯函数: 替换判定
# ---------------------------------------------------------------------------

def decide_replacement(
    old_meta: dict | None,
    new_metrics: dict | None,
    tolerance: float = 0.0,
    force: bool = False,
) -> tuple[bool, str]:
    """返回 (是否采纳新模型, 中文原因). 纯函数, 全部分支可单测."""
    if force:
        return True, "--force 指定强制替换(跳过回测对比)"
    if new_metrics is None or new_metrics.get("direction_accuracy") is None:
        return False, "新模型回测失败或无有效方向准确率指标, 不替换"
    new_da = float(new_metrics["direction_accuracy"])
    if pd.isna(new_da):
        return False, "新模型方向准确率无效(NaN), 不替换"
    if old_meta is None:
        return True, "无现役模型(首次训练), 直接采纳"
    old_bt = old_meta.get("backtest")
    if not old_bt or old_bt.get("direction_accuracy") is None:
        return True, "旧模型无可对比回测基准(Phase 7 之前的工件), 默认采纳并记录"
    old_da = float(old_bt["direction_accuracy"])
    if new_da >= old_da - tolerance:
        return True, (f"新模型方向准确率 {new_da:.1%} >= 旧模型 {old_da:.1%} "
                      f"- 容差 {tolerance:.1%}, 采纳")
    return False, (f"新模型方向准确率 {new_da:.1%} < 旧模型 {old_da:.1%} "
                   f"- 容差 {tolerance:.1%}, 拒绝替换(现役模型保留)")


# ---------------------------------------------------------------------------
# 归档与回滚(文件操作)
# ---------------------------------------------------------------------------

def archive_current(
    models_dir: Path = MODELS_DIR, archive_root: Path = MODELS_ARCHIVE_DIR
) -> Path | None:
    """把现役工件移入 archive/{时间戳}/; 无任何现役工件 -> None."""
    models_dir, archive_root = Path(models_dir), Path(archive_root)
    existing = [models_dir / n for n in ARTIFACT_NAMES if (models_dir / n).exists()]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_root / stamp
    seq = 1
    while dest.exists():  # 同秒重训的极端情况: 追加序号
        dest = archive_root / f"{stamp}_{seq}"
        seq += 1
    dest.mkdir(parents=True)
    for f in existing:
        shutil.move(str(f), str(dest / f.name))
    return dest


def list_archives(archive_root: Path = MODELS_ARCHIVE_DIR) -> list[dict]:
    """归档清单(新在前): 目录名 / 训练时刻 / 文件数."""
    archive_root = Path(archive_root)
    if not archive_root.exists():
        return []
    out = []
    dirs = sorted((p for p in archive_root.iterdir() if p.is_dir()),
                  key=lambda p: p.name, reverse=True)
    for d in dirs:
        trained_at = None
        meta_path = d / registry.META_FILE
        if meta_path.exists():
            try:
                trained_at = json.loads(meta_path.read_text(encoding="utf-8")).get("trained_at")
            except (json.JSONDecodeError, OSError):
                pass
        out.append({"name": d.name, "trained_at": trained_at,
                    "n_files": sum(1 for _ in d.iterdir())})
    return out


def restore_latest(
    models_dir: Path = MODELS_DIR, archive_root: Path = MODELS_ARCHIVE_DIR
) -> Path | None:
    """恢复最近一次归档到现役位置(复制, 归档保留); 无归档 -> None."""
    archives = list_archives(archive_root)
    if not archives:
        return None
    src = Path(archive_root) / archives[0]["name"]
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(models_dir / f.name))
    return src


# ---------------------------------------------------------------------------
# 重训主流程
# ---------------------------------------------------------------------------

def run_retrain(
    dry_run: bool = False,
    force: bool = False,
    tolerance: float | None = None,
    pool: list[str] | None = None,
    settings=None,
) -> dict:
    """重训编排. 返回 {ok, error?} | {ok, adopted, reason, old_metrics,
    new_metrics, dry_run, archive_dir?}; 告警落盘在本函数内完成."""
    settings = settings or get_settings()
    tol = settings.retrain_tolerance if tolerance is None else tolerance
    pool = pool or settings.stock_pool

    console.print(f"重训回测进行中(票池 {len(pool)} 只, walk-forward 多模型, 与 backtest --pool 同口径)…")
    try:
        X, y = bt.build_pool_dataset(pool)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    Xn = X.drop(columns=["symbol"])

    out = bt.run_multimodel_backtest(Xn, y, settings.train_ratio, settings.test_window)
    results, failed = out["results"], out["failed"]
    if registry.LGBMModel.name not in results:
        return {"ok": False, "error": "LightGBM 基线模型回测失败, 无法重训"}

    # 判定口径: 集成优先(与最终落盘 predictor 的构成一致, 见 bt.final_metrics)
    key = "ensemble" if "ensemble" in results else registry.LGBMModel.name
    new_metrics = bt.pack_summary(results[key][1])

    loaded = registry.load_predictor()
    old_meta = loaded[1] if loaded else None
    old_metrics = (old_meta or {}).get("backtest")

    adopt, reason = decide_replacement(old_meta, new_metrics, tol, force)
    base = {"ok": True, "adopted": adopt, "reason": reason,
            "old_metrics": old_metrics, "new_metrics": new_metrics,
            "new_members": key, "tolerance": tol}

    if dry_run:
        return {**base, "dry_run": True}
    if not adopt:
        append_alert("retrain_rejected", "warn", reason=reason,
                     old=old_metrics, new=new_metrics)
        return {**base, "dry_run": False}

    members = bt.fit_final_members(Xn, y, failed)
    if not members:
        return {"ok": False, "error": "全部模型最终训练失败, 现役模型保留未动"}
    predictor = registry.EnsembleModel(members) if len(members) > 1 else members[0]

    archive_dir = archive_current()  # 先归档现役代际(含漂移基线), 再落盘新代
    bt.persist_final(predictor, Xn, results)
    append_alert("retrain_adopted", "info", reason=reason, members=predictor.name,
                 old=old_metrics, new=new_metrics,
                 archive=str(archive_dir) if archive_dir else None)
    return {**base, "dry_run": False, "members": predictor.name,
            "archive_dir": str(archive_dir) if archive_dir else None}


def _metrics_table(result: dict) -> None:
    t = Table(title="重训对比(方向准确率为主指标)")
    t.add_column("")
    t.add_column("旧模型(现役)", justify="right")
    t.add_column("新模型(本次回测)", justify="right")
    old, new = result.get("old_metrics"), result.get("new_metrics")

    def _fmt(m, k, pct=True):
        if not m or m.get(k) is None:
            return "-"
        return f"{m[k]:.1%}" if pct else str(m[k])

    t.add_row("方向准确率", _fmt(old, "direction_accuracy"), _fmt(new, "direction_accuracy"))
    t.add_row("整体准确率", _fmt(old, "overall_accuracy"), _fmt(new, "overall_accuracy"))
    t.add_row("逐轮std", _fmt(old, "fold_accuracy_std"), _fmt(new, "fold_accuracy_std"))
    t.add_row("样本数", _fmt(old, "n_samples", False), _fmt(new, "n_samples", False))
    console.print(t)


def cmd_retrain(args: argparse.Namespace) -> int:
    result = run_retrain(dry_run=args.dry_run, force=args.force,
                         tolerance=args.tolerance)
    if not result["ok"]:
        console.print(f"[red]重训失败: {result['error']}[/red]")
        return 1
    _metrics_table(result)
    if result["dry_run"]:
        console.print(f"[yellow]演练模式(--dry-run): 判定为 "
                      f"{'采纳' if result['adopted'] else '拒绝'} — {result['reason']}; "
                      "未训练最终模型, 未写任何工件.[/yellow]")
        return 0
    if result["adopted"]:
        console.print(f"[green]新模型已替换现役: 成员={result.get('members')}, "
                      f"原因: {result['reason']}[/green]")
        if result.get("archive_dir"):
            console.print(f"[dim]旧代际已归档: {result['archive_dir']} "
                          f"(回滚: rollback)[/dim]")
    else:
        console.print(f"[yellow]保留现役模型: {result['reason']} "
                      "(已记入 alerts.csvl)[/yellow]")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    if args.list:
        archives = list_archives()
        if not archives:
            console.print("暂无归档(尚未发生过重训替换)。")
            return 0
        t = Table(title=f"模型归档({MODELS_ARCHIVE_DIR})")
        t.add_column("归档")
        t.add_column("训练时刻")
        t.add_column("文件数", justify="right")
        for a in archives:
            t.add_row(a["name"], str(a["trained_at"] or "-"), str(a["n_files"]))
        console.print(t)
        return 0

    src = restore_latest()
    if src is None:
        console.print("[red]无可回滚的归档.[/red]")
        return 1
    append_alert("rollback", "info", archive=src.name)
    console.print(f"[green]已从归档 {src.name} 恢复现役模型"
                  "(归档保留, 重复回滚幂等; 漂移基线一并恢复).[/green]")
    return 0

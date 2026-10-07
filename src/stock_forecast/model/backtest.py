"""walk-forward 回测: expanding window, 严禁随机分割.

流程:
1. 汇总票池各股 (X, y), 按日期全局排序;
2. 初始训练窗 = 前 train_ratio 个交易日, 之后每轮预测 test_window 个交易日,
   预测完把这些日并入训练窗, 向前滚动;
3. 每轮训练内部, 早停验证集取该轮训练窗尾部(时间有序);
4. 汇总方向准确率(剔除真实"平")/整体准确率/分类别召回/逐轮稳定性;
5. >70% 直接打出泄露警告.

回测结束后用全量数据训练最终模型并落盘, 供 predict 使用.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import MODELS_DIR, get_settings
from stock_forecast.data import collector
from stock_forecast.features import regime
from stock_forecast.features.build import make_dataset
from stock_forecast.model import metrics, registry, train

console = Console()


def build_pool_dataset(pool: list[str], refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """汇总票池各股样本; 单只失败不阻塞整体."""
    Xs, ys = [], []
    for sym in pool:
        try:
            X, y = make_dataset(sym, refresh=refresh)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]{sym} 数据不可用, 跳过: {e}[/yellow]")
            continue
        Xs.append(X.assign(symbol=sym))
        ys.append(y)
    if not Xs:
        raise RuntimeError("票池内没有任何可用数据")
    X = pd.concat(Xs).sort_index()
    y = pd.concat(ys).sort_index()
    return X, y


def walk_forward_folds(
    dates: pd.DatetimeIndex, train_ratio: float, test_window: int
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """返回 [(train_dates, test_dates), ...], expanding window."""
    uniq = dates.unique().sort_values()
    n_train = max(30, int(len(uniq) * train_ratio))
    folds = []
    cursor = n_train
    while cursor < len(uniq):
        test_dates = uniq[cursor : cursor + test_window]
        folds.append((uniq[:cursor], test_dates))
        cursor += len(test_dates)
    return folds


def run_backtest(
    X: pd.DataFrame, y: pd.Series, train_ratio: float, test_window: int, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """执行回测, 返回 (逐样本预测明细, 汇总指标)."""
    dates = pd.DatetimeIndex(X.index)
    folds = walk_forward_folds(dates, train_ratio, test_window)
    if not folds:
        raise RuntimeError("样本时间跨度不足以构建回测分割")

    records = []
    for i, (train_dates, test_dates) in enumerate(folds):
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        model = train.train_model(X[train_mask], y[train_mask], seed=seed)
        probs = train.predict_proba_df(model.booster_, X[test_mask])
        preds = probs.values.argmax(axis=1)
        fold_df = pd.DataFrame({
            "date": X.index[test_mask],
            "symbol": X["symbol"].values[test_mask] if "symbol" in X.columns else "",
            "fold": i,
            "p_down": probs["p_down"].values,
            "p_flat": probs["p_flat"].values,
            "p_up": probs["p_up"].values,
            "y_true": y.values[test_mask],
            "y_pred": preds,
        })
        records.append(fold_df)

    detail = pd.concat(records, ignore_index=True)
    detail["y_pred"] = detail["y_pred"].map(train.INV_LABEL_MAP)
    summary = metrics.summarize(detail)
    return detail, summary


def _detail_frame(
    X: pd.DataFrame, y: pd.Series, test_mask, fold_i: int, probs: pd.DataFrame
) -> pd.DataFrame:
    """一轮测试窗的逐样本预测明细(与 run_backtest 的 fold_df 同构)."""
    preds = probs.to_numpy().argmax(axis=1)
    return pd.DataFrame({
        "date": X.index[test_mask],
        "symbol": X["symbol"].values[test_mask] if "symbol" in X.columns else "",
        "fold": fold_i,
        "p_down": probs["p_down"].to_numpy(),
        "p_flat": probs["p_flat"].to_numpy(),
        "p_up": probs["p_up"].to_numpy(),
        "y_true": y.values[test_mask],
        "y_pred": pd.Series(preds).map(train.INV_LABEL_MAP).to_numpy(),
    })


def run_multimodel_backtest(
    X: pd.DataFrame, y: pd.Series, train_ratio: float, test_window: int, seed: int = 42
) -> dict:
    """多模型 walk-forward(4.2): 同一分割下各模型独立训练/预测, 集成=等权概率平均(4.3).

    返回 {"results": {name: (detail, summary)}, "comparison": {...}, "failed": {name: err}}。
    results 含各成员模型, 成员 ≥2 时另含 "ensemble"(等权平均后再 argmax)。
    某模型任一轮训练失败 -> 整体剔除该模型(保证同样本公平对比), 不阻塞其余模型
    (不变量 5); 每个 summary 独立带 possible_leakage 泄露报警(验收: 逐模型检查)。
    """
    dates = pd.DatetimeIndex(X.index)
    folds = walk_forward_folds(dates, train_ratio, test_window)
    if not folds:
        raise RuntimeError("样本时间跨度不足以构建回测分割")

    classes = registry.available_models()
    probs_by_fold: list[dict[str, pd.DataFrame]] = []
    failed: dict[str, str] = {}
    for i, (train_dates, test_dates) in enumerate(folds):
        train_mask, test_mask = dates.isin(train_dates), dates.isin(test_dates)
        fold_probs: dict[str, pd.DataFrame] = {}
        for cls in classes:
            if cls.name in failed:
                continue
            try:
                m = cls(seed=seed)
                m.fit(X[train_mask], y[train_mask])
                fold_probs[cls.name] = m.predict_proba(X[test_mask])
            except Exception as e:  # noqa: BLE001 - 单模型失败剔除, 不阻塞其余
                failed[cls.name] = f"{type(e).__name__}: {e}"
                console.print(f"[yellow]模型 {cls.name} 第 {i} 轮训练失败, 剔除出对比: {e}[/yellow]")
        probs_by_fold.append(fold_probs)

    valid = [
        cls.name for cls in classes
        if cls.name not in failed and all(cls.name in fp for fp in probs_by_fold)
    ]
    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    for name in valid:
        records = [
            _detail_frame(X, y, dates.isin(test_dates), i, fp[name])
            for i, ((_train_dates, test_dates), fp) in enumerate(zip(folds, probs_by_fold))
        ]
        detail = pd.concat(records, ignore_index=True)
        results[name] = (detail, metrics.summarize(detail))

    if len(valid) >= 2:  # 4.3: 等权概率平均集成, 与成员同一分割同一样本
        records = []
        for i, ((_train_dates, test_dates), fp) in enumerate(zip(folds, probs_by_fold)):
            mean_probs = sum(fp[n] for n in valid) / len(valid)
            records.append(_detail_frame(X, y, dates.isin(test_dates), i, mean_probs))
        ens = pd.concat(records, ignore_index=True)
        results["ensemble"] = (ens, metrics.summarize(ens))

    comparison = _model_comparison(results) if len(results) >= 2 else {}
    return {"results": results, "comparison": comparison, "failed": failed}


def _model_comparison(results: dict) -> dict:
    """成员间 p_up 皮尔逊相关矩阵 + argmax 方向一致率(4.2 相关性分析).

    各 detail 由同一 fold 序列按同一测试窗构造, 行序天然对齐, 可直接按位比较。
    """
    names = list(results)
    p_up = {n: results[n][0]["p_up"].to_numpy() for n in names}
    y_pred = {n: results[n][0]["y_pred"].to_numpy() for n in names}
    corr: dict[str, dict[str, float]] = {}
    agree: dict[str, dict[str, float]] = {}
    for a in names:
        corr[a], agree[a] = {}, {}
        for b in names:
            if a == b:
                corr[a][b], agree[a][b] = 1.0, 1.0
                continue
            c = float(np.corrcoef(p_up[a], p_up[b])[0, 1])
            corr[a][b] = c
            agree[a][b] = float((y_pred[a] == y_pred[b]).mean())
    return {"p_up_corr": corr, "dir_agreement": agree}


def _print_model_comparison(results: dict, comparison: dict, failed: dict) -> None:
    t = Table(title="多模型对比 (walk-forward 同一分割)")
    t.add_column("模型")
    t.add_column("样本", justify="right")
    t.add_column("方向准确率")
    t.add_column("整体准确率")
    t.add_column("逐轮std")
    t.add_column("泄露报警")
    for name, (_detail, s) in results.items():
        std = s.get("fold_accuracy_std")
        t.add_row(
            name, str(s["n_samples"]),
            f"{s['direction_accuracy']:.1%}", f"{s['overall_accuracy']:.1%}",
            f"{std:.1%}" if std is not None and pd.notna(std) else "-",
            "[red]⚠[/red]" if s["possible_leakage"] else "-",
        )
    console.print(t)
    for name, err in failed.items():
        console.print(f"[yellow]模型 {name} 训练失败, 已排除出对比与集成: {err}[/yellow]")

    if comparison:
        names = list(results)
        c = Table(title="p_up 相关 / 方向一致率")
        c.add_column("")
        for n in names:
            c.add_column(n)
        for a in names:
            row = [a]
            for b in names:
                row.append(
                    f"{comparison['p_up_corr'][a][b]:.2f} / "
                    f"{comparison['dir_agreement'][a][b]:.0%}"
                )
            c.add_row(*row)
        console.print(c)


def regime_bucket_report(
    detail: pd.DataFrame, index: pd.DataFrame | None, settings=None
) -> dict | None:
    """回测明细按市场状态分桶的方向准确率(Phase 3 验收报表).

    状态取 shifted_regimes(行 T = T-1 收盘状态, 与 X[T] 信息时点严格对齐,
    否则状态会泄露当日收盘信息); detail 日期未被指数覆盖或无前日 -> "未知"桶。
    index 不可用返回 None(调用方降级提示, 不阻塞回测主流程)。
    """
    if index is None or len(index) == 0:
        return None
    shifted = regime.shifted_regimes(index, settings)
    states = shifted.reindex(pd.DatetimeIndex(detail["date"])).fillna(regime.REGIME_UNKNOWN)
    d = detail.assign(_regime=states.to_numpy())
    out: dict = {}
    for state, g in d.groupby("_regime"):
        da = metrics.direction_accuracy(g["y_true"], g["y_pred"])
        out[str(state)] = {
            "n": len(g),
            "direction_accuracy": None if pd.isna(da) else float(da),
            "overall_accuracy": metrics.overall_accuracy(g["y_true"], g["y_pred"]),
        }
    return out


def _print_regime_buckets(buckets: dict | None) -> None:
    if buckets is None:
        console.print("[yellow]指数数据不可用, 跳过市场状态分桶报表.[/yellow]")
        return
    t = Table(title="按市场状态分桶(状态取预测信息时点 T-1)")
    t.add_column("状态")
    t.add_column("样本", justify="right")
    t.add_column("方向准确率")
    t.add_column("整体准确率")
    for state in ("牛市", "熊市", "震荡", "未知"):
        b = buckets.get(state)
        if not b:
            continue
        da = "-" if b["direction_accuracy"] is None else f"{b['direction_accuracy']:.1%}"
        t.add_row(state, str(b["n"]), da, f"{b['overall_accuracy']:.1%}")
    console.print(t)


def _print_summary(summary: dict) -> None:
    table = Table(title="回测结果")
    table.add_column("指标")
    table.add_column("值")
    table.add_row("样本数", str(summary["n_samples"]))
    table.add_row("整体三分类准确率", f"{summary['overall_accuracy']:.1%}")
    table.add_row("[bold]方向准确率(剔除'平')[/bold]", f"{summary['direction_accuracy']:.1%}")
    for lab, name in ((1, "涨"), (0, "平"), (-1, "跌")):
        r = summary["class_recall"][lab]
        table.add_row(f"召回-{name}", f"{r:.1%}" if pd.notna(r) else "-")
    if "per_fold_direction_accuracy" in summary:
        folds = summary["per_fold_direction_accuracy"]
        table.add_row("逐轮方向准确率", " ".join(f"{a:.0%}" for a in folds))
        table.add_row("逐轮准确率std", f"{summary['fold_accuracy_std']:.1%}")
    console.print(table)

    if summary["possible_leakage"]:
        console.print(
            "[bold red]⚠ 方向准确率 > 70%, 大概率存在过拟合或数据泄露, "
            "请排查特征/标签对齐后再采信![/bold red]"
        )
    else:
        console.print(
            "[dim]参考: 日频方向预测 52%~65% 即属可信水平; 本系统价值在于"
            "不确定性量化与可解释性, 不追求高准确率.[/dim]"
        )


def fit_final_members(
    Xn: pd.DataFrame, y: pd.Series, failed: dict | None = None, seed: int = 42
) -> list:
    """用全量数据训练最终落盘成员(回测中已失败的模型直接跳过); 单成员失败剔除."""
    members = []
    for cls in registry.available_models():
        if failed and cls.name in failed:
            continue
        m = cls(seed=seed)
        try:
            m.fit(Xn, y)
            members.append(m)
        except Exception as e:  # noqa: BLE001 - 单成员最终训练失败降级剔除
            console.print(f"[yellow]模型 {cls.name} 最终训练失败, 排除出落盘: {e}[/yellow]")
    return members


def pack_summary(s: dict) -> dict:
    """回测 summary -> 可 JSON 落盘的精简指标(retrain 对比基准的统一口径)."""
    std = s.get("fold_accuracy_std")
    return {
        "direction_accuracy": s.get("direction_accuracy"),
        "overall_accuracy": s.get("overall_accuracy"),
        "fold_accuracy_std": None if std is None or pd.isna(std) else float(std),
        "n_samples": s.get("n_samples"),
        "possible_leakage": bool(s.get("possible_leakage")),
    }


def final_metrics(predictor, results: dict) -> dict | None:
    """从回测 results 中取与最终 predictor 同口径的指标(集成 -> ensemble 行).

    Phase 7.2 自动重训的"不劣于旧模型"对比基准, 随 pool_meta.json 落盘。
    """
    key = "ensemble" if isinstance(predictor, registry.EnsembleModel) else predictor.name
    if key not in results:
        return None
    return pack_summary(results[key][1])


def persist_final(predictor, Xn: pd.DataFrame, results: dict | None = None) -> None:
    """落盘最终模型工件 + meta(含回测指标) + **PSI 漂移基线**(Phase 7.1).

    cmd_backtest 与 retrain 共用同一落盘路径, 保证基线与模型代际一致
    (retrain 替换/回滚时基线随工件一起归档/恢复)。
    """
    from stock_forecast.monitor import drift

    registry.save_predictor(
        predictor, feature_names=list(Xn.columns), train_dates=Xn.index,
        n_samples=len(Xn),
        backtest_summary=final_metrics(predictor, results) if results else None,
    )
    drift.save_baseline(drift.build_baseline(Xn))


def cmd_backtest(args: argparse.Namespace) -> int:
    settings = get_settings()
    pool = [args.symbol] if args.symbol else settings.stock_pool
    try:
        X, y = build_pool_dataset(pool)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    console.print(f"票池样本: {X.shape[0]} 行 × {X.shape[1]} 列, "
                  f"时间 {X.index.min().date()} ~ {X.index.max().date()}")
    if args.symbol:
        console.print("[yellow]单票视图样本极少, 结果仅作管道验证参考.[/yellow]")

    Xn = X.drop(columns=["symbol"])
    out = run_multimodel_backtest(Xn, y, settings.train_ratio, settings.test_window)
    results, comparison, failed = out["results"], out["comparison"], out["failed"]
    if registry.LGBMModel.name not in results:
        console.print("[red]LightGBM 基线模型回测失败, 无法继续.[/red]")
        return 1

    # 主表 = LightGBM 基线(与历史回测口径一致, 便于跨 Phase 对比)
    detail, summary = results[registry.LGBMModel.name]
    _print_summary(summary)
    if len(results) > 1 or failed:
        _print_model_comparison(results, comparison, failed)

    # Phase 3 (L4): 按市场状态分桶的方向准确率; 指数数据不可用时降级跳过
    try:
        index_df = collector.fetch_index()
    except Exception as e:  # noqa: BLE001 - 分桶报表属附加输出, 失败不阻塞主流程
        console.print(f"[yellow]指数数据获取失败, 跳过市场状态分桶: {e}[/yellow]")
        index_df = None
    _print_regime_buckets(regime_bucket_report(detail, index_df, settings))

    # 最终模型落盘供 predict 使用(4.1: 注册表工件 + pool_meta.json 元数据;
    # 成员 ≥2 时为等权集成, 单成员失败剔除不阻塞; Phase 7: 附回测指标+漂移基线)
    members = fit_final_members(Xn, y, failed)
    if not members:
        console.print("[red]全部模型最终训练失败, 未落盘任何模型.[/red]")
        return 1
    predictor = registry.EnsembleModel(members) if len(members) > 1 else members[0]
    persist_final(predictor, Xn, results)
    console.print(
        f"[green]最终模型已保存: 成员={predictor.name}, "
        f"工件+元数据 {registry.META_FILE}+漂移基线 位于 {MODELS_DIR}[/green]"
    )
    return 0

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

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import get_settings
from stock_forecast.features.build import make_dataset
from stock_forecast.model import metrics, train

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

    detail, summary = run_backtest(
        X.drop(columns=["symbol"]), y,
        settings.train_ratio, settings.test_window,
    )
    _print_summary(summary)

    # 训练最终模型落盘供 predict 使用
    model = train.train_final_model(X.drop(columns=["symbol"]), y)
    console.print(f"[green]最终模型已保存: {train.MODEL_PATH}[/green]")
    return 0

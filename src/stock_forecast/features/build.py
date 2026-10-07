"""特征矩阵构建: 多表按交易日对齐 + 统一 shift(1) 防前视.

时间语义约定:
- 特征原始帧第 T 行 = "截至第 T 日收盘可得的信息"(含 T 日行情);
- build 统一对全部特征列 shift(1) → X[T] 只含 ≤ T-1 收盘后信息;
- 标签 y[T] = 第 T 日 close-to-close 方向;
- 因此 (X[T], y[T]) 表示"盘前用 T-1 数据预测 T 日走势", 严格无前视.

注意: 交易时段内运行时 akshare 日K会含当日未收盘行, MVP 假定在
收盘后或次日开盘前运行; 若盘中运行, 最后一行特征含义会有偏差.
"""

from __future__ import annotations

import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import get_settings
from stock_forecast.data import collector
from stock_forecast.features import indicators, labeling, sources

console = Console()

# Phase 2 扩展数据源在 fetch_all 结果中的键(全部可选, 缺失 -> 特征列全 NaN)
EXTRA_KEYS = ("industry", "north", "margin", "lhb", "holder", "unlock", "global", "macro")


def make_extras(data: dict) -> dict:
    """从 fetch_all 结果中提取扩展数据源字典(供 build_feature_matrix 的 extras 参数)."""
    return {k: data.get(k) for k in EXTRA_KEYS}


def _index_features(index: pd.DataFrame) -> pd.DataFrame:
    """市场(上证指数)特征: 与个股一样先按 T 行计算, 交给上层统一 shift."""
    c = index["close"]
    f = pd.DataFrame(index=index.index)
    for n in (1, 5, 20):
        f[f"idx_ret{n}"] = c.pct_change(n)
    f["idx_vol20"] = c.pct_change().rolling(20).std()
    return f


def _fund_flow_features(ff: pd.DataFrame) -> pd.DataFrame:
    """资金流特征: *_net_pct 列本身无量纲; 再加主力净流入占比的短均线."""
    f = pd.DataFrame(index=ff.index)
    for col in ("main_net_pct", "super_net_pct", "big_net_pct"):
        if col in ff.columns:
            f[col] = ff[col]
    if "main_net_pct" in ff.columns:
        f["main_net_pct_sum3"] = f["main_net_pct"].rolling(3).sum()
        f["main_net_pct_sum5"] = f["main_net_pct"].rolling(5).sum()
    return f


def build_feature_matrix(
    kline: pd.DataFrame,
    index: pd.DataFrame,
    fund_flow: pd.DataFrame | None,
    threshold: float,
    extras: dict[str, pd.DataFrame | None] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """纯函数: 输入原始帧, 输出 (X, y, optional_cols), 按 kline 的 date 索引对齐.

    X 所有列已 shift(1); optional_cols 为资金流 + Phase 2 扩展源列(历史覆盖期短
    或允许缺失, NaN 保留给 LightGBM 原生处理); 其余列暖机期 NaN 由调用方 dropna.

    extras: EXTRA_KEYS 中各扩展源帧(可为 None), 特征函数见 features/sources.py,
    全部输出恒定列集合(源缺失 -> 全 NaN 列), 保证跨股票 schema 稳定。
    """
    kline = kline.reset_index(drop=True)
    f = indicators.compute_all(kline)

    idx = _index_features(index.reset_index(drop=True))
    idx_aligned = pd.DataFrame(index=kline.index)
    idx_dates = pd.to_datetime(index["date"]).reset_index(drop=True)
    for col in idx.columns:
        s = idx[col].set_axis(idx_dates)
        idx_aligned[col] = s.reindex(pd.to_datetime(kline["date"])).values

    f = pd.concat([f, idx_aligned], axis=1)

    optional_cols: list[str] = []
    if fund_flow is not None and not fund_flow.empty:
        ff = _fund_flow_features(fund_flow.reset_index(drop=True))
        ff_aligned = pd.DataFrame(index=kline.index)
        ff_dates = pd.to_datetime(fund_flow["date"]).reset_index(drop=True)
        for col in ff.columns:
            s = ff[col].set_axis(ff_dates)
            ff_aligned[col] = s.reindex(pd.to_datetime(kline["date"])).values
        f = pd.concat([f, ff_aligned], axis=1)
        optional_cols = list(ff.columns)

    # Phase 2 扩展源: 全部进 optional_cols(不进 dropna, NaN 由 LightGBM 处理)。
    # sources.py 保证"第 T 行 = 截至 T 日收盘可得信息", 时间平移仍只由下方 shift 完成。
    extras = extras or {}
    ext_frames = [
        sources.industry_features(extras.get("industry"), kline),
        sources.north_features(extras.get("north"), kline, index),
        sources.margin_features(extras.get("margin"), kline),
        sources.event_decay_features(extras.get("lhb"), kline,
                                     value_col="net_buy", prefix="lhb", scale=1e-8),
        sources.event_decay_features(extras.get("holder"), kline,
                                     value_col="amount", prefix="holder", scale=1e-6),
        sources.event_decay_features(extras.get("unlock"), kline,
                                     value_col="ratio", prefix="unlock"),
        sources.global_features(extras.get("global"), kline),
        sources.macro_features(extras.get("macro"), kline),
    ]
    for ef in ext_frames:
        f = pd.concat([f, ef], axis=1)
        optional_cols += list(ef.columns)

    # 核心防前视: 全部特征统一后移一天
    X = f.shift(1)

    y = labeling.make_labels(kline["close"], threshold).reset_index(drop=True)
    X.index = pd.to_datetime(kline["date"]).values
    y.index = X.index
    return X, y, optional_cols


def make_dataset(symbol: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """拉取(或读缓存)数据并构建该股 (X, y), 已 dropna 暖机期.

    资金流列(optional)不参与 dropna 判定, NaN 保留给模型.
    """
    settings = get_settings()
    data = collector.fetch_all(symbol, refresh=refresh)
    kline, index, ff = data["kline"], data["index"], data["fund_flow"]
    if kline is None or kline.empty:
        raise RuntimeError(f"{symbol} K线数据不可用, 请先 fetch")
    X, y, optional_cols = build_feature_matrix(
        kline, index, ff, settings.label_threshold, extras=make_extras(data)
    )
    core_cols = [c for c in X.columns if c not in optional_cols]
    mask = X[core_cols].notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def cmd_features(args: argparse.Namespace) -> int:
    try:
        X, y = make_dataset(args.symbol)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    table = Table(title=f"features {args.symbol}")
    table.add_column("指标")
    table.add_column("值")
    table.add_row("矩阵 shape", f"{X.shape[0]} 行 × {X.shape[1]} 列")
    dist = y.value_counts(normalize=True).sort_index()
    for lab, ratio in dist.items():
        name = labeling.LABEL_NAMES.get(int(lab), str(lab))
        table.add_row(f"标签 {name}", f"{ratio:.1%}")
    nan_cols = int(X.isna().any(axis=1).sum())
    table.add_row("残留 NaN 行(已丢弃口径外)", str(nan_cols))
    console.print(table)
    console.print(f"特征列: {', '.join(X.columns)}")
    return 0

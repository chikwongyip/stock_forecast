"""组合采集器: fetch 子命令的实现.

策略: 缓存未覆盖最近交易日才访问网络; 任一数据源失败不阻塞其他源,
失败源明确报告(LLM/特征层按缺失处理).
"""

from __future__ import annotations

import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.data import akshare_client as akc
from stock_forecast.data import cache

console = Console()

INDEX_SYMBOL = "000001"  # 上证指数


def fetch_kline(symbol: str, refresh: bool = False) -> pd.DataFrame:
    key = f"kline_{symbol}"
    if not refresh and cache.is_fresh(key):
        return cache.load(key)  # type: ignore[return-value]
    df = akc.get_kline(symbol)
    cache.save(key, df)
    return df


def fetch_index(refresh: bool = False) -> pd.DataFrame:
    key = f"index_{INDEX_SYMBOL}"
    if not refresh and cache.is_fresh(key):
        return cache.load(key)  # type: ignore[return-value]
    df = akc.get_index_kline(INDEX_SYMBOL)
    cache.save(key, df)
    return df


def fetch_fund_flow(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """资金流向: 部分新股/次新股无数据, 允许为空."""
    key = f"fundflow_{symbol}"
    if not refresh and cache.is_fresh(key):
        cached = cache.load(key)
        if cached is not None:
            return cached
    try:
        df = akc.get_fund_flow(symbol)
    except akc.AkshareError as e:
        console.print(f"[yellow]资金流向获取失败: {e}[/yellow]")
        return None
    if df.empty:
        return None
    cache.save(key, df)
    return df


def fetch_all(symbol: str, refresh: bool = False) -> dict[str, pd.DataFrame | None]:
    """拉取个股全部 MVP 数据. 失败的数据源置 None."""
    out: dict[str, pd.DataFrame | None] = {}
    for name, fn in (
        ("kline", lambda: fetch_kline(symbol, refresh)),
        ("index", lambda: fetch_index(refresh)),
        ("fund_flow", lambda: fetch_fund_flow(symbol, refresh)),
    ):
        try:
            out[name] = fn()
        except akc.AkshareError as e:
            console.print(f"[red]{name} 获取失败: {e}[/red]")
            out[name] = None
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    data = fetch_all(args.symbol, refresh=args.refresh)
    kline = data.get("kline")

    table = Table(title=f"fetch {args.symbol} 结果")
    table.add_column("数据源")
    table.add_column("行数", justify="right")
    table.add_column("最新日期")
    for name, df in data.items():
        if df is None or len(df) == 0:
            table.add_row(name, "0", "-")
        else:
            table.add_row(name, str(len(df)), str(df["date"].max().date()))
    console.print(table)

    if kline is None or kline.empty:
        console.print("[red]核心K线数据获取失败, 无法继续.[/red]")
        return 1
    console.print(
        f"[green]缓存写入 {cache._CACHE_DIR}[/green] "
        f"(kline {len(kline)} 行, 覆盖 {kline['date'].max().date()})"
    )
    return 0

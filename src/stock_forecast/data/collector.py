"""组合采集器: fetch 子命令的实现.

策略: 缓存未覆盖最近交易日才访问网络; 任一数据源失败不阻塞其他源,
失败源明确报告(LLM/特征层按缺失处理).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.data import akshare_client as akc
from stock_forecast.data import cache

console = Console()

INDEX_SYMBOL = "000001"  # 上证指数

# 各数据源必须具备的最小列集合(健康检查用; amount/pct_change 等允许版本差异)
EXPECTED_COLUMNS = {
    "kline": ["date", "open", "close", "high", "low", "volume"],
    "index": ["date", "open", "close", "high", "low", "volume"],
    "fund_flow": ["date", "main_net_pct"],
}


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
    # 取 2 年历史: 市场状态识别(Phase 3)需 MA60 + 250 日波动率分位暖机;
    # 旧 365 天缓存仍可用(分位样本不足时自动不触发高波动覆盖), 下次过期自愈
    df = akc.get_index_kline(INDEX_SYMBOL, days=730)
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


# ---------------------------------------------------------------------------
# Phase 2 扩展数据源: 全部可选, 任何失败降级为 None(特征层自动 NaN)。
# 缓存策略: 当日即有数据的源用 is_fresh(按最新数据日期); T+1 公布/全市场
# 共享/事件日历/月频源用 saved_within(按文件写入时间), 避免永远"不新鲜"反复拉网。
# ---------------------------------------------------------------------------


def fetch_industry(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """个股所属行业板块的日K线. 行业名与板块行情分开缓存(板块行情同行业共享)."""
    name_key = f"industry_{symbol}"
    name: str | None = None
    if not refresh and cache.saved_within(name_key, days=30):
        cached = cache.load_json(name_key)
        if isinstance(cached, str) and cached:
            name = cached
    if name is None:
        try:
            name = akc.get_stock_industry(symbol)
        except akc.AkshareError as e:
            console.print(f"[yellow]行业归属获取失败: {e}[/yellow]")
            return None
        if not name:
            return None
        cache.save_json(name_key, name)
    hist_key = f"industryhist_{name}"
    if not refresh and cache.is_fresh(hist_key):
        return cache.load(hist_key)
    try:
        df = akc.get_industry_hist(name)
    except akc.AkshareError as e:
        console.print(f"[yellow]行业板块行情获取失败({name}): {e}[/yellow]")
        return None
    if df is None or df.empty:
        return None
    cache.save(hist_key, df)
    return df


def fetch_stock_name(symbol: str, refresh: bool = False) -> str | None:
    """股票简称(ST 识别用, Phase 5 / 5.1). 缓存 30 天; 任何失败降级 None(跳过 ST 检查)."""
    key = f"stockname_{symbol}"
    if not refresh and cache.saved_within(key, days=30):
        cached = cache.load_json(key)
        if isinstance(cached, str) and cached:
            return cached
    try:
        name = akc.get_stock_name(symbol)
    except akc.AkshareError as e:
        console.print(f"[yellow]股票简称获取失败(跳过 ST 识别): {e}[/yellow]")
        return None
    except Exception as e:  # noqa: BLE001 - 简称为可选信息, 任何异常都不阻塞主链路
        console.print(f"[yellow]股票简称获取异常(跳过 ST 识别): {type(e).__name__}: {e}[/yellow]")
        return None
    if not name:
        return None
    cache.save_json(key, name)
    return name


def fetch_northbound(refresh: bool = False) -> pd.DataFrame | None:
    """北向资金逐日净流入(全市场共享). ⚠ 2024-08 起交易所停止逐日披露."""
    key = "north_flow"
    if not refresh and cache.saved_within(key, days=1):
        return cache.load(key)
    try:
        df = akc.get_northbound()
    except akc.AkshareError as e:
        console.print(f"[yellow]北向资金获取失败: {e}[/yellow]")
        return None
    if df is None or df.empty:
        return None
    cache.save(key, df)
    stale_days = (datetime.now().date() - pd.to_datetime(df["date"]).max().date()).days
    if stale_days > 30:
        console.print(
            f"[yellow]北向资金数据最新至 {df['date'].max().date()} (滞后 {stale_days} 天), "
            "疑似已停更(交易所 2024-08 起停止逐日披露), 该组特征将为 NaN[/yellow]"
        )
    return df


def fetch_margin(refresh: bool = False) -> pd.DataFrame | None:
    """沪市融资融券总量(全市场共享, T+1 早晨公布 -> saved_within)."""
    key = "margin_sse"
    if not refresh and cache.saved_within(key, days=1):
        return cache.load(key)
    try:
        df = akc.get_margin_sse()
    except akc.AkshareError as e:
        console.print(f"[yellow]融资融券获取失败: {e}[/yellow]")
        return None
    if df is None or df.empty:
        return None
    cache.save(key, df)
    return df


def fetch_lhb(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """个股龙虎榜事件(date/net_buy). 全市场表共享缓存, 本地按 symbol 过滤.

    源可用但该股无上榜记录 -> 返回空帧(特征层记 0, 区别于源不可用的 NaN)。
    """
    key = "lhb_recent"
    df: pd.DataFrame | None = None
    if not refresh and cache.saved_within(key, days=1):
        df = cache.load(key)
    if df is None:
        try:
            df = akc.get_lhb()
        except akc.AkshareError as e:
            console.print(f"[yellow]龙虎榜获取失败: {e}[/yellow]")
            return None
        if df is None or df.empty:
            return None
        cache.save(key, df)
    return df[df["symbol"] == symbol][["date", "net_buy"]].reset_index(drop=True)


def fetch_holder_events(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """股东增减持事件(date/amount, 增持为正). 无事件返回空帧."""
    key = f"holder_{symbol}"
    if not refresh and cache.saved_within(key, days=1):
        cached = cache.load(key)
        return cached if cached is not None else pd.DataFrame(columns=["date", "amount"])
    try:
        df = akc.get_holder_events(symbol)
    except akc.AkshareError as e:
        console.print(f"[yellow]股东增减持获取失败: {e}[/yellow]")
        return None
    cache.save(key, df)
    return df


def fetch_unlock(symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    """限售股解禁日历(date/ratio, 可含未来行; 特征层只用已发生事件)."""
    key = f"unlock_{symbol}"
    if not refresh and cache.saved_within(key, days=7):
        cached = cache.load(key)
        return cached if cached is not None else pd.DataFrame(columns=["date", "ratio"])
    try:
        df = akc.get_unlock_schedule(symbol)
    except akc.AkshareError as e:
        console.print(f"[yellow]解禁日历获取失败: {e}[/yellow]")
        return None
    cache.save(key, df)
    return df


GLOBAL_SOURCES = (("nasdaq", ".IXIC"), ("spx", ".INX"), ("vix", ".VIX"))


def fetch_global(refresh: bool = False) -> pd.DataFrame | None:
    """外盘指数合并帧(date/nasdaq/spx/vix). 单指数失败降级为缺列, 全失败返回 None."""
    key = "global_idx"
    if not refresh and cache.saved_within(key, days=1):
        return cache.load(key)
    merged: pd.DataFrame | None = None
    for col, sina_symbol in GLOBAL_SOURCES:
        try:
            df = akc.get_global_index(sina_symbol)
        except akc.AkshareError as e:
            console.print(f"[yellow]外盘指数 {sina_symbol} 获取失败: {e}[/yellow]")
            continue
        s = df.set_index("date")["close"].rename(col)
        merged = s.to_frame() if merged is None else merged.join(s, how="outer")
    if merged is None or merged.empty:
        return None
    merged = merged.sort_index().reset_index()
    cache.save(key, merged)
    return merged


def fetch_macro(refresh: bool = False) -> pd.DataFrame | None:
    """宏观月度弱特征帧(date/pmi/cpi_yoy/m2_yoy, 月频 -> saved_within 7 天)."""
    key = "macro_china"
    if not refresh and cache.saved_within(key, days=7):
        return cache.load(key)
    try:
        df = akc.get_macro_china()
    except akc.AkshareError as e:
        console.print(f"[yellow]宏观指标获取失败: {e}[/yellow]")
        return None
    if df is None or df.empty:
        return None
    cache.save(key, df)
    return df


def fetch_all(symbol: str, refresh: bool = False) -> dict[str, pd.DataFrame | None]:
    """拉取个股全部数据(核心 3 源 + Phase 2 扩展 8 源). 失败的数据源置 None."""
    out: dict[str, pd.DataFrame | None] = {}
    for name, fn in (
        ("kline", lambda: fetch_kline(symbol, refresh)),
        ("index", lambda: fetch_index(refresh)),
        ("fund_flow", lambda: fetch_fund_flow(symbol, refresh)),
        ("industry", lambda: fetch_industry(symbol, refresh)),
        ("north", lambda: fetch_northbound(refresh)),
        ("margin", lambda: fetch_margin(refresh)),
        ("lhb", lambda: fetch_lhb(symbol, refresh)),
        ("holder", lambda: fetch_holder_events(symbol, refresh)),
        ("unlock", lambda: fetch_unlock(symbol, refresh)),
        ("global", lambda: fetch_global(refresh)),
        ("macro", lambda: fetch_macro(refresh)),
    ):
        try:
            out[name] = fn()
        except akc.AkshareError as e:
            console.print(f"[red]{name} 获取失败: {e}[/red]")
            out[name] = None
        except Exception as e:  # noqa: BLE001 - 可选源任何异常都不阻塞主链路
            console.print(f"[red]{name} 获取异常(降级跳过): {type(e).__name__}: {e}[/red]")
            out[name] = None
    return out


def source_health(symbol: str, loader=cache.load) -> tuple[list[dict], date]:
    """各数据源缓存健康检查(纯本地, 不发网络请求). loader 可注入便于测试.

    返回 (逐源状态列表, 最近应有交易日). status:
    ok / expired(未覆盖最近交易日) / incomplete(缺关键列) / missing(无缓存),
    可组合为 expired+incomplete。fund_flow 为可选源, missing 不算异常。
    """
    expected = cache.latest_expected_trade_date()
    keys = {
        "kline": f"kline_{symbol}",
        "index": f"index_{INDEX_SYMBOL}",
        "fund_flow": f"fundflow_{symbol}",
    }
    report = []
    for name, key in keys.items():
        df = loader(key)
        entry: dict = {
            "source": name, "key": key, "rows": 0, "latest_date": None,
            "missing_cols": list(EXPECTED_COLUMNS[name]), "fresh": False,
        }
        if df is None or len(df) == 0:
            entry["status"] = "missing"
        else:
            entry["rows"] = len(df)
            if "date" in df.columns:
                ld = pd.to_datetime(df["date"]).max().date()
                entry["latest_date"] = ld
                entry["fresh"] = ld >= expected
            missing = [c for c in EXPECTED_COLUMNS[name] if c not in df.columns]
            entry["missing_cols"] = missing
            problems = []
            if not entry["fresh"]:
                problems.append("expired")
            if missing:
                problems.append("incomplete")
            entry["status"] = "ok" if not problems else "+".join(problems)
        report.append(entry)
    return report, expected


def run_health_check(symbol: str) -> int:
    report, expected = source_health(symbol)
    table = Table(title=f"fetch --check {symbol} (最近应有交易日 {expected})")
    table.add_column("数据源")
    table.add_column("缓存键")
    table.add_column("行数", justify="right")
    table.add_column("最新日期")
    table.add_column("缺失列")
    table.add_column("状态")
    for e in report:
        st = e["status"]
        if st == "ok":
            st_txt = "[green]ok[/green]"
        elif e["source"] == "fund_flow" and st == "missing":
            st_txt = "[yellow]missing(可选源)[/yellow]"
        else:
            st_txt = f"[red]{st}[/red]"
        table.add_row(
            e["source"], e["key"], str(e["rows"]), str(e["latest_date"] or "-"),
            ",".join(e["missing_cols"]) or "-", st_txt,
        )
    console.print(table)

    kline_status = report[0]["status"]
    if kline_status == "missing":
        console.print("[red]核心K线缓存缺失, predict/backtest 不可用; 请先 fetch.[/red]")
        return 1
    if kline_status != "ok":
        console.print(f"[yellow]K线缓存状态 {kline_status}, 建议 fetch --refresh.[/yellow]")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    if getattr(args, "check", False):
        return run_health_check(args.symbol)
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

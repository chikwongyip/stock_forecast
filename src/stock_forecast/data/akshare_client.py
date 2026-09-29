"""akshare 适配层: 所有外部数据请求的唯一出口.

职责: 重试、超时控制、列名归一化(中文列名 -> 英文 snake_case)、明确报错.
akshare 接口随版本变动, 全部收敛到本文件, 修复时只改这里.
"""

import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

RETRIES = 3
RETRY_BACKOFF = 2.0  # 秒, 指数退避基数


class AkshareError(RuntimeError):
    """akshare 调用最终失败."""


def _retry(func, *args, **kwargs) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            df = func(*args, **kwargs)
            if df is None:
                raise AkshareError(f"{func.__name__} 返回 None")
            return df
        except AkshareError:
            raise
        except Exception as e:  # noqa: BLE001 - akshare 异常类型杂, 统一捕获
            last_err = e
            if attempt < RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2**attempt))
    raise AkshareError(f"akshare 调用失败 (重试{RETRIES}次): {last_err}") from last_err


def _col_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """按映射重命名列; 允许部分列缺失(akshare 版本差异), 缺失列记 None."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for cn, en in mapping.items():
        if cn in df.columns and en not in df.columns:
            df = df.rename(columns={cn: en})
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def market_of(symbol: str) -> str:
    """个股所属交易所: 6 开头沪市, 其余按深市处理(0/3 开头)."""
    return "sh" if symbol.startswith("6") else "sz"


def get_kline(symbol: str, days: int = 365) -> pd.DataFrame:
    """日K线(前复权), 英文列: date/open/close/high/low/volume/amount/pct_change/turnover."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(
        ak.stock_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="qfq",
    )
    df = _col_map(df, {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_change", "换手率": "turnover",
    })
    keep = ["date", "open", "close", "high", "low", "volume", "amount",
            "pct_change", "turnover"]
    return df[[c for c in keep if c in df.columns]]


def get_index_kline(symbol: str = "000001", days: int = 365) -> pd.DataFrame:
    """指数日K线, 默认上证指数. 列同 get_kline(无 turnover)."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = _retry(
        ak.index_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start, end_date=end,
    )
    df = _col_map(df, {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
    })
    keep = ["date", "open", "close", "high", "low", "volume", "amount", "pct_change"]
    return df[[c for c in keep if c in df.columns]]


def get_fund_flow(symbol: str) -> pd.DataFrame:
    """个股资金流向(收盘后数据). 关键列: main_net / main_net_pct."""
    df = _retry(ak.stock_individual_fund_flow, stock=symbol, market=market_of(symbol))
    df = _col_map(df, {
        "日期": "date", "收盘价": "close", "涨跌幅": "pct_change",
        "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct",
        "超大单净流入-净额": "super_net", "超大单净流入-净占比": "super_net_pct",
        "大单净流入-净额": "big_net", "大单净流入-净占比": "big_net_pct",
        "中单净流入-净额": "mid_net", "中单净流入-净占比": "mid_net_pct",
        "小单净流入-净额": "small_net", "小单净流入-净占比": "small_net_pct",
    })
    drop = [c for c in ("Unnamed: 0",) if c in df.columns]
    df = df.drop(columns=drop)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_news_raw(symbol: str) -> pd.DataFrame:
    """个股新闻(东方财富). 列: keyword/title/content/publish_time/source_url.

    注意: 接口按关键词匹配, 噪音较多, M4 的 llm/news.py 负责过滤.
    """
    df = _retry(ak.stock_news_em, symbol=symbol)
    if "发布时间" not in df.columns:
        # 列名变动的兜底: 尝试常见等价名
        ren = {"时间": "发布时间", "publish_time": "发布时间"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df

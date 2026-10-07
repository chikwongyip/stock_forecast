"""本地 parquet 缓存: 避免重复拉取网络数据.

约定: 每份数据一个文件 data/cache/{key}.parquet,
key 形如 kline_600519 / fundflow_600519 / index_000001.
"""

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from stock_forecast.config import CACHE_DIR

_CACHE_DIR = Path(CACHE_DIR)


def _path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.parquet"


def load(key: str) -> pd.DataFrame | None:
    """读取缓存, 不存在返回 None."""
    path = _path(key)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # 缓存损坏视同无缓存
        return None


def save(key: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(key), index=False)


def latest_date(key: str) -> pd.Timestamp | None:
    """缓存中最大日期列的值; 无日期列或无缓存返回 None."""
    df = load(key)
    if df is None or df.empty or "date" not in df.columns:
        return None
    return pd.to_datetime(df["date"]).max()


def latest_expected_trade_date(today: date | None = None) -> date:
    """最近的"应当已有数据"的交易日(粗略按工作日估计).

    节假日会被高估, 此时尝试拉网会返回与缓存相同的最新数据, 无害.
    """
    today = today or datetime.now().date()
    d = today
    while d.weekday() >= 5:  # 5=周六 6=周日
        d -= timedelta(days=1)
    return d


def is_fresh(key: str, today: date | None = None) -> bool:
    """缓存是否已覆盖最近一个应有交易日."""
    ld = latest_date(key)
    if ld is None:
        return False
    expected = latest_expected_trade_date(today)
    return ld.date() >= expected


def _any_path(key: str) -> Path | None:
    """parquet 或 pickle 缓存文件, 存在哪个返回哪个."""
    for p in (_path(key), _CACHE_DIR / f"{key}.pkl"):
        if p.exists():
            return p
    return None


def saved_within(key: str, days: float) -> bool:
    """缓存文件是否在最近 days 天内写入(按文件 mtime).

    用于无法按"最新数据日期"判定新鲜度的源: 全市场共享表(龙虎榜)、
    T+1 早晨才公布的滞后数据(融资融券/北向/外盘)、月频宏观、事件日历等。
    """
    p = _any_path(key)
    if p is None:
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return datetime.now() - mtime <= timedelta(days=days)


def save_json(key: str, obj: object) -> None:
    """LLM 结果等非表格数据缓存 (pickle)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_json(key: str) -> object | None:
    p = _CACHE_DIR / f"{key}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

"""个股新闻采集与清洗: stock_news_em -> 近 N 天、去重、截断.

东方财富个股新闻按关键词匹配, 噪音多(可能含非本股资讯), 这里只做
基础清洗; 相关性判断交给 LLM 打分环节(prompt 中要求标注无关新闻).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from stock_forecast.data import akshare_client as akc
from stock_forecast.data import cache

MAX_NEWS = 10          # 单次喂给 LLM 的最大条数(成本控制)
CONTENT_TRUNCATE = 300  # 每条正文截断字数
RECENT_DAYS = 5         # 近 N 天新闻


def _parse_time(v) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(v)
    except Exception:  # noqa: BLE001
        return None


def fetch_news(symbol: str, days: int = RECENT_DAYS, force: bool = False) -> list[dict]:
    """返回近 days 天新闻列表: [{title, content, publish_time}].

    当日缓存: 同 (symbol, date) 不重复拉取.
    """
    today = datetime.now().date().isoformat()
    key = f"news_raw_{symbol}_{today}"
    if not force:
        cached = cache.load_json(key)
        if cached is not None:
            return _filter_recent(cached, days)

    try:
        raw = akc.get_news_raw(symbol)
    except akc.AkshareError:
        return []
    if raw is None or raw.empty:
        cache.save_json(key, [])
        return []

    items = []
    for _, row in raw.iterrows():
        title = str(row.get("新闻标题") or row.get("title") or "").strip()
        content = str(row.get("新闻内容") or row.get("content") or "").strip()
        t = _parse_time(row.get("发布时间") or row.get("publish_time"))
        if not title:
            continue
        items.append({
            "title": title[:200],
            "content": content[:CONTENT_TRUNCATE],
            "publish_time": t.isoformat() if t is not None else None,
        })
    cache.save_json(key, items)
    return _filter_recent(items, days)


def _filter_recent(items: list[dict], days: int) -> list[dict]:
    """过滤近 days 天 + 标题去重 + 截断至 MAX_NEWS 条."""
    cutoff = datetime.now() - timedelta(days=days)
    seen: set[str] = set()
    out = []
    for it in items:
        t = _parse_time(it.get("publish_time")) if it.get("publish_time") else None
        if t is not None and t.tz_localize(None) < cutoff:
            continue
        title = it["title"]
        if title in seen:
            continue
        seen.add(title)
        out.append(it)
        if len(out) >= MAX_NEWS:
            break
    return out


def format_for_prompt(symbol: str, stock_name: str, news: list[dict]) -> str:
    """把新闻列表渲染成 prompt 片段."""
    if not news:
        return "(近期无相关新闻)"
    lines = []
    for i, it in enumerate(news, 1):
        lines.append(
            f"[{i}] ({it.get('publish_time') or '时间未知'}) {it['title']}\n    {it['content']}"
        )
    return "\n".join(lines)

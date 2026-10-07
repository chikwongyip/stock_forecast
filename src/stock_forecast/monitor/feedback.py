"""用户反馈收集 (Phase 7 / 7.4, L8): CLI 子命令与 Web 按钮共用同一落盘函数.

- 记录: `data/feedback.csvl` 一行一条 {schema_version, ts, symbol, rating,
  note, date, source}; rating 用稳定英文键(helpful/wrong/other), 展示层映射中文;
- `feedback --list`: 按月复盘口径输出统计(按评级/按标的计数 + 最近明细);
- Web 端(web/app.py 预测结果下方按钮)以 source="web" 调 `append_feedback`,
  与 CLI 完全同构, 无第二套写入逻辑;
- 读取容错与 track.load_predictions 同口径(坏行/空行静默跳过)。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from stock_forecast.config import FEEDBACK_LOG

console = Console()

SCHEMA_VERSION = 1
RATINGS = ("helpful", "wrong", "other")
RATING_NAMES = {"helpful": "有帮助", "wrong": "有误", "other": "其他"}


def make_record(
    symbol: str, rating: str, note: str = "",
    date: str | None = None, source: str = "cli",
) -> dict:
    """构造一条反馈记录; rating 非法直接 ValueError(调用方给出中文提示)."""
    if rating not in RATINGS:
        raise ValueError(f"rating 必须是 {RATINGS} 之一, 收到: {rating!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "symbol": str(symbol), "rating": rating, "note": str(note or ""),
        "date": date,  # 被反馈的预测日期(可为 None = 泛化反馈)
        "source": source,
    }


def append_feedback(
    symbol: str, rating: str, note: str = "",
    date: str | None = None, source: str = "cli",
    path: Path | None = None,
) -> dict:
    """落盘一条反馈并返回记录; 目录不存在自动创建."""
    record = make_record(symbol, rating, note, date, source)
    path = Path(path or FEEDBACK_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_feedback(path: Path | None = None) -> list[dict]:
    """读取反馈文件; 缺失 -> 空列表, 坏行静默跳过."""
    out: list[dict] = []
    try:
        with open(Path(path or FEEDBACK_LOG), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def summarize_feedback(records: list[dict]) -> dict:
    """月度复盘口径: 总数 / 按评级 / 按标的(降序) / 负反馈带备注明细."""
    by_rating = Counter(r.get("rating", "?") for r in records)
    by_symbol = Counter(r.get("symbol", "?") for r in records)
    notes = [
        {"ts": r.get("ts"), "symbol": r.get("symbol"), "rating": r.get("rating"),
         "note": r.get("note")}
        for r in records
        if r.get("rating") in ("wrong", "other") and r.get("note")
    ]
    return {
        "n": len(records),
        "by_rating": dict(by_rating),
        "by_symbol": dict(by_symbol.most_common()),
        "notes": notes[-20:],  # 最近 20 条带备注的负反馈
    }


def cmd_feedback(args: argparse.Namespace) -> int:
    """feedback 子命令: 无 symbol 时 --list 复盘; 有 symbol 时必须给 --rating."""
    if args.list or not args.symbol:
        return _render_stats()
    if not args.rating:
        console.print("[red]请提供 --rating (helpful/wrong/other), "
                      "或改用 feedback --list 查看统计[/red]")
        return 1
    try:
        record = append_feedback(args.symbol, args.rating, args.note or "",
                                 date=args.date, source="cli")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    console.print(f"[green]已记录反馈: {record['symbol']} "
                  f"{RATING_NAMES[record['rating']]}[/green]"
                  + (f" 备注: {record['note']}" if record["note"] else ""))
    return 0


def _render_stats() -> int:
    records = load_feedback()
    if not records:
        console.print("暂无反馈记录(用法: feedback <代码> --rating helpful|wrong|other "
                      "[--note 说明], 或 Web 端预测结果下方的按钮)。")
        return 0
    s = summarize_feedback(records)
    t = Table(title=f"用户反馈统计(共 {s['n']} 条)")
    t.add_column("评级")
    t.add_column("条数", justify="right")
    for rating in RATINGS:
        t.add_row(RATING_NAMES[rating], str(s["by_rating"].get(rating, 0)))
    console.print(t)

    t2 = Table(title="按标的")
    t2.add_column("标的")
    t2.add_column("条数", justify="right")
    for sym, n in list(s["by_symbol"].items())[:10]:
        t2.add_row(str(sym), str(n))
    console.print(t2)

    if s["notes"]:
        t3 = Table(title="负反馈备注(月度复盘用, 最近 20 条)")
        t3.add_column("时间")
        t3.add_column("标的")
        t3.add_column("评级")
        t3.add_column("备注")
        for n in s["notes"]:
            t3.add_row(str(n["ts"]), str(n["symbol"]),
                       RATING_NAMES.get(n["rating"], str(n["rating"])), str(n["note"]))
        console.print(t3)
    return 0

"""预测埋点结算与实盘准确率追踪 (L8 度量闭环).

读取 data/predictions.csvl, 用结算后的日K计算每条预测的真实方向,
输出滚动准确率、置信度分桶命中率、市场状态分桶(埋点 v2 起)、ML 线 vs 融合线对比。

设计: 纯函数(settle_records/summarize)与 IO(cmd_track)分离, 离线可单测;
结算行情拉取失败按缺数据降级跳过, 不 crash。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from stock_forecast.config import DISCLAIMER, PREDICTIONS_LOG, TRACK_SUMMARY_FILE, get_settings
from stock_forecast.data import collector
from stock_forecast.features import labeling
from stock_forecast.fusion.combine import DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP
from stock_forecast.model import metrics

# 埋点 schema 版本: v0 = 无该字段的旧记录, v1 = 含 schema_version,
# v2 = + regime/regime_as_of(预测信息时点 T-1 的市场状态, Phase 3),
# v3 = + model/disagreement(集成成员名与分歧度, Phase 4),
# v4 = + trading_status/applicability(交易状态与适用性, Phase 5; 读取端全兼容)
SCHEMA_VERSION = 4

DIRECTION_TO_LABEL = {DIRECTION_UP: 1, DIRECTION_FLAT: 0, DIRECTION_DOWN: -1}

console = Console()


def load_predictions(path: Path) -> list[dict]:
    """逐行读取埋点; 坏行/缺关键字段行静默跳过(局部损坏不影响整体结算)."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("symbol") and obj.get("date"):
                obj.setdefault("schema_version", 0)  # 旧埋点无版本号, 视为 v0
                records.append(obj)
    return records


def actual_label_on(kline: pd.DataFrame, date: str, threshold: float) -> int | None:
    """结算日真实三分类标签(close-to-close); K线未覆盖该日(未结算/停牌)返回 None."""
    dates = pd.to_datetime(kline["date"])
    mask = dates == pd.Timestamp(date)
    if not bool(mask.any()):
        return None
    pos = int(mask.values.argmax())
    if pos == 0:
        return None  # 无前收, 无法结算
    pct = float(kline["close"].iloc[pos]) / float(kline["close"].iloc[pos - 1]) - 1
    return labeling.pct_to_label(pct, threshold)


def ml_direction(s_ml: float, direction_threshold: float) -> str:
    """ML 线单独的方向判定(与融合层同一阈值口径), 用于两线对比."""
    if s_ml > direction_threshold:
        return DIRECTION_UP
    if s_ml < -direction_threshold:
        return DIRECTION_DOWN
    return DIRECTION_FLAT


def settle_records(
    records: list[dict],
    kline_by_symbol: dict[str, pd.DataFrame | None],
    label_threshold: float,
    direction_threshold: float,
) -> tuple[pd.DataFrame, int]:
    """埋点记录 -> (已结算明细 DataFrame, 未结算条数).

    明细列: ts/date/symbol/confidence/llm_available/y_true/y_pred_fused/
    y_pred_ml/hit_fused/hit_ml, 按 (date, ts) 升序。
    K线缺失、预测日尚未有行情、direction 字段非法的行计入未结算。
    """
    rows: list[dict] = []
    pending = 0
    for r in records:
        kline = kline_by_symbol.get(r["symbol"])
        if kline is None or len(kline) == 0:
            pending += 1
            continue
        y_true = actual_label_on(kline, str(r["date"]), label_threshold)
        y_fused = DIRECTION_TO_LABEL.get(r.get("direction"))
        if y_true is None or y_fused is None:
            pending += 1
            continue
        row = {
            "ts": str(r.get("ts", "")),
            "date": str(r["date"]),
            "symbol": r["symbol"],
            "confidence": str(r.get("confidence", "")),
            "llm_available": bool(r.get("llm_available", False)),
            # v2 起埋点记录预测时的市场状态; 旧记录回填"未标注"
            "regime": str(r.get("regime") or "未标注"),
            "y_true": y_true,
            "y_pred_fused": y_fused,
            "hit_fused": int(y_true == y_fused),
        }
        s_ml = r.get("s_ml")
        if s_ml is None:
            row["y_pred_ml"] = None
            row["hit_ml"] = None
        else:
            y_ml = DIRECTION_TO_LABEL[ml_direction(float(s_ml), direction_threshold)]
            row["y_pred_ml"] = y_ml
            row["hit_ml"] = int(y_true == y_ml)
        rows.append(row)
    if not rows:
        return pd.DataFrame(), pending
    settled = pd.DataFrame(rows)
    settled["_d"] = pd.to_datetime(settled["date"])
    settled = settled.sort_values(["_d", "ts"]).drop(columns="_d").reset_index(drop=True)
    return settled, pending


def _acc_block(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """一组 (真实, 预测) 的准确率块; 预测缺失的行剔除."""
    mask = y_pred.notna()
    yt, yp = y_true[mask], y_pred[mask].astype(int)
    block: dict = {"n": len(yt)}
    if len(yt) == 0:
        block.update(overall_accuracy=None, direction_accuracy=None)
        return block
    block["overall_accuracy"] = metrics.overall_accuracy(yt, yp)
    da = metrics.direction_accuracy(yt, yp)
    block["direction_accuracy"] = None if pd.isna(da) else float(da)
    return block


def summarize(settled: pd.DataFrame, recent_windows: tuple[int, ...] = (20, 60)) -> dict:
    """已结算明细 -> 汇总指标(滚动窗口 / 置信度分桶 / 双线对比)."""
    out: dict = {"n_settled": len(settled)}
    if settled.empty:
        return out
    out["date_range"] = [settled["date"].iloc[0], settled["date"].iloc[-1]]
    out["fused"] = _acc_block(settled["y_true"], settled["y_pred_fused"])
    out["ml"] = _acc_block(
        settled["y_true"], pd.Series(settled["y_pred_ml"], dtype="Float64")
    )

    rolling: dict = {}
    for w in recent_windows:
        tail = settled.tail(w)
        da = metrics.direction_accuracy(tail["y_true"], tail["y_pred_fused"])
        rolling[str(w)] = {
            "n": len(tail),
            "hit_rate": float(tail["hit_fused"].mean()),
            "direction_accuracy": None if pd.isna(da) else float(da),
        }
    out["rolling"] = rolling

    by_conf: dict = {}
    for conf, g in settled.groupby("confidence"):
        by_conf[str(conf)] = {"n": len(g), "hit_rate": float(g["hit_fused"].mean())}
    out["by_confidence"] = by_conf

    # 市场状态分桶(埋点 v2 起有 regime 字段; 手工构造的旧明细可能无该列)
    by_regime: dict = {}
    if "regime" in settled.columns:
        for rg, g in settled.groupby("regime"):
            da = metrics.direction_accuracy(g["y_true"], g["y_pred_fused"])
            by_regime[str(rg)] = {
                "n": len(g),
                "hit_rate": float(g["hit_fused"].mean()),
                "direction_accuracy": None if pd.isna(da) else float(da),
            }
    out["by_regime"] = by_regime

    # 分 symbol 汇总(Phase 6 / 6.1: predict 展示该股滚动实盘准确率)
    by_symbol: dict = {}
    if "symbol" in settled.columns:
        for sym, g in settled.groupby("symbol"):
            da = metrics.direction_accuracy(g["y_true"], g["y_pred_fused"])
            recent = g.tail(20)
            by_symbol[str(sym)] = {
                "n": len(g),
                "hit_rate": float(g["hit_fused"].mean()),
                "direction_accuracy": None if pd.isna(da) else float(da),
                "recent20_hit_rate": float(recent["hit_fused"].mean()),
            }
    out["by_symbol"] = by_symbol
    return out


def save_summary(summary: dict, path: Path = TRACK_SUMMARY_FILE) -> None:
    """汇总落盘(供 predict/Web 展示该股与全票池实盘准确率). IO 失败静默."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def load_summary(path: Path = TRACK_SUMMARY_FILE) -> dict | None:
    """读取上次 track 汇总; 不存在/损坏返回 None(展示端降级隐藏)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def summary_view(symbol: str | None = None, path: Path = TRACK_SUMMARY_FILE) -> dict | None:
    """predict/Web 展示用紧凑视图: 全票池滚动 + 该股分桶; 无缓存/无结算返回 None."""
    s = load_summary(path)
    if not s or not s.get("n_settled"):
        return None
    rolling = s.get("rolling") or {}
    pool = {
        "n_settled": s["n_settled"],
        "rolling20_direction_accuracy": (rolling.get("20") or {}).get("direction_accuracy"),
        "rolling60_direction_accuracy": (rolling.get("60") or {}).get("direction_accuracy"),
    }
    sym_block = (s.get("by_symbol") or {}).get(str(symbol)) if symbol else None
    return {"as_of": s.get("generated_at"), "pool": pool, "symbol": sym_block}


def _fmt_pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "-"


def _render(summary: dict, pending: int, n_records: int) -> None:
    if summary["n_settled"] == 0:
        console.print(
            f"[yellow]{n_records} 条埋点均未结算(K线尚未覆盖预测日, 或缓存过旧; "
            "可加 --refresh 重拉行情).[/yellow]"
        )
        return
    f, m = summary["fused"], summary["ml"]
    t = Table(title=f"实盘追踪 ({summary['date_range'][0]} ~ {summary['date_range'][1]})")
    t.add_column("指标")
    t.add_column("融合线")
    t.add_column("纯ML线")
    t.add_row("已结算样本数", str(f["n"]), str(m["n"]))
    t.add_row("整体三分类准确率", _fmt_pct(f["overall_accuracy"]), _fmt_pct(m["overall_accuracy"]))
    t.add_row(
        "[bold]方向准确率(剔除'平')[/bold]",
        _fmt_pct(f["direction_accuracy"]), _fmt_pct(m["direction_accuracy"]),
    )
    for w, r in summary["rolling"].items():
        t.add_row(f"滚动{w}日方向准确率(近{r['n']}条)", _fmt_pct(r["direction_accuracy"]), "-")
    console.print(t)

    c = Table(title="置信度分桶命中率(融合线)")
    c.add_column("置信度")
    c.add_column("条数", justify="right")
    c.add_column("命中率")
    for conf in ("高", "中", "低"):
        if conf in summary["by_confidence"]:
            b = summary["by_confidence"][conf]
            c.add_row(conf, str(b["n"]), f"{b['hit_rate']:.1%}")
    console.print(c)

    if summary.get("by_regime"):
        g = Table(title="市场状态分桶(融合线, 埋点 v2 起记录)")
        g.add_column("状态")
        g.add_column("条数", justify="right")
        g.add_column("命中率")
        g.add_column("方向准确率")
        for rg in ("牛市", "熊市", "震荡", "未识别", "未标注"):
            if rg in summary["by_regime"]:
                b = summary["by_regime"][rg]
                g.add_row(rg, str(b["n"]), f"{b['hit_rate']:.1%}",
                          _fmt_pct(b["direction_accuracy"]))
        console.print(g)

    # 分标的汇总(Phase 6 / 6.1): 多标的时才输出, 按样本数降序
    by_symbol = summary.get("by_symbol", {})
    if len(by_symbol) > 1:
        s = Table(title="分标的汇总(融合线)")
        s.add_column("标的")
        s.add_column("条数", justify="right")
        s.add_column("命中率")
        s.add_column("方向准确率")
        s.add_column("近20条命中率")
        for sym, b in sorted(by_symbol.items(), key=lambda kv: -kv[1]["n"]):
            s.add_row(sym, str(b["n"]), f"{b['hit_rate']:.1%}",
                      _fmt_pct(b["direction_accuracy"]), f"{b['recent20_hit_rate']:.1%}")
        console.print(s)

    if pending:
        console.print(f"[dim]{pending} 条埋点未结算(预测日行情尚未覆盖).[/dim]")
    console.print(f"[dim]{DISCLAIMER}[/dim]")


def cmd_track(args: argparse.Namespace) -> int:
    settings = get_settings()
    records = load_predictions(PREDICTIONS_LOG)
    if not records:
        console.print(
            "[yellow]暂无预测埋点: 请先运行 predict (或用 scripts/daily.sh 每日例行).[/yellow]"
        )
        return 0

    kline_by_symbol: dict[str, pd.DataFrame | None] = {}
    for sym in sorted({r["symbol"] for r in records}):
        try:
            kline_by_symbol[sym] = collector.fetch_kline(sym, refresh=args.refresh)
        except Exception as e:  # noqa: BLE001 - 结算行情失败降级跳过, 不阻塞
            console.print(f"[yellow]{sym} 结算K线获取失败, 跳过: {e}[/yellow]")
            kline_by_symbol[sym] = None

    settled, pending = settle_records(
        records, kline_by_symbol, settings.label_threshold, settings.direction_threshold
    )
    summary = summarize(settled)
    # Phase 6 / 6.1: 摘要缓存落盘(含生成时间), predict/Web 直接读取展示,
    # 避免每次预测都重结算全部历史
    save_summary({
        **summary,
        "n_records": len(records), "n_pending": pending,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    if args.json:
        print(json.dumps(
            {"n_records": len(records), "n_pending": pending, "summary": summary},
            ensure_ascii=False, default=str,
        ))
    else:
        _render(summary, pending, len(records))
    return 0

"""输出层(L7): rich 终端报告 + --json 机器可读输出."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stock_forecast.output.scenarios import format_scenario_lines

console = Console()


def _pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "-"

_DIR_COLOR = {"偏涨": "green", "震荡/观望": "yellow", "偏跌": "red"}
_REGIME_COLOR = {"牛市": "green", "熊市": "red", "震荡": "yellow", "未知": "white"}
# 适用性(5.1): 适用=正常, 仅供参考=涨跌停/ST/近期停牌, 不适用=当前停牌
_APPLIC_COLOR = {"适用": "green", "仅供参考": "yellow", "不适用": "red"}


def build_report_dict(symbol: str, date: str, fused: dict, sentiment: dict, narrative: str,
                      regime_info: dict | None = None, scenarios: dict | None = None,
                      track_summary: dict | None = None) -> dict:
    """报告的唯一数据源(CLI --json 与 Web 共用同一 dict, 一致性由构造保证, 6.3 验收)."""
    return {
        "symbol": symbol,
        "date": date,
        "direction": fused["direction"],
        "confidence": fused["confidence"],
        "probs": fused["probs"],
        "signals": {"ml": fused["s_ml"], "llm": fused["s_llm"], "final": fused["s_final"]},
        "llm_available": fused["llm_available"],
        # 市场状态(L4, Phase 3): 预测口径 = T-1 收盘, 与特征信息时点对齐
        "regime": regime_info,
        # 交易状态与适用性(L6, Phase 5): 停牌/涨跌停/ST 降级
        "trading_status": fused.get("trading_status", "正常"),
        "applicability": fused.get("applicability", "适用"),
        "llm_scores": {
            "policy": sentiment.get("policy_score", 0),
            "industry": sentiment.get("industry_score", 0),
            "sentiment": sentiment.get("sentiment_score", 0),
            "confidence": sentiment.get("confidence", 0.0),
        },
        "key_positive": sentiment.get("key_positive", []) if fused["llm_available"] else [],
        "key_negative": sentiment.get("key_negative", []) if fused["llm_available"] else [],
        "warnings": fused["warnings"],
        "narrative": narrative,
        # 预测分布区间(Phase 6 / 6.1): 三点混合分位数 + 情景表; 数据不足为 null
        "scenarios": scenarios,
        # 历史实盘准确率(Phase 6 / 6.1, 来自 track 摘要缓存); 无缓存为 null
        "track_summary": track_summary,
        "disclaimer": fused["disclaimer"],
    }


def build_json_report(symbol: str, date: str, fused: dict, sentiment: dict, narrative: str,
                      regime_info: dict | None = None, scenarios: dict | None = None,
                      track_summary: dict | None = None) -> str:
    report = build_report_dict(symbol, date, fused, sentiment, narrative,
                               regime_info=regime_info, scenarios=scenarios,
                               track_summary=track_summary)
    return json.dumps(report, ensure_ascii=False, indent=2)


def render_report(symbol: str, date: str, fused: dict, sentiment: dict, narrative: str,
                  regime_info: dict | None = None, scenarios: dict | None = None,
                  track_summary: dict | None = None) -> None:
    color = _DIR_COLOR.get(fused["direction"], "white")
    header = (
        f"[bold]{symbol}[/bold]  {date}  "
        f"预测方向: [{color} bold]{fused['direction']}[/{color} bold]  "
        f"置信度: [bold]{fused['confidence']}[/bold]"
    )
    console.print(Panel(header, title="当日走势预测", border_style=color))

    # 适用性横幅(5.1): 停牌 -> 不适用(红); 涨跌停/ST/近期停牌 -> 仅供参考(黄)
    applicability = fused.get("applicability", "适用")
    if applicability != "适用":
        ac = _APPLIC_COLOR.get(applicability, "yellow")
        console.print(Panel(
            f"[{ac} bold]{applicability}[/{ac} bold] — 交易状态: "
            f"{fused.get('trading_status', '-')}",
            title="⚠ 交易状态降级", border_style=ac,
        ))

    p = fused["probs"]
    t = Table(show_header=False)
    t.add_column("项目")
    t.add_column("值")
    t.add_row("上涨概率", f"{p['p_up']:.1%}")
    t.add_row("平盘概率", f"{p['p_flat']:.1%}")
    t.add_row("下跌概率", f"{p['p_down']:.1%}")
    t.add_row("技术面信号 s_ml", f"{fused['s_ml']:+.3f}")
    if fused["llm_available"]:
        t.add_row("舆情面信号 s_llm", f"{fused['s_llm']:+.3f}")
    t.add_row("融合信号 s_final", f"{fused['s_final']:+.3f}")
    if regime_info is not None:
        state = regime_info["regime"] if regime_info["available"] else "未识别"
        rc = _REGIME_COLOR.get(state, "white")
        t.add_row("市场状态(T-1口径)",
                  f"[{rc}]{state}[/{rc}] @ {regime_info.get('as_of') or '-'} "
                  f"— {regime_info['reason']}")
    console.print(t)

    # 预测分布区间(6.1): 三点混合分位数 + 情景表; 数据不足时隐藏
    if scenarios:
        lines = format_scenario_lines(scenarios)
        console.print(
            f"[bold]预测分布区间[/bold] [dim](情景示意, 回看近 "
            f"{scenarios['lookback_days']} 日)[/dim]"
        )
        console.print(f"  {lines[0]}")
        console.print(f"  [dim]{lines[1]} — {lines[2]}[/dim]")

    if fused["llm_available"]:
        if sentiment.get("key_positive"):
            console.print("[green]有利因素:[/green]")
            for x in sentiment["key_positive"]:
                console.print(f"  + {x}")
        if sentiment.get("key_negative"):
            console.print("[red]不利因素:[/red]")
            for x in sentiment["key_negative"]:
                console.print(f"  - {x}")
    else:
        console.print(f"[dim]舆情面不可用: {sentiment.get('reasoning', '')}[/dim]")

    console.print(Panel(narrative, title="综合研判", border_style="blue"))

    # 历史实盘准确率(6.1): 来自 track 摘要缓存; 无缓存/未结算时隐藏
    if track_summary:
        pool = track_summary.get("pool") or {}
        sym = track_summary.get("symbol")
        tk = Table(title=f"历史实盘准确率(track 摘要, 截至 {track_summary.get('as_of') or '-'})")
        tk.add_column("范围")
        tk.add_column("样本", justify="right")
        tk.add_column("指标")
        tk.add_row("全票池", str(pool.get("n_settled") or "-"),
                   f"滚动20日方向准确率 {_pct(pool.get('rolling20_direction_accuracy'))} | "
                   f"滚动60日 {_pct(pool.get('rolling60_direction_accuracy'))}")
        if sym:
            tk.add_row(f"该股 {symbol}", str(sym.get("n") or "-"),
                       f"命中率 {_pct(sym.get('hit_rate'))} | "
                       f"方向准确率 {_pct(sym.get('direction_accuracy'))} | "
                       f"近20条命中率 {_pct(sym.get('recent20_hit_rate'))}")
        console.print(tk)

    for w in fused["warnings"]:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    console.print(f"[dim]{fused['disclaimer']}[/dim]")


def build_blocked_json(payload: dict) -> str:
    """合规阻断时的 --json 输出(不含任何研判, 仅违规说明 + 免责声明)."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_report_simple(payload: dict) -> None:
    """新手模式终端输出(6.2): 方向+置信度+一句话理由+风险提示, 零术语."""
    color = _DIR_COLOR.get(payload.get("direction"), "white")
    header = (
        f"[bold]{payload['symbol']}[/bold]  {payload['date']}  "
        f"研判: [{color} bold]{payload['direction']}[/{color} bold]  "
        f"把握程度: [bold]{payload['confidence']}[/bold]"
    )
    console.print(Panel(header, title="当日走势预测(新手模式)", border_style=color))
    if payload.get("applicability", "适用") != "适用":
        ac = _APPLIC_COLOR.get(payload["applicability"], "yellow")
        console.print(
            f"[{ac} bold]⚠ {payload['applicability']}"
            f"(交易状态: {payload.get('trading_status', '-')})[/{ac} bold]"
        )
    console.print(payload["reason"])
    for w in payload.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/yellow]")
    console.print(f"[dim]{payload['disclaimer']}[/dim]")


def render_blocked(payload: dict) -> None:
    """合规阻断时的终端输出(5.2): 明确告知已阻断, 不渲染任何方向/置信度."""
    console.print(Panel(
        f"[red bold]{payload.get('reason', '合规校验未通过, 已阻断输出')}[/red bold]\n"
        f"标的: [bold]{payload.get('symbol')}[/bold]  日期: {payload.get('date')}",
        title="⛔ 合规拦截", border_style="red",
    ))
    for v in payload.get("violations", []):
        console.print(f"[red]✗ {v}[/red]")
    console.print("[dim]违规已记录至 data/compliance_log.jsonl 供审计。[/dim]")
    console.print(f"[dim]{payload.get('disclaimer', '')}[/dim]")

"""输出层(L7): rich 终端报告 + --json 机器可读输出."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_DIR_COLOR = {"偏涨": "green", "震荡/观望": "yellow", "偏跌": "red"}


def build_json_report(symbol: str, date: str, fused: dict, sentiment: dict, narrative: str) -> str:
    report = {
        "symbol": symbol,
        "date": date,
        "direction": fused["direction"],
        "confidence": fused["confidence"],
        "probs": fused["probs"],
        "signals": {"ml": fused["s_ml"], "llm": fused["s_llm"], "final": fused["s_final"]},
        "llm_available": fused["llm_available"],
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
        "disclaimer": fused["disclaimer"],
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


def render_report(symbol: str, date: str, fused: dict, sentiment: dict, narrative: str) -> None:
    color = _DIR_COLOR.get(fused["direction"], "white")
    header = (
        f"[bold]{symbol}[/bold]  {date}  "
        f"预测方向: [{color} bold]{fused['direction']}[/{color} bold]  "
        f"置信度: [bold]{fused['confidence']}[/bold]"
    )
    console.print(Panel(header, title="当日走势预测", border_style=color))

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
    console.print(t)

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

    for w in fused["warnings"]:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    console.print(f"[dim]{fused['disclaimer']}[/dim]")

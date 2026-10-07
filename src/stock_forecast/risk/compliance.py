"""合规检查清单 (Phase 5 / L6 / 5.2): 输出前固定校验, 违规阻断并记日志.

`fusion.combine` 产出研判后、`output.render` 渲染前的最后一道闸门。三项固定校验:
1. **免责声明存在**: disclaimer 非空且含关键句(不构成任何投资建议);
2. **无买卖措辞**: 方向/警告/研判/LLM 摘要文本不含指令性买卖词(黑名单扫描);
3. **置信度与警告一致**: 有风险警告时置信度不得为"高"(融合层降级的兜底断言),
   方向取值必须合法。

`validate` 为纯函数(返回违规列表, 空 = 通过); `log_violations` 落
`data/compliance_log.jsonl`; `blocked_payload` 构造阻断时的安全输出。
黑名单为指令性**词组**(非单字), 以免"涨跌停/偏涨跌"等合规用词误伤。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from stock_forecast.config import COMPLIANCE_LOG, DISCLAIMER
from stock_forecast.fusion.combine import (
    CONF_HIGH,
    DIRECTION_DOWN,
    DIRECTION_FLAT,
    DIRECTION_UP,
)

# 指令性买卖/诱导措辞黑名单(词组级, 避免误伤"涨跌停""偏涨/偏跌"等合规用词)
FORBIDDEN_WORDS: tuple[str, ...] = (
    "买入", "卖出", "建仓", "清仓", "加仓", "减仓", "抄底", "逃顶",
    "追涨", "杀跌", "满仓", "空仓", "重仓", "目标价", "稳赚", "包赚",
    "无风险", "保本", "推荐买", "推荐卖", "建议买", "建议卖", "必涨", "必跌",
)

_VALID_DIRECTIONS = (DIRECTION_UP, DIRECTION_FLAT, DIRECTION_DOWN)


def validate(fused: dict, narrative: str = "", extra_texts: list[str] | None = None) -> list[str]:
    """对融合输出做合规校验, 返回违规说明列表(空列表 = 通过).

    fused: combine.fuse 的输出 dict; narrative: llm.report 拼装的研判文本;
    extra_texts: 其他会被渲染的用户可见文本(如 LLM key_positive/key_negative)。
    """
    fused = fused or {}
    violations: list[str] = []

    # ① 免责声明存在且完整
    disc = str(fused.get("disclaimer") or "")
    if not disc.strip() or "不构成任何投资建议" not in disc:
        violations.append("免责声明缺失或不完整")

    # ② 无买卖措辞(扫描全部会被渲染的文本)
    texts = [str(fused.get("direction", "")), narrative, *map(str, fused.get("warnings", []))]
    if extra_texts:
        texts += [str(t) for t in extra_texts if t]
    blob = "\n".join(t for t in texts if t)
    hits = [w for w in FORBIDDEN_WORDS if w in blob]
    if hits:
        violations.append(f"出现买卖/诱导措辞: {', '.join(hits)}")

    # ③ 置信度与警告一致 + 方向合法
    warns = fused.get("warnings") or []
    if fused.get("confidence") == CONF_HIGH and warns:
        violations.append(f"置信度为'高'但存在 {len(warns)} 条风险警告, 二者不一致")
    if fused.get("direction") not in _VALID_DIRECTIONS:
        violations.append(f"方向取值非法: {fused.get('direction')!r}")

    return violations


def log_violations(symbol: str, date: str, violations: list[str], path: Path = COMPLIANCE_LOG) -> None:
    """违规项落盘(jsonl, 追加), 供审计. IO 失败静默(不阻塞主链路)."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "symbol": symbol, "date": date, "violations": violations,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def blocked_payload(symbol: str, date: str, violations: list[str]) -> dict:
    """阻断输出时的安全载荷(替代正常研判, 仍带免责声明)."""
    return {
        "blocked": True,
        "symbol": symbol,
        "date": date,
        "reason": "合规校验未通过, 已阻断本次输出",
        "violations": violations,
        "disclaimer": DISCLAIMER,
    }

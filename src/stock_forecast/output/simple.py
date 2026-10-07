"""新手模式输出 (Phase 6 / 6.2): 方向 + 置信度 + 一句话理由 + 风险提示, 零术语.

验收约束(计划 6.2): **simple 模式不含任何专业术语黑名单词**。实现:
- `SIMPLE_FORBIDDEN` 黑名单(模型/信号术语, 中英混合, 大小写不敏感);
- 已知警告逐条映射为白话(`_PLAIN_MAP`, 关键词命中即改写), 未知警告统一兜底
  "存在风险提示, 建议谨慎观望" —— 不透传原文, 从机制上保证不漏术语;
- `build_simple_payload` 产出精简载荷(方向/置信度/一句话理由/白话风险提示/
  免责声明/适用性), 终端与 --json 共用;
- `find_jargon` 供单测断言序列化后的载荷无黑名单词。
"""

from __future__ import annotations

# 专业术语黑名单(小写比对; 验收 = simple 输出序列化后不含任何一项)
SIMPLE_FORBIDDEN: tuple[str, ...] = (
    "s_ml", "s_llm", "s_final", "p_up", "p_flat", "p_down",
    "lgbm", "xgb", "lightgbm", "xgboost", "walk-forward", "booster",
    "特征", "集成", "分歧度", "阈值", "分位", "熵", "argmax", "shift",
    "regime", "schema", "psi", "tv 距离", "nan", "ma20", "ma60",
)

# 已知警告 → 白话改写(按关键词命中, 顺序即优先级)
_PLAIN_MAP: tuple[tuple[str, str], ...] = (
    ("分歧度", "系统内部判断分歧较大, 建议观望"),
    ("熊市", "当前大盘整体偏弱, 看多需格外谨慎"),
    ("冲突", "不同信号方向相反, 建议观望"),
    ("停牌", "该股当前或近期停止交易, 价格信息不完整"),
    ("一字板", "该股近期几乎无法成交, 价格参考意义有限"),
    ("极端行情", "近期价格波动过大, 参考意义有限"),
    ("振幅", "近期价格大起大落, 风险较高"),
    ("量能枯竭", "成交明显萎缩, 市场参与度低"),
    ("涨停", "该股当日涨停, 次日价格可能失真"),
    ("跌停", "该股当日跌停, 次日价格可能失真"),
    ("ST", "该股带有退市风险警示(ST), 请谨慎对待"),
)

_FALLBACK_WARNING = "存在风险提示, 建议谨慎观望"


def plain_warning(warning: str) -> str:
    """单条警告 → 白话(未知警告统一兜底, 不透传原文)."""
    for kw, plain in _PLAIN_MAP:
        if kw in warning:
            return plain
    return _FALLBACK_WARNING


def plain_warnings(warnings: list[str]) -> list[str]:
    """批量改写并去重(保持顺序)."""
    out: list[str] = []
    for w in warnings or []:
        p = plain_warning(str(w))
        if p not in out:
            out.append(p)
    return out


def build_simple_payload(fused: dict, symbol: str = "", date: str = "") -> dict:
    """融合输出 → 新手模式精简载荷(方向+置信度+一句话理由+风险提示)."""
    p = fused.get("probs", {})
    reason = (
        f"综合判断为【{fused.get('direction', '-')}】, 把握程度【{fused.get('confidence', '-')}】。"
        f"统计上看: 上涨可能约 {p.get('p_up', 0):.0%}, 走平约 {p.get('p_flat', 0):.0%}, "
        f"下跌约 {p.get('p_down', 0):.0%}。"
    )
    applicability = fused.get("applicability", "适用")
    payload = {
        "mode": "simple",
        "symbol": symbol,
        "date": date,
        "direction": fused.get("direction", "-"),
        "confidence": fused.get("confidence", "-"),
        "applicability": applicability,
        "trading_status": fused.get("trading_status", "正常"),
        "reason": reason,
        "warnings": plain_warnings(fused.get("warnings", [])),
        "disclaimer": fused.get("disclaimer", ""),
    }
    if applicability == "不适用":
        payload["reason"] = "该股当前不适合给出预测(见风险提示)。" + reason
    elif applicability == "仅供参考":
        payload["reason"] = "该股当前情况特殊, 预测仅供参考。" + reason
    return payload


def find_jargon(text: str) -> list[str]:
    """返回文本中命中的黑名单词(空 = 合规). 供单测与自检使用."""
    low = str(text).lower()
    return [w for w in SIMPLE_FORBIDDEN if w in low]

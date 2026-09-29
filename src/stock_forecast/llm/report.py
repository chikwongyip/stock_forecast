"""综合研判文本: 模板化生成(MVP 不追加 LLM 调用, 零额外成本).

设计文档中"LLM 综合研判"的输出诉求(方向+依据+风险提示)由本模板 +
sentiment.py 的结构化打分共同满足; 若后续需要更自然的叙述, 可在
_generate_narrative 处追加一次 DeepSeek 调用(接口已预留).
"""

from __future__ import annotations


def build_narrative(
    direction: str,
    confidence: str,
    ml_text: str,
    llm_text: str,
    risk_notes: list[str],
) -> str:
    """拼装最终研判段落."""
    parts = [f"综合研判: 模型倾向【{direction}】, 置信度【{confidence}】."]
    if ml_text:
        parts.append(f"技术面: {ml_text}")
    if llm_text:
        parts.append(f"舆情面: {llm_text}")
    if risk_notes:
        parts.append("风险提示: " + "; ".join(risk_notes) + ".")
    return " ".join(parts)


def ml_narrative(p_up: float, p_flat: float, p_down: float, top_features: list[str]) -> str:
    head = ", ".join(top_features[:5]) if top_features else "无"
    return (
        f"上涨概率 {p_up:.0%} / 平 {p_flat:.0%} / 下跌 {p_down:.0%}, "
        f"主要驱动特征: {head}"
    )


def llm_narrative(sentiment: dict) -> str:
    if not sentiment.get("available"):
        return "LLM 线不可用, 仅依据技术面研判."
    pos = "; ".join(sentiment.get("key_positive", [])[:2]) or "无明显利好"
    neg = "; ".join(sentiment.get("key_negative", [])[:2]) or "无明显利空"
    return (
        f"政策 {sentiment['policy_score']:+d} / 行业 {sentiment['industry_score']:+d} / "
        f"舆情 {sentiment['sentiment_score']:+d} (置信 {sentiment['confidence']:.2f}). "
        f"利好: {pos}. 利空: {neg}."
    )

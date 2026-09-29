"""融合层 + 最小风控(L5 融合 / L6 风控).

ML 侧: s_ml = P(涨) - P(跌) ∈ [-1, 1]
LLM 侧: s_llm = (policy + industry + sentiment) / 15 ∈ [-1, 1]
融合:   s_final = w_ml * s_ml + w_llm * s_llm (LLM 不可用时权重归 1/0)

风控:
- 两线方向冲突且模型不确定(|s_ml| < 0.2) -> 强制"震荡/观望", 置信度压到"低"
- 近 5 日累计涨跌超阈值或一字板 -> 加"极端行情, 参考价值低"提示并降置信
- 置信度 = |s_final| 与概率集中度(1-归一化熵)的乘积, 分高/中/低三档
"""

from __future__ import annotations

import math

import pandas as pd

from stock_forecast.config import DISCLAIMER, get_settings

DIRECTION_UP, DIRECTION_FLAT, DIRECTION_DOWN = "偏涨", "震荡/观望", "偏跌"
CONF_HIGH, CONF_MID, CONF_LOW = "高", "中", "低"


def ml_signal(p_up: float, p_down: float) -> float:
    return float(p_up - p_down)


def ml_uncertainty(p_up: float, p_flat: float, p_down: float) -> float:
    """1 - 归一化熵: 0=最大不确定, 1=完全确定."""
    ps = [p for p in (p_up, p_flat, p_down) if p > 0]
    if not ps:
        return 0.0
    h = -sum(p * math.log(p) for p in ps) / math.log(3)
    return 1.0 - h


def llm_signal(sentiment: dict) -> float:
    total = (
        sentiment.get("policy_score", 0)
        + sentiment.get("industry_score", 0)
        + sentiment.get("sentiment_score", 0)
    )
    return total / 15.0


def _confidence_level(score: float, concentration: float) -> str:
    c = abs(score) * concentration
    if c >= 0.5:
        return CONF_HIGH
    if c >= 0.25:
        return CONF_MID
    return CONF_LOW


def detect_extreme(kline_tail: pd.DataFrame, settings) -> list[str]:
    """近 5 日累计涨跌幅超阈值, 或当日一字板 -> 返回风险提示列表."""
    notes = []
    if kline_tail is None or kline_tail.empty:
        return notes
    closes = kline_tail["close"].tail(6)
    if len(closes) >= 2:
        cum = closes.iloc[-1] / closes.iloc[0] - 1
        if abs(cum) >= settings.extreme_move_5d:
            notes.append(f"近5日累计涨跌幅 {cum:+.1%}, 属极端行情, 历史模式参考价值低")
    last = kline_tail.iloc[-1]
    if "open" in last and "high" in last and "low" in last:
        if last["high"] == last["low"]:
            notes.append("当日为一字板, 流动性/价格发现受限")
    return notes


def fuse(
    probs: dict[str, float],
    sentiment: dict,
    kline_tail: pd.DataFrame | None = None,
    settings=None,
) -> dict:
    """融合两线信号, 输出结构化结果(供 render 与 --json 使用)."""
    settings = settings or get_settings()
    p_up, p_flat, p_down = probs["p_up"], probs["p_flat"], probs["p_down"]
    s_ml = ml_signal(p_up, p_down)
    concentration = ml_uncertainty(p_up, p_flat, p_down)

    llm_ok = bool(sentiment.get("available"))
    s_llm = llm_signal(sentiment) if llm_ok else 0.0
    w_ml = settings.ml_weight if llm_ok else 1.0
    w_llm = settings.llm_weight if llm_ok else 0.0
    s_final = w_ml * s_ml + w_llm * s_llm

    warnings: list[str] = []
    direction = (
        DIRECTION_UP if s_final > settings.direction_threshold
        else DIRECTION_DOWN if s_final < -settings.direction_threshold
        else DIRECTION_FLAT
    )
    confidence = _confidence_level(s_final, concentration)

    # 冲突降级: 两线方向相反且模型本身不确定
    if llm_ok and s_ml * s_llm < 0 and abs(s_ml) < 0.2:
        direction = DIRECTION_FLAT
        confidence = CONF_LOW
        warnings.append("技术面与舆情面方向冲突且模型不确定, 已强制降级为震荡/观望")

    # 极端行情降级
    for note in detect_extreme(kline_tail, settings):
        warnings.append(note)
    if warnings and any("极端行情" in w or "一字板" in w for w in warnings):
        if confidence == CONF_HIGH:
            confidence = CONF_MID

    return {
        "direction": direction,
        "confidence": confidence,
        "s_ml": round(s_ml, 4),
        "s_llm": round(s_llm, 4),
        "s_final": round(s_final, 4),
        "probs": {"p_up": round(p_up, 4), "p_flat": round(p_flat, 4), "p_down": round(p_down, 4)},
        "llm_available": llm_ok,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }

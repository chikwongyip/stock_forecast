"""融合层 + 最小风控(L5 融合 / L6 风控).

ML 侧: s_ml = P(涨) - P(跌) ∈ [-1, 1]
LLM 侧: s_llm = (policy + industry + sentiment) / 15 ∈ [-1, 1]
融合:   s_final = w_ml * s_ml + w_llm * s_llm (LLM 不可用时权重归 1/0)

风控:
- 两线方向冲突且模型不确定(|s_ml| < 0.2) -> 强制"震荡/观望", 置信度压到"低"
- 近 5 日累计涨跌超阈值或一字板 -> 加"极端行情, 参考价值低"提示并降置信
- 置信度 = |s_final| 与概率集中度(1-归一化熵)的乘积, 分高/中/低三档

市场状态调整(L4, Phase 3): regime 由 features.regime 判定并做过 T-1 信息时点
对齐(见该模块 docstring); None = 未识别, 不做任何状态调整。规则(参数在 config):
- 震荡市: 方向阈值 × regime_range_threshold_mult(收紧 -> 更多观望)
- 熊市: "偏涨"属逆趋势研判, 置信度下调一档(regime_bear_downgrade_up 可关)

集成分歧度(L5, Phase 4 / 4.5): disagreement = 集成成员概率两两全变差(TV)距离
∈ [0,1](registry.EnsembleModel.disagreement); >= ensemble_disagree_threshold 时
置信度下调一档并警告; None = 单模型运行, 不生效。

风控与合规(L6, Phase 5):
- 极端场景扩充(5.3): 除 detect_extreme 的近5日涨跌/单日一字板外, 追加
  risk.extreme 的连续一字板/近20日振幅异常/量能枯竭; 任一命中把高置信封顶为中。
- 交易状态(5.1): risk_status(risk.status.assess_trading_status 产出)——
  applicable=False(当前停牌)-> 强制"震荡/观望 + 低置信"(不适用);
  reference_only=True(涨跌停/ST/近期停牌)-> 置信度封顶"低"(仅供参考);
  None = 未评估, 完全向后兼容。
- 合规校验(5.2)不在本模块: 见 risk.compliance, 由 cli 在 render 前作最后闸门。
调整顺序: 按(可能收紧的)阈值定方向 -> 冲突降级(硬规则, 可覆盖为观望/低)
-> 熊市降档 -> 分歧度降档 -> 极端行情封顶(高->中) -> 交易状态降级(最强, 可强制观望/低)。
"""

from __future__ import annotations

import math

import pandas as pd

from stock_forecast.config import DISCLAIMER, get_settings
from stock_forecast.features.regime import REGIME_BEAR, REGIME_RANGE
from stock_forecast.risk import extreme as risk_extreme

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
    regime: str | None = None,
    disagreement: float | None = None,
    risk_status: dict | None = None,
) -> dict:
    """融合两线信号, 输出结构化结果(供 render 与 --json 使用).

    regime: 牛市/熊市/震荡(features.regime 判定, T-1 信息时点口径);
    None = 未识别, 不做状态调整(向后兼容)。
    disagreement: 集成成员概率 TV 距离 ∈ [0,1]; None = 单模型, 不做分歧降档。
    risk_status: risk.status.assess_trading_status 产出(停牌/涨跌停/ST 降级,
    Phase 5 / 5.1); None = 未评估, 完全向后兼容。
    kline_tail: 需覆盖最长极端场景窗口(默认量能分位 60 日), 调用方传 tail(>=61)。
    """
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
    # 震荡市收紧方向阈值(更多信号落入观望); 其余状态用基准阈值
    threshold = settings.direction_threshold
    if regime == REGIME_RANGE:
        threshold *= settings.regime_range_threshold_mult
    direction = (
        DIRECTION_UP if s_final > threshold
        else DIRECTION_DOWN if s_final < -threshold
        else DIRECTION_FLAT
    )
    confidence = _confidence_level(s_final, concentration)

    # 冲突降级: 两线方向相反且模型本身不确定
    if llm_ok and s_ml * s_llm < 0 and abs(s_ml) < 0.2:
        direction = DIRECTION_FLAT
        confidence = CONF_LOW
        warnings.append("技术面与舆情面方向冲突且模型不确定, 已强制降级为震荡/观望")

    # 熊市逆趋势降档(在冲突降级之后: 已被强制观望的信号不再触发)
    if regime == REGIME_BEAR and direction == DIRECTION_UP and settings.regime_bear_downgrade_up:
        downgraded = {CONF_HIGH: CONF_MID, CONF_MID: CONF_LOW}.get(confidence)
        if downgraded is not None:
            confidence = downgraded
            warnings.append("市场处于熊市状态, 逆趋势偏涨研判置信度已下调一档")
        else:
            warnings.append("市场处于熊市状态, 逆趋势偏涨研判(置信度已为低)")

    # 集成分歧度降档(4.5): 成员模型概率分歧大 -> 不确定性高, 置信度下调一档
    if (
        disagreement is not None
        and disagreement >= settings.ensemble_disagree_threshold
    ):
        downgraded = {CONF_HIGH: CONF_MID, CONF_MID: CONF_LOW}.get(confidence)
        if downgraded is not None:
            confidence = downgraded
            warnings.append(
                f"集成模型分歧度 {disagreement:.2f} ≥ 阈值 "
                f"{settings.ensemble_disagree_threshold:.2f}, 置信度已下调一档"
            )
        else:
            warnings.append(
                f"集成模型分歧度 {disagreement:.2f} ≥ 阈值 "
                f"{settings.ensemble_disagree_threshold:.2f}(置信度已为低)"
            )

    # 极端行情降级(近5日涨跌/单日一字板 + 5.3 连续一字板/振幅异常/量能枯竭)
    extreme_notes = detect_extreme(kline_tail, settings)
    extreme_notes += risk_extreme.detect_extreme_scenarios(kline_tail, settings)
    for note in extreme_notes:
        warnings.append(note)
    if extreme_notes and confidence == CONF_HIGH:
        confidence = CONF_MID

    # 交易状态降级(5.1, 最强规则, 置于最后): 停牌 -> 不适用; 涨跌停/ST -> 仅供参考
    applicability = "适用"
    trading_status = "正常"
    if risk_status is not None:
        trading_status = risk_status.get("label", "正常")
        for note in risk_status.get("warnings", []):
            warnings.append(note)
        if not risk_status.get("applicable", True):
            # 当前停牌: 价格不产生, 强制观望 + 低置信(不适用)
            direction = DIRECTION_FLAT
            confidence = CONF_LOW
            applicability = "不适用"
        elif risk_status.get("reference_only", False):
            # 涨跌停/ST/近期停牌: 信号可能失真, 置信度封顶"低"(仅供参考)
            confidence = CONF_LOW
            applicability = "仅供参考"

    return {
        "direction": direction,
        "confidence": confidence,
        "s_ml": round(s_ml, 4),
        "s_llm": round(s_llm, 4),
        "s_final": round(s_final, 4),
        "probs": {"p_up": round(p_up, 4), "p_flat": round(p_flat, 4), "p_down": round(p_down, 4)},
        "llm_available": llm_ok,
        "regime": regime or "未识别",
        "disagreement": None if disagreement is None else round(float(disagreement), 4),
        "applicability": applicability,          # 适用 / 仅供参考 / 不适用 (5.1)
        "trading_status": trading_status,        # 交易状态主标签
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }

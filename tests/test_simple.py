"""新手模式(Phase 6 / 6.2)单测. 验收: simple 输出不含任何专业术语黑名单词.

构造覆盖全部警告来源(熊市/分歧度/冲突/极端行情/一字板/振幅/量能枯竭/停牌/
涨跌停/ST)的融合输出, 断言序列化后的载荷无黑名单词、警告已全部白话化。
"""

import json

import numpy as np
import pandas as pd

from stock_forecast.config import get_settings
from stock_forecast.fusion import combine
from stock_forecast.output import simple

S = get_settings()
HI = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
OFF = {"available": False}


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _payload(fused: dict) -> dict:
    return simple.build_simple_payload(fused, "600519", "2026-10-05")


def _assert_clean(payload: dict) -> None:
    hits = simple.find_jargon(_dump(payload))
    assert hits == [], f"simple 输出含术语: {hits}"


def _tail_extreme() -> pd.DataFrame:
    c = pd.Series([100, 101, 102, 103, 104, 125.0])
    return pd.DataFrame({"close": c, "open": c, "high": c, "low": c})  # 连续一字板+极端行情


def _tail_dry_wide() -> pd.DataFrame:
    """量能枯竭 + 近20日振幅异常."""
    n = 61
    close = np.full(n, 10.0)
    high = np.full(n, 10.2)
    low = np.full(n, 9.8)
    high[-20:] = 16.0                    # 近20日振幅 (16-9)/10 = 70% >= 40%
    low[-20:] = 9.0
    vol = np.full(n, 1e6)
    vol[-1] = 1e3                        # 当日量能枯竭
    return pd.DataFrame({"close": close, "open": close, "high": high, "low": low,
                         "volume": vol})


def _status(applicable=True, reference_only=False, label="正常", warning=""):
    return {"applicable": applicable, "reference_only": reference_only, "label": label,
            "statuses": [label], "reason": "", "warnings": [warning] if warning else []}


def _fused_variants() -> list[dict]:
    sent_neg = {
        "available": True, "policy_score": -3, "industry_score": -2, "sentiment_score": -1,
        "confidence": 0.8, "key_positive": [], "key_negative": ["大股东减持"],
        "reasoning": "舆情偏空",
    }
    return [
        # 熊市 + 集成分歧度
        combine.fuse(HI, OFF, settings=S, regime="熊市", disagreement=0.3),
        # 两线冲突(s_ml=0.15 < 0.2 且舆情反向)
        combine.fuse({"p_up": 0.4, "p_flat": 0.35, "p_down": 0.25}, sent_neg, settings=S),
        # 极端行情 + 连续一字板
        combine.fuse(HI, OFF, kline_tail=_tail_extreme(), settings=S),
        # 振幅异常 + 量能枯竭
        combine.fuse(HI, OFF, kline_tail=_tail_dry_wide(), settings=S),
        # 交易状态: ST / 停牌 / 涨停
        combine.fuse(HI, OFF, settings=S, risk_status=_status(
            reference_only=True, label="ST",
            warning="*ST某某 为 ST/*ST 股票(退市风险, 主板限幅 5%), 预测仅供参考")),
        combine.fuse(HI, OFF, settings=S, risk_status=_status(
            applicable=False, label="停牌",
            warning="个股当前停牌(K线止于 2026-09-28, 大盘已到 2026-10-05), 预测不适用")),
        combine.fuse(HI, OFF, settings=S, risk_status=_status(
            reference_only=True, label="涨停",
            warning="当日涨停, 单边行情情绪极端, 次日价格发现可能失真, 预测仅供参考")),
    ]


def test_all_variants_jargon_free():
    """验收硬断言: 全部警告来源的 simple 输出序列化后不含黑名单词."""
    for fused in _fused_variants():
        assert fused["warnings"], "变体应至少含一条警告(否则测试失去意义)"
        _assert_clean(_payload(fused))


def test_payload_structure_and_reason():
    fused = combine.fuse(HI, OFF, settings=S)
    p = _payload(fused)
    assert p["mode"] == "simple"
    assert p["direction"] == fused["direction"] and p["confidence"] == fused["confidence"]
    assert fused["direction"] in p["reason"]            # 一句话理由含方向
    assert p["disclaimer"] == fused["disclaimer"]       # 免责声明恒在
    _assert_clean(p)


def test_not_applicable_reason_prefix():
    fused = combine.fuse(HI, OFF, settings=S, risk_status=_status(
        applicable=False, label="停牌", warning="个股当前停牌, 预测不适用"))
    p = _payload(fused)
    assert p["applicability"] == "不适用"
    assert p["reason"].startswith("该股当前不适合给出预测")


def test_unknown_warning_fallback():
    """未知警告不透传原文, 统一兜底(机制上保证不漏术语)."""
    assert simple.plain_warning("未知的量子波动告警") == "存在风险提示, 建议谨慎观望"
    assert simple.find_jargon(simple.plain_warning("特征分位异常, 阈值突破")) == []


def test_warnings_deduped_and_plain():
    fused = combine.fuse(HI, OFF, kline_tail=_tail_extreme(), settings=S)
    p = _payload(fused)
    assert len(p["warnings"]) == len(set(p["warnings"]))   # 去重
    assert all(simple.find_jargon(w) == [] for w in p["warnings"])


def test_find_jargon_detects():
    assert "s_ml" in simple.find_jargon("模型 s_ml 信号偏高, 特征分位超阈值")

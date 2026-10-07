"""融合层规则测试: 权重、冲突降级、极端行情、置信度、免责声明."""

import numpy as np
import pandas as pd

from stock_forecast.config import DISCLAIMER, get_settings
from stock_forecast.fusion import combine

S = get_settings()


def _sentiment(policy=0, industry=0, sent=0, available=True, conf=0.7):
    return {
        "available": available, "policy_score": policy, "industry_score": industry,
        "sentiment_score": sent, "confidence": conf,
        "key_positive": [], "key_negative": [], "reasoning": "",
    }


def test_ml_only_weight_is_one():
    probs = {"p_up": 0.6, "p_flat": 0.3, "p_down": 0.1}
    out = combine.fuse(probs, _sentiment(available=False), settings=S)
    assert out["llm_available"] is False
    assert np.isclose(out["s_final"], out["s_ml"])  # 权重归 1/0
    assert out["direction"] == combine.DIRECTION_UP  # s_ml=0.5 > 0.15


def test_llm_agreement_boosts():
    probs = {"p_up": 0.55, "p_flat": 0.3, "p_down": 0.15}  # s_ml=0.4
    out = combine.fuse(probs, _sentiment(3, 2, 1), settings=S)  # s_llm=6/15=0.4
    expected = 0.7 * 0.4 + 0.3 * 0.4
    assert np.isclose(out["s_final"], expected, atol=1e-3)
    assert out["direction"] == combine.DIRECTION_UP


def test_conflict_downgrades_to_flat():
    # s_ml = +0.1 (不确定), s_llm = -0.4 -> 方向冲突且 |s_ml|<0.2
    probs = {"p_up": 0.4, "p_flat": 0.35, "p_down": 0.3}
    out = combine.fuse(probs, _sentiment(-3, -2, -1), settings=S)
    assert out["direction"] == combine.DIRECTION_FLAT
    assert out["confidence"] == combine.CONF_LOW
    assert any("冲突" in w for w in out["warnings"])


def test_extreme_move_warning_and_confidence_cap():
    closes = pd.Series([100, 101, 102, 103, 104, 125], dtype=float)  # 5日累计 +25%
    tail = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes})
    probs = {"p_up": 0.8, "p_flat": 0.1, "p_down": 0.1}  # 本应高置信
    out = combine.fuse(probs, _sentiment(available=False), kline_tail=tail, settings=S)
    assert any("极端行情" in w for w in out["warnings"])
    assert out["confidence"] == combine.CONF_MID  # 高 -> 中 降级


def test_limit_up_flag():
    tail = pd.DataFrame({
        "close": [10.0] * 6, "open": [10.0] * 6,
        "high": [11.0] * 5 + [10.0], "low": [9.0] * 5 + [10.0],  # 最后一日 high==low
    })
    out = combine.fuse(
        {"p_up": 0.4, "p_flat": 0.4, "p_down": 0.2},
        _sentiment(available=False), kline_tail=tail, settings=S,
    )
    assert any("一字板" in w for w in out["warnings"])


def test_uncertainty_entropy_bounds():
    uniform = combine.ml_uncertainty(1 / 3, 1 / 3, 1 / 3)
    onehot = combine.ml_uncertainty(1.0, 0.0, 0.0)
    assert np.isclose(uniform, 0.0, atol=1e-9)
    assert np.isclose(onehot, 1.0, atol=1e-9)


def test_disclaimer_always_present():
    out = combine.fuse(
        {"p_up": 0.4, "p_flat": 0.3, "p_down": 0.3}, _sentiment(available=False), settings=S
    )
    assert out["disclaimer"] == DISCLAIMER
    assert "不构成任何投资建议" in out["disclaimer"]


# ---------------------------------------------------------------------------
# Phase 3 (L4): 市场状态调整
# ---------------------------------------------------------------------------

def test_regime_range_tightens_threshold():
    """s_final=0.2: 高于基准阈值 0.15, 低于震荡收紧阈值 0.15*1.5=0.225 -> 观望."""
    probs = {"p_up": 0.6, "p_flat": 0.2, "p_down": 0.4}
    base = combine.fuse(probs, _sentiment(available=False), settings=S)
    assert base["direction"] == combine.DIRECTION_UP
    assert base["regime"] == "未识别"  # regime=None 的展示兜底
    out = combine.fuse(probs, _sentiment(available=False), settings=S, regime="震荡")
    assert out["direction"] == combine.DIRECTION_FLAT
    assert out["regime"] == "震荡"


def test_regime_bear_downgrades_up_confidence():
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}  # s=0.85, 高置信
    base = combine.fuse(probs, _sentiment(available=False), settings=S)
    assert base["direction"] == combine.DIRECTION_UP
    assert base["confidence"] == combine.CONF_HIGH
    out = combine.fuse(probs, _sentiment(available=False), settings=S, regime="熊市")
    assert out["direction"] == combine.DIRECTION_UP  # 只降置信, 不改方向
    assert out["confidence"] == combine.CONF_MID
    assert any("熊市" in w for w in out["warnings"])


def test_regime_bear_downgrade_can_be_disabled():
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    s2 = S.model_copy(update={"regime_bear_downgrade_up": False})
    out = combine.fuse(probs, _sentiment(available=False), settings=s2, regime="熊市")
    assert out["confidence"] == combine.CONF_HIGH


def test_regime_bear_no_effect_on_down():
    probs = {"p_up": 0.05, "p_flat": 0.05, "p_down": 0.9}  # 顺势偏跌
    base = combine.fuse(probs, _sentiment(available=False), settings=S)
    out = combine.fuse(probs, _sentiment(available=False), settings=S, regime="熊市")
    assert base["direction"] == out["direction"] == combine.DIRECTION_DOWN
    assert base["confidence"] == out["confidence"]
    assert not any("熊市" in w for w in out["warnings"])


def test_regime_bull_identical_to_none():
    """牛市不做任何调整(v1 规则只有震荡收紧与熊市降档), 输出与 regime=None 一致."""
    probs = {"p_up": 0.6, "p_flat": 0.2, "p_down": 0.2}
    a = combine.fuse(probs, _sentiment(available=False), settings=S)
    b = combine.fuse(probs, _sentiment(available=False), settings=S, regime="牛市")
    strip = lambda d: {k: v for k, v in d.items() if k != "regime"}  # noqa: E731
    assert strip(a) == strip(b)
    assert b["regime"] == "牛市"


def test_regime_conflict_downgrade_precedes_bear_rule():
    """冲突降级(硬规则)先把方向压成观望 -> 熊市偏涨降档不再触发."""
    probs = {"p_up": 0.4, "p_flat": 0.35, "p_down": 0.3}  # s_ml=+0.1
    out = combine.fuse(probs, _sentiment(-3, -2, -1), settings=S, regime="熊市")
    assert out["direction"] == combine.DIRECTION_FLAT
    assert out["confidence"] == combine.CONF_LOW
    assert any("冲突" in w for w in out["warnings"])
    assert not any("熊市" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Phase 4 (4.5): 集成分歧度降档
# ---------------------------------------------------------------------------

def test_disagreement_downgrades_confidence():
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}  # 高置信偏涨
    base = combine.fuse(probs, _sentiment(available=False), settings=S)
    assert base["confidence"] == combine.CONF_HIGH
    assert base["disagreement"] is None  # 单模型向后兼容
    out = combine.fuse(probs, _sentiment(available=False), settings=S, disagreement=0.2)
    assert out["confidence"] == combine.CONF_MID
    assert out["disagreement"] == 0.2
    assert any("分歧度" in w for w in out["warnings"])


def test_disagreement_below_threshold_no_effect():
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    out = combine.fuse(probs, _sentiment(available=False), settings=S, disagreement=0.05)
    assert out["confidence"] == combine.CONF_HIGH  # 默认阈值 0.10 以下不降档
    assert not any("分歧度" in w for w in out["warnings"])


def test_disagreement_stacks_with_bear_downgrade():
    """降档可叠加: 高 --熊市--> 中 --分歧度--> 低(顺序见 fuse docstring)."""
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    out = combine.fuse(
        probs, _sentiment(available=False), settings=S,
        regime="熊市", disagreement=0.3,
    )
    assert out["confidence"] == combine.CONF_LOW
    assert any("熊市" in w for w in out["warnings"])
    assert any("分歧度" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Phase 5 (L6): 交易状态降级(5.1) + 极端场景扩充接入(5.3)
# ---------------------------------------------------------------------------

def _status(applicable=True, reference_only=False, label="正常", warnings=None):
    return {
        "applicable": applicable, "reference_only": reference_only, "label": label,
        "statuses": [label], "reason": "", "warnings": warnings or [],
    }


def test_fuse_status_not_applicable_forces_flat_low():
    """当前停牌 -> 不适用: 强制震荡/观望 + 低置信(覆盖原偏涨高置信)."""
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    st = _status(applicable=False, label="停牌", warnings=["个股当前停牌, 预测不适用"])
    out = combine.fuse(probs, _sentiment(available=False), settings=S, risk_status=st)
    assert out["direction"] == combine.DIRECTION_FLAT
    assert out["confidence"] == combine.CONF_LOW
    assert out["applicability"] == "不适用"
    assert out["trading_status"] == "停牌"
    assert any("停牌" in w for w in out["warnings"])


def test_fuse_status_reference_only_caps_low():
    """涨跌停/ST -> 仅供参考: 置信度封顶低, 方向保留."""
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    st = _status(reference_only=True, label="涨停", warnings=["当日涨停, 预测仅供参考"])
    out = combine.fuse(probs, _sentiment(available=False), settings=S, risk_status=st)
    assert out["confidence"] == combine.CONF_LOW
    assert out["applicability"] == "仅供参考"
    assert out["direction"] == combine.DIRECTION_UP  # 方向不改


def test_fuse_status_none_backward_compat():
    probs = {"p_up": 0.6, "p_flat": 0.2, "p_down": 0.2}
    out = combine.fuse(probs, _sentiment(available=False), settings=S)
    assert out["applicability"] == "适用" and out["trading_status"] == "正常"


def test_fuse_status_normal_applicable():
    """正常交易状态: 适用, 且不改动置信度(与无 risk_status 一致)."""
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}  # 高置信
    base = combine.fuse(probs, _sentiment(available=False), settings=S)
    out = combine.fuse(probs, _sentiment(available=False), settings=S, risk_status=_status())
    assert out["applicability"] == "适用"
    assert out["confidence"] == base["confidence"] == combine.CONF_HIGH


def test_fuse_extreme_scenario_volume_dry_caps_high():
    """5.3 量能枯竭经 fuse 接入: 高置信被封顶为中."""
    n = 61
    closes = np.full(n, 10.0)
    vol = np.full(n, 1e6)
    vol[-1] = 1e3
    tail = pd.DataFrame({
        "close": closes, "open": closes, "high": closes * 1.02,
        "low": closes * 0.98, "volume": vol,
    })
    probs = {"p_up": 0.9, "p_flat": 0.05, "p_down": 0.05}
    out = combine.fuse(probs, _sentiment(available=False), kline_tail=tail, settings=S)
    assert any("量能枯竭" in w for w in out["warnings"])
    assert out["confidence"] == combine.CONF_MID  # 高 -> 中封顶

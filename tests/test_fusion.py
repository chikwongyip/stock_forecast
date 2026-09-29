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

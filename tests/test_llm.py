"""LLM 线容错测试: 无 key 降级、JSON schema 校验、字段截断.

全部离线, 不发起任何网络调用.
"""

import pytest

from stock_forecast.config import Settings
from stock_forecast.llm.sentiment import _validate, analyze_sentiment


def _valid_obj():
    return {
        "policy_score": 2, "industry_score": -1, "sentiment_score": 3,
        "key_positive": ["a", "b"], "key_negative": ["c"],
        "confidence": 0.8, "reasoning": "ok",
    }


def test_validate_passes_valid():
    obj = _validate(_valid_obj())
    assert obj["policy_score"] == 2
    assert obj["confidence"] == 0.8


def test_validate_missing_field_raises():
    obj = _valid_obj()
    del obj["industry_score"]
    with pytest.raises(ValueError):
        _validate(obj)


def test_validate_out_of_range_raises():
    obj = _valid_obj()
    obj["sentiment_score"] = 9
    with pytest.raises(ValueError):
        _validate(obj)


def test_validate_truncates_lists_and_clamps_confidence():
    obj = _valid_obj()
    obj["key_positive"] = ["1", "2", "3", "4", "5"]
    obj["confidence"] = 1.7
    out = _validate(obj)
    assert len(out["key_positive"]) == 3
    assert out["confidence"] == 1.0


def test_validate_float_scores_rounded():
    obj = _valid_obj()
    obj["policy_score"] = 2.4
    out = _validate(obj)
    assert out["policy_score"] == 2


def test_analyze_sentiment_no_key_degrades(monkeypatch):
    """无 DEEPSEEK_API_KEY 时不 crash, 返回 available=False."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LLM_ENABLED", "")
    result = analyze_sentiment("600519")
    assert result["available"] is False
    assert result["policy_score"] == 0
    assert "LLM" in result["reasoning"] or "不可用" in result["reasoning"]


def test_llm_available_flag():
    from stock_forecast.config import llm_available

    assert llm_available(Settings(deepseek_api_key="", llm_enabled="")) is False
    assert llm_available(Settings(deepseek_api_key="sk-x", llm_enabled="")) is True
    assert llm_available(Settings(deepseek_api_key="sk-x", llm_enabled="false")) is False
    assert llm_available(Settings(deepseek_api_key="", llm_enabled="true")) is True

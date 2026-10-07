"""合规校验(Phase 5 / 5.2)单测: 免责声明/买卖措辞/置信度一致性/阻断落盘. 全离线."""

import json

import pandas as pd

from stock_forecast.config import DISCLAIMER, get_settings
from stock_forecast.fusion import combine
from stock_forecast.risk import compliance

S = get_settings()


def _fused(**over):
    base = combine.fuse({"p_up": 0.4, "p_flat": 0.3, "p_down": 0.3},
                        {"available": False}, settings=S)
    base.update(over)
    return base


# ---------- ① 免责声明 ----------

def test_clean_output_passes():
    assert compliance.validate(_fused(), narrative="综合研判: 观望.") == []


def test_missing_disclaimer_flagged():
    f = _fused()
    f["disclaimer"] = ""
    assert any("免责声明" in x for x in compliance.validate(f))


def test_incomplete_disclaimer_flagged():
    f = _fused()
    f["disclaimer"] = "仅供参考"  # 缺关键句
    assert any("免责声明" in x for x in compliance.validate(f))


# ---------- ② 买卖措辞 ----------

def test_forbidden_word_in_narrative():
    v = compliance.validate(_fused(), narrative="建议买入该股, 目标价 100 元")
    assert any("措辞" in x for x in v)


def test_forbidden_word_in_warnings():
    v = compliance.validate(_fused(warnings=["可适当加仓"]))
    assert any("加仓" in x for x in v)


def test_forbidden_word_in_llm_text():
    v = compliance.validate(_fused(), narrative="", extra_texts=["某券商推荐买入"])
    assert any("推荐买" in x for x in v)


def test_direction_words_not_flagged():
    """涨跌停/偏涨/偏跌等合规用词不应误伤."""
    f = _fused(direction="偏涨", warnings=["当日涨停, 预测仅供参考"])
    assert compliance.validate(f, narrative="模型倾向偏涨") == []


# ---------- ③ 置信度与警告一致 ----------

def test_confidence_warning_inconsistency():
    f = _fused(confidence="高", warnings=["近5日极端行情"])
    assert any("不一致" in x for x in compliance.validate(f))


def test_high_confidence_no_warning_ok():
    assert compliance.validate(_fused(confidence="高", warnings=[])) == []


def test_invalid_direction_flagged():
    f = _fused(direction="大涨")  # 非合法枚举且非黑名单词
    assert any("方向取值非法" in x for x in compliance.validate(f))


# ---------- 真实融合输出恒通过(集成回归) ----------

def test_real_fuse_output_passes_compliance():
    """含极端行情警告的真实 fuse 输出: 置信度已降级, 合规校验应通过."""
    closes = pd.Series([100, 101, 102, 103, 104, 125], dtype=float)
    tail = pd.DataFrame({"close": closes, "open": closes, "high": closes, "low": closes})
    f = combine.fuse({"p_up": 0.8, "p_flat": 0.1, "p_down": 0.1},
                     {"available": False}, kline_tail=tail, settings=S)
    assert f["warnings"] and f["confidence"] != "高"
    assert compliance.validate(f, narrative="综合研判") == []


# ---------- 阻断落盘 + 安全载荷 ----------

def test_log_violations_writes(tmp_path):
    p = tmp_path / "compliance.jsonl"
    compliance.log_violations("600519", "2025-01-06", ["测试违规"], path=p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["symbol"] == "600519" and obj["violations"] == ["测试违规"]
    assert "ts" in obj


def test_blocked_payload():
    p = compliance.blocked_payload("600519", "2025-01-06", ["违规A"])
    assert p["blocked"] is True
    assert p["disclaimer"] == DISCLAIMER
    assert "违规A" in p["violations"]

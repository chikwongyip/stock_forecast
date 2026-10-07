"""Web 与 CLI --json 一致性(Phase 6 / 6.3 验收) + 历史浏览纯函数单测.

一致性由构造保证: CLI --json 打印 pipeline.report_json_text(result),
Web 展示 web.app.report_for_display(result) —— 两者都是 result["report"]
这同一个 dict(render.build_report_dict 是唯一数据源)。本测试做结构断言,
并覆盖 accuracy_curve 纯函数与 web 模块 import 无副作用。
"""

import json

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="streamlit 未安装(镜像 rebuild 后应存在)")

from stock_forecast import pipeline  # noqa: E402
from stock_forecast.config import get_settings  # noqa: E402
from stock_forecast.fusion import combine  # noqa: E402
from stock_forecast.output import render  # noqa: E402
from stock_forecast.web import app as web_app  # noqa: E402  (import 即验证无副作用)

S = get_settings()


def _fake_result() -> dict:
    """离线构造 pipeline 成功结果的骨架(report 走真实 build_report_dict)."""
    fused = combine.fuse(
        {"p_up": 0.5, "p_flat": 0.3, "p_down": 0.2},
        {"available": False},
        settings=S, regime="震荡",
        risk_status={"applicable": True, "reference_only": True, "label": "涨停",
                     "statuses": ["涨停"], "reason": "", "warnings": ["当日涨停, 预测仅供参考"]},
    )
    report = render.build_report_dict(
        "600519", "2026-10-05", fused, {"available": False}, "综合研判: …",
        regime_info={"available": True, "regime": "震荡", "as_of": "2026-10-02", "reason": "…"},
        scenarios={"q25": -0.01, "q25_scenario": "悲观", "q50": 0.0, "q50_scenario": "中性",
                   "q75": 0.012, "q75_scenario": "乐观", "scenarios": [], "lookback_days": 120,
                   "note": "…"},
        track_summary=None,
    )
    return {"ok": True, "blocked": False, "report": report, "fused": fused,
            "symbol": "600519", "date": "2026-10-05"}


# ---------- 6.3 验收: 一致性 ----------

def test_cli_json_and_web_same_source():
    result = _fake_result()
    from_cli = json.loads(pipeline.report_json_text(result))
    from_web = web_app.report_for_display(result)
    assert from_cli == result["report"]
    assert from_web == result["report"]
    assert from_cli == from_web


def test_report_contains_phase6_fields():
    report = _fake_result()["report"]
    for key in ("direction", "confidence", "applicability", "trading_status",
                "regime", "scenarios", "track_summary", "disclaimer", "warnings"):
        assert key in report
    assert report["applicability"] == "仅供参考"   # 涨停降级已进报告
    assert report["scenarios"]["q75_scenario"] == "乐观"


# ---------- 6.4: 准确率曲线纯函数 ----------

def test_accuracy_curve():
    settled = pd.DataFrame({
        "date": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-06", "2026-10-07"],
        "hit_fused": [1, 0, 1, 1, 0],
    })
    curve = web_app.accuracy_curve(settled, window=2)
    assert list(curve["date"]) == list(settled["date"])
    # 累计命中率: 1, 1/2, 2/3, 3/4, 3/5
    assert curve["expanding_hit_rate"].round(4).tolist() == [1.0, 0.5, 0.6667, 0.75, 0.6]
    # 滚动2条命中率: 1, 0.5, 0.5, 1, 0.5
    assert curve["rolling2_hit_rate"].tolist() == [1.0, 0.5, 0.5, 1.0, 0.5]


def test_accuracy_curve_empty():
    curve = web_app.accuracy_curve(pd.DataFrame())
    assert curve.empty
    assert "expanding_hit_rate" in curve.columns


def test_web_app_main_guarded():
    """app 模块可在非 streamlit 运行时下安全 import(main 有 __main__ 守卫)."""
    assert callable(web_app.main)
    assert callable(web_app.settle_all)

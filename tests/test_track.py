"""埋点结算与实盘追踪单测: 全部离线合成数据, 不触网.

覆盖: 埋点读取容错(坏行/旧版本)、真实标签结算、ML/融合双线判定、
滚动窗口与置信度分桶汇总。
"""

import json

import numpy as np
import pandas as pd

from stock_forecast.fusion.combine import DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP
from stock_forecast.monitor import track

# 10 个工作日; close 序列设计成有涨/平/跌三种日
DATES = pd.bdate_range("2025-01-06", periods=10)
CLOSES = [100.0, 101.0, 100.5, 102.0, 101.8, 103.0, 102.9, 104.0, 103.5, 105.0]
THRESHOLD = 0.005
DIR_THRESHOLD = 0.15


def _kline(dates=DATES, closes=CLOSES) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": c, "close": c, "high": c, "low": c,
        "volume": np.full(len(c), 1e6),
    })


def _record(date, direction, s_ml, confidence="中", **kw) -> dict:
    r = {
        "schema_version": 1, "ts": f"{date}T17:00:00", "symbol": "600519",
        "date": date, "direction": direction, "confidence": confidence,
        "p_up": 0.4, "p_flat": 0.3, "p_down": 0.3,
        "s_ml": s_ml, "s_llm": 0.0, "s_final": s_ml, "llm_available": False,
    }
    r.update(kw)
    return r


# ---------- load_predictions ----------

def test_load_predictions_tolerates_bad_lines(tmp_path):
    p = tmp_path / "predictions.csvl"
    good = _record("2025-01-07", DIRECTION_UP, 0.3)
    legacy = _record("2025-01-08", DIRECTION_FLAT, 0.0)
    del legacy["schema_version"]  # 旧版埋点无该字段
    lines = [
        json.dumps(good, ensure_ascii=False),
        "{broken json",                       # 坏行 -> 跳过
        json.dumps({"symbol": "600519"}),     # 缺 date -> 跳过
        "",                                    # 空行 -> 跳过
        json.dumps(legacy, ensure_ascii=False),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    records = track.load_predictions(p)
    assert len(records) == 2
    assert records[0]["schema_version"] == 1
    assert records[1]["schema_version"] == 0  # 旧记录回填 v0


def test_load_predictions_missing_file(tmp_path):
    assert track.load_predictions(tmp_path / "nope.csvl") == []


# ---------- actual_label_on ----------

def test_actual_label_three_ways():
    k = _kline()
    # 01-07: 101/100 - 1 = +1.0% -> 涨
    assert track.actual_label_on(k, "2025-01-07", THRESHOLD) == 1
    # 01-08: 100.5/101 - 1 ≈ -0.495% -> 噪音带内, 平
    assert track.actual_label_on(k, "2025-01-08", THRESHOLD) == 0
    # 01-09: 102/100.5 - 1 ≈ +1.49% -> 涨 (构造明确下跌日用自定义序列)
    k2 = _kline(dates=DATES[:2], closes=[100.0, 98.0])
    assert track.actual_label_on(k2, str(DATES[1].date()), THRESHOLD) == -1


def test_actual_label_unsettled_and_first_row():
    k = _kline()
    assert track.actual_label_on(k, "2025-02-01", THRESHOLD) is None  # K线未覆盖
    assert track.actual_label_on(k, str(DATES[0].date()), THRESHOLD) is None  # 无前收


# ---------- ml_direction ----------

def test_ml_direction_boundaries():
    assert track.ml_direction(0.16, DIR_THRESHOLD) == DIRECTION_UP
    assert track.ml_direction(0.15, DIR_THRESHOLD) == DIRECTION_FLAT  # 恰等于阈值 -> 震荡
    assert track.ml_direction(-0.16, DIR_THRESHOLD) == DIRECTION_DOWN


# ---------- settle_records ----------

def test_settle_records_mixed():
    k = _kline()
    records = [
        _record(str(DATES[1].date()), DIRECTION_UP, 0.3, confidence="高"),    # 真实涨 -> 命中
        _record(str(DATES[2].date()), DIRECTION_UP, 0.3, confidence="低"),    # 真实平 -> 未中
        _record(str(DATES[9].date()), DIRECTION_DOWN, -0.4, confidence="中"),  # 真实涨 -> 未中
        _record("2025-03-01", DIRECTION_UP, 0.5),                              # K线未覆盖 -> pending
        {**_record(str(DATES[3].date()), "非法方向", 0.0)},                    # direction 非法 -> pending
    ]
    settled, pending = track.settle_records(
        records, {"600519": k}, THRESHOLD, DIR_THRESHOLD
    )
    assert pending == 2
    assert len(settled) == 3
    # 按日期升序
    assert settled["date"].tolist() == [str(DATES[i].date()) for i in (1, 2, 9)]
    assert settled["hit_fused"].tolist() == [1, 0, 0]
    # ML 线: s_ml=0.3 -> 偏涨; -0.4 -> 偏跌
    assert settled["y_pred_ml"].tolist() == [1, 1, -1]
    assert settled["hit_ml"].tolist() == [1, 0, 0]


def test_settle_records_no_kline_all_pending():
    records = [_record(str(DATES[1].date()), DIRECTION_UP, 0.3)]
    settled, pending = track.settle_records(records, {"600519": None}, THRESHOLD, DIR_THRESHOLD)
    assert settled.empty and pending == 1


def test_settle_records_without_s_ml():
    """旧埋点可能缺 s_ml: 融合线照常结算, ML 线记 None 不崩."""
    r = _record(str(DATES[1].date()), DIRECTION_UP, 0.3)
    del r["s_ml"]
    settled, _ = track.settle_records([r], {"600519": _kline()}, THRESHOLD, DIR_THRESHOLD)
    assert settled["hit_fused"].iloc[0] == 1
    assert pd.isna(settled["y_pred_ml"].iloc[0])


# ---------- summarize ----------

def _settled_df(n: int = 5) -> pd.DataFrame:
    """构造 n 条已结算明细: 前一半命中, 后一半未中."""
    dates = pd.bdate_range("2025-01-06", periods=n)
    rows = []
    for i, d in enumerate(dates):
        hit = 1 if i < (n + 1) // 2 else 0
        rows.append({
            "ts": f"{d.date()}T17:00:00", "date": str(d.date()), "symbol": "600519",
            "confidence": "高" if i % 2 == 0 else "低",
            "llm_available": False,
            "y_true": 1, "y_pred_fused": 1 if hit else -1,
            "hit_fused": hit, "y_pred_ml": 1 if hit else -1, "hit_ml": hit,
        })
    return pd.DataFrame(rows)


def test_summarize_empty():
    s = track.summarize(pd.DataFrame())
    assert s == {"n_settled": 0}


def test_summarize_blocks():
    settled = _settled_df(6)  # 命中 3, 未中 3
    s = track.summarize(settled, recent_windows=(2, 4))
    assert s["n_settled"] == 6
    assert s["fused"]["n"] == 6
    assert np.isclose(s["fused"]["overall_accuracy"], 0.5)
    # y_true 全为涨(无"平"), 方向准确率 = 整体准确率
    assert np.isclose(s["fused"]["direction_accuracy"], 0.5)
    assert np.isclose(s["ml"]["overall_accuracy"], 0.5)
    # 滚动窗口取尾部: 近2条全未中, 近4条 1命中/3未中(前6条中 i=0,1,2 命中)
    assert s["rolling"]["2"]["n"] == 2 and s["rolling"]["2"]["hit_rate"] == 0.0
    assert s["rolling"]["4"]["n"] == 4 and np.isclose(s["rolling"]["4"]["hit_rate"], 0.25)
    # 置信度分桶: 高=i%2==0 的 3 条(命中 i=0,2 -> 2/3), 低=3 条(命中 i=1 -> 1/3)
    assert s["by_confidence"]["高"]["n"] == 3
    assert np.isclose(s["by_confidence"]["高"]["hit_rate"], 2 / 3)
    assert np.isclose(s["by_confidence"]["低"]["hit_rate"], 1 / 3)


def test_summarize_direction_accuracy_excludes_flat():
    """真实标签为"平"的样本不进方向准确率(与回测口径一致)."""
    settled = _settled_df(2)
    settled.loc[0, "y_true"] = 0  # 第 1 条真实为平且预测涨(未命中口径外)
    s = track.summarize(settled, recent_windows=(2,))
    assert s["fused"]["n"] == 2
    # 方向准确率只看第 2 条(y_true=1, 未中) -> 0.0
    assert s["fused"]["direction_accuracy"] == 0.0


# ---------- Phase 3: 市场状态分桶(埋点 v2 起) ----------

def test_schema_version_current():
    # v4 = + trading_status/applicability(Phase 5); 读取端对 v0~v4 全兼容
    assert track.SCHEMA_VERSION == 4


def test_settle_carries_regime_field():
    """v2 埋点带 regime 字段; 旧记录回填"未标注"."""
    k = _kline()
    r_new = _record(str(DATES[1].date()), DIRECTION_UP, 0.3,
                    regime="牛市", regime_as_of=str(DATES[0].date()))
    r_old = _record(str(DATES[2].date()), DIRECTION_UP, 0.3)
    settled, _ = track.settle_records([r_new, r_old], {"600519": k}, THRESHOLD, DIR_THRESHOLD)
    assert settled["regime"].tolist() == ["牛市", "未标注"]


def test_summarize_by_regime():
    settled = _settled_df(6)  # i=0,1,2 命中; i=3,4,5 未中
    settled["regime"] = ["牛市", "牛市", "熊市", "熊市", "未标注", "未标注"]
    s = track.summarize(settled, recent_windows=(2,))
    br = s["by_regime"]
    assert br["牛市"]["n"] == 2 and br["牛市"]["hit_rate"] == 1.0
    # 熊市桶: i=2 命中 + i=3 未中
    assert br["熊市"]["n"] == 2 and br["熊市"]["hit_rate"] == 0.5
    assert br["未标注"]["hit_rate"] == 0.0
    # y_true 全为涨(无"平") -> 方向准确率与命中率一致
    assert np.isclose(br["牛市"]["direction_accuracy"], 1.0)


def test_summarize_without_regime_column():
    """手工构造的旧明细无 regime 列: 分桶为空, 不 crash."""
    s = track.summarize(_settled_df(2), recent_windows=(2,))
    assert s["by_regime"] == {}


# ---------- Phase 6 (6.1): 分标的汇总 + 摘要缓存 ----------

def test_summarize_by_symbol():
    settled = _settled_df(6)
    settled["symbol"] = ["600519"] * 4 + ["000002"] * 2
    settled["hit_fused"] = [1, 1, 0, 0, 1, 0]
    s = track.summarize(settled, recent_windows=(2,))
    assert s["by_symbol"]["600519"]["n"] == 4
    assert np.isclose(s["by_symbol"]["600519"]["hit_rate"], 0.5)
    assert s["by_symbol"]["000002"]["n"] == 2
    assert np.isclose(s["by_symbol"]["000002"]["hit_rate"], 0.5)
    assert "recent20_hit_rate" in s["by_symbol"]["600519"]


def test_summary_cache_roundtrip(tmp_path):
    p = tmp_path / "track_summary.json"
    track.save_summary({"n_settled": 3, "generated_at": "2026-10-07T09:00:00"}, path=p)
    loaded = track.load_summary(p)
    assert loaded["n_settled"] == 3 and loaded["generated_at"] == "2026-10-07T09:00:00"


def test_load_summary_missing_or_bad(tmp_path):
    assert track.load_summary(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert track.load_summary(bad) is None


def test_summary_view(tmp_path):
    p = tmp_path / "s.json"
    track.save_summary({
        "n_settled": 5, "generated_at": "2026-10-07T09:00:00",
        "rolling": {"20": {"n": 5, "hit_rate": 0.6, "direction_accuracy": 0.57},
                    "60": {"n": 5, "hit_rate": 0.6, "direction_accuracy": 0.55}},
        "by_symbol": {"600519": {"n": 3, "hit_rate": 0.67,
                                 "direction_accuracy": 0.6, "recent20_hit_rate": 0.67}},
    }, path=p)
    v = track.summary_view("600519", path=p)
    assert v["pool"]["n_settled"] == 5
    assert np.isclose(v["pool"]["rolling20_direction_accuracy"], 0.57)
    assert np.isclose(v["pool"]["rolling60_direction_accuracy"], 0.55)
    assert v["symbol"]["n"] == 3
    assert v["as_of"] == "2026-10-07T09:00:00"
    # 未收录的标的: symbol 块为 None, 全票池块照常
    assert track.summary_view("999999", path=p)["symbol"] is None


def test_summary_view_empty(tmp_path):
    p = tmp_path / "s.json"
    track.save_summary({"n_settled": 0}, path=p)
    assert track.summary_view("600519", path=p) is None
    assert track.summary_view("600519", path=tmp_path / "nope.json") is None

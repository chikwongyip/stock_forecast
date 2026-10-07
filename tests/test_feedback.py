"""用户反馈收集单测 (Phase 7.4): 记录构造/落盘回环/容错/复盘统计/CLI 分派.

Web 端按钮与 CLI 共用 monitor.feedback.append_feedback(同一 schema),
此处直接测共享函数即同时覆盖两端的写入路径。
"""

import argparse
import json

import pytest

from stock_forecast.monitor import feedback


# ---------- make_record / append / load ----------

def test_make_record_fields():
    r = feedback.make_record("600519", "wrong", note="方向判错", date="2026-10-07",
                             source="web")
    assert r["schema_version"] == feedback.SCHEMA_VERSION
    assert r["symbol"] == "600519" and r["rating"] == "wrong"
    assert r["note"] == "方向判错" and r["date"] == "2026-10-07"
    assert r["source"] == "web" and "ts" in r


def test_make_record_invalid_rating():
    with pytest.raises(ValueError):
        feedback.make_record("600519", "buy")  # 非枚举评级直接拒绝


def test_append_and_load_roundtrip(tmp_path):
    p = tmp_path / "feedback.csvl"
    feedback.append_feedback("600519", "helpful", date="2026-10-07", path=p)
    feedback.append_feedback("000858", "wrong", note="漏了停牌提示", source="web", path=p)
    records = feedback.load_feedback(p)
    assert len(records) == 2
    assert records[0]["rating"] == "helpful" and records[0]["source"] == "cli"
    assert records[1]["note"] == "漏了停牌提示" and records[1]["source"] == "web"


def test_load_feedback_missing_and_bad_lines(tmp_path):
    assert feedback.load_feedback(tmp_path / "nope.csvl") == []
    p = tmp_path / "f.csvl"
    good = json.dumps(feedback.make_record("600519", "other"), ensure_ascii=False)
    p.write_text(f"{good}\n{{broken\n\n", encoding="utf-8")
    records = feedback.load_feedback(p)
    assert len(records) == 1 and records[0]["rating"] == "other"


# ---------- 复盘统计 ----------

def test_summarize_feedback():
    records = [
        feedback.make_record("600519", "helpful"),
        feedback.make_record("600519", "wrong", note="判反了"),
        feedback.make_record("000858", "wrong"),          # 无备注 -> 不进 notes
        feedback.make_record("000858", "other", note="置信度存疑"),
    ]
    s = feedback.summarize_feedback(records)
    assert s["n"] == 4
    assert s["by_rating"] == {"helpful": 1, "wrong": 2, "other": 1}
    assert s["by_symbol"]["600519"] == 2
    assert len(s["notes"]) == 2  # 只收 wrong/other 且有备注的
    assert all(n["rating"] in ("wrong", "other") for n in s["notes"])


def test_summarize_empty():
    s = feedback.summarize_feedback([])
    assert s["n"] == 0 and s["by_rating"] == {} and s["notes"] == []


# ---------- CLI 分派 ----------

def _args(**kw) -> argparse.Namespace:
    base = {"symbol": None, "rating": None, "note": "", "date": None, "list": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_feedback_writes_record(tmp_path, monkeypatch):
    p = tmp_path / "feedback.csvl"
    monkeypatch.setattr(feedback, "FEEDBACK_LOG", p)
    rc = feedback.cmd_feedback(_args(symbol="600519", rating="helpful", note="有用"))
    assert rc == 0
    records = feedback.load_feedback(p)
    assert len(records) == 1 and records[0]["note"] == "有用"


def test_cmd_feedback_symbol_without_rating_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_LOG", tmp_path / "f.csvl")
    assert feedback.cmd_feedback(_args(symbol="600519")) == 1


def test_cmd_feedback_list_empty_and_with_records(tmp_path, monkeypatch, capsys):
    p = tmp_path / "feedback.csvl"
    monkeypatch.setattr(feedback, "FEEDBACK_LOG", p)
    assert feedback.cmd_feedback(_args(list=True)) == 0  # 空文件 -> 提示, 退出 0
    feedback.append_feedback("600519", "wrong", note="x", path=p)
    assert feedback.cmd_feedback(_args(list=True)) == 0

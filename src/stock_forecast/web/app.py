"""Streamlit Web 界面 v1 (Phase 6 / 6.3 预测 + 6.4 历史浏览). 只读, 无账户体系.

运行: `docker compose up web` → http://localhost:8501
- 当日预测: 输入代码 → `pipeline.predict_once`(与 CLI 完全同一编排)→ 渲染报告;
  Web 展示数据源 = result["report"], 即 CLI --json 输出的**同一个 dict**
  (6.3 验收"Web 与 CLI --json 一致性"由构造保证, 结构断言见 test_web_consistency)。
- 历史浏览: 读 data/predictions.csvl → track 结算 → 准确率曲线 + 分标的汇总。
- 纯函数(report_for_display/accuracy_curve)与 UI 分离, 离线可单测;
  `main()` 带 __main__ 守卫, import 本模块无副作用。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_forecast.config import DISCLAIMER, PREDICTIONS_LOG, get_settings
from stock_forecast.monitor import feedback, track
from stock_forecast.pipeline import predict_once


# ---------------------------------------------------------------------------
# 纯函数(可离线单测)
# ---------------------------------------------------------------------------

def report_for_display(result: dict) -> dict:
    """Web 展示的数据源 = result["report"](CLI --json 的同一 dict)."""
    return result["report"]


def accuracy_curve(settled: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """已结算明细 → 准确率曲线帧(date / 累计命中率 / 滚动命中率). 纯函数."""
    cols = ["date", "expanding_hit_rate", f"rolling{window}_hit_rate"]
    if settled is None or settled.empty:
        return pd.DataFrame(columns=cols)
    df = settled.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"].astype(str)})
    out["expanding_hit_rate"] = df["hit_fused"].expanding().mean()
    out[f"rolling{window}_hit_rate"] = df["hit_fused"].rolling(window, min_periods=1).mean()
    return out


def settle_all(records: list[dict], settings=None) -> tuple[pd.DataFrame, dict, int]:
    """拉行情结算全部埋点(cache 优先; 单标的行情失败降级跳过, 不阻塞)."""
    from stock_forecast.data import collector

    settings = settings or get_settings()
    kline_by_symbol: dict[str, pd.DataFrame | None] = {}
    for sym in sorted({r["symbol"] for r in records}):
        try:
            kline_by_symbol[sym] = collector.fetch_kline(sym)
        except Exception:  # noqa: BLE001 - 结算行情失败降级跳过
            kline_by_symbol[sym] = None
    settled, pending = track.settle_records(
        records, kline_by_symbol, settings.label_threshold, settings.direction_threshold
    )
    return settled, track.summarize(settled), pending


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _render_simple(payload: dict) -> None:
    """新手模式(6.2): 方向+置信度+一句话理由+白话风险提示, 零术语."""
    c1, c2, c3 = st.columns(3)
    c1.metric("研判", payload["direction"])
    c2.metric("把握程度", payload["confidence"])
    c3.metric("适用性", payload["applicability"])
    st.markdown(f"**{payload['reason']}**")
    for w in payload.get("warnings", []):
        st.warning(w)
    st.caption(payload["disclaimer"])


def _render_pro(result: dict) -> None:
    """全量报告: 与 CLI pro 模式同源数据(result['report'])."""
    report = report_for_display(result)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预测方向", report["direction"])
    c2.metric("置信度", report["confidence"])
    c3.metric("适用性", report["applicability"])
    regime = report.get("regime") or {}
    c4.metric("市场状态(T-1)", regime.get("regime", "未识别") if regime else "未识别")

    p = report["probs"]
    st.markdown(f"**三分类概率**: 上涨 {p['p_up']:.1%} | 平盘 {p['p_flat']:.1%} | "
                f"下跌 {p['p_down']:.1%}")
    st.progress(p["p_up"], text=f"上涨概率 {p['p_up']:.1%}")

    scenarios = report.get("scenarios")
    if scenarios:
        with st.expander(f"预测分布区间(情景示意, 回看近 {scenarios['lookback_days']} 日)"):
            q = pd.DataFrame({
                "分位": ["p25", "p50", "p75"],
                "幅度": [f"{scenarios['q25']:+.1%}", f"{scenarios['q50']:+.1%}",
                        f"{scenarios['q75']:+.1%}"],
                "情景": [scenarios["q25_scenario"], scenarios["q50_scenario"],
                        scenarios["q75_scenario"]],
            })
            st.table(q)
            st.dataframe(pd.DataFrame(scenarios["scenarios"]).rename(columns={
                "name": "情景", "prob": "概率", "typical_move": "典型幅度"}), hide_index=True)
            st.caption(scenarios["note"])

    ts = report.get("track_summary")
    if ts:
        with st.expander(f"历史实盘准确率(track 摘要, 截至 {ts.get('as_of') or '-'})"):
            pool = ts.get("pool") or {}
            st.markdown(
                f"全票池: 已结算 {pool.get('n_settled', '-')} 条 | 滚动20日方向准确率 "
                f"{pool.get('rolling20_direction_accuracy')} | 滚动60日 "
                f"{pool.get('rolling60_direction_accuracy')}"
            )
            if ts.get("symbol"):
                st.write("该股:", ts["symbol"])

    if report.get("llm_available"):
        with st.expander("LLM 舆情线"):
            st.write(report["llm_scores"])
            if report.get("key_positive"):
                st.success("有利因素: " + "; ".join(report["key_positive"]))
            if report.get("key_negative"):
                st.error("不利因素: " + "; ".join(report["key_negative"]))

    for w in report["warnings"]:
        st.warning(w)
    st.info(report["narrative"])
    with st.expander("机器可读报告(与 CLI --json 同一数据源)"):
        st.json(report)
    st.caption(report["disclaimer"])


def _feedback_area(result: dict) -> None:
    """7.4 Web 端反馈: 与 CLI `feedback` 共用 monitor.feedback.append_feedback."""
    st.divider()
    st.caption("这次研判对你有帮助吗? 反馈落 data/feedback.csvl(纳入月度复盘)")
    note = st.text_input("备注(可选)", key="fb_note",
                         placeholder="如: 方向判错 / 风险提示有用 …")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("👍 有帮助"):
        feedback.append_feedback(result["symbol"], "helpful", note,
                                 date=result["date"], source="web")
        st.success("已记录反馈, 谢谢!")
    if c2.button("👎 有误"):
        feedback.append_feedback(result["symbol"], "wrong", note,
                                 date=result["date"], source="web")
        st.success("已记录反馈, 月度复盘时会核对错误原因。")


def _predict_tab() -> None:
    with st.sidebar:
        st.header("预测设置")
        symbol = st.text_input("股票代码", value="600519")
        use_llm = st.checkbox("启用 LLM 舆情线", value=False,
                              help="需在 .env 配置 DEEPSEEK_API_KEY")
        mode = st.radio("输出模式", ("pro", "simple"), index=0,
                        help="simple = 新手模式(零术语)")
        go = st.button("开始预测", type="primary")
    result = None
    if go:
        with st.spinner("预测中(数据 → 特征 → 模型 → 舆情 → 融合 → 风控 → 合规)…"):
            result = predict_once(symbol.strip(), use_llm=use_llm,
                                  llm_skip_reason="Web 端未启用 LLM 舆情线")
        # 结果进会话态: 点击反馈按钮触发 rerun 时报告不丢失(7.4)
        st.session_state["sf_last_result"] = result
    else:
        result = st.session_state.get("sf_last_result")
    if result is None:
        st.info("在左侧输入股票代码后点击「开始预测」。"
                "需先执行 `docker compose run --rm sf backtest --pool` 训练模型。")
        return
    for note in result.get("notices", []):
        st.warning(note)
    if not result["ok"]:
        st.error(result["error"])
        return
    if result["blocked"]:
        st.error(result["blocked_payload"]["reason"])
        for v in result["violations"]:
            st.markdown(f"- ✗ {v}")
        st.caption("违规已记录至 data/compliance_log.jsonl")
        return
    st.subheader(f"{result['symbol']}  {result['date']}")
    if mode == "simple":
        _render_simple(result["simple"])
    else:
        _render_pro(result)
    _feedback_area(result)


def _history_tab() -> None:
    records = track.load_predictions(PREDICTIONS_LOG)
    if not records:
        st.info("暂无预测埋点: 请先运行 predict(或 scripts/daily.sh 每日例行)。")
        return
    st.markdown(f"共 **{len(records)}** 条埋点, 覆盖 {records[0]['date']} ~ {records[-1]['date']}")
    if st.button("结算并生成准确率曲线"):
        with st.spinner("结算中(拉取行情, 缓存优先)…"):
            settled, summary, pending = settle_all(records)
        if settled.empty:
            st.warning("所有埋点均未结算(行情尚未覆盖预测日, 或缓存过旧)。")
            return
        f = summary.get("fused", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("已结算样本", f.get("n", 0))
        c2.metric("三分类准确率", f"{f.get('overall_accuracy') or 0:.1%}")
        da = f.get("direction_accuracy")
        c3.metric("方向准确率(剔除'平')", f"{da:.1%}" if da is not None else "-")
        st.line_chart(accuracy_curve(settled).set_index("date"))
        if summary.get("by_symbol"):
            st.subheader("分标的汇总")
            st.dataframe(pd.DataFrame(summary["by_symbol"]).T.rename(columns={
                "n": "条数", "hit_rate": "命中率",
                "direction_accuracy": "方向准确率", "recent20_hit_rate": "近20条命中率",
            }), hide_index=True)
        with st.expander(f"已结算明细(最近 50 条, 另 {pending} 条未结算)"):
            st.dataframe(settled.tail(50), hide_index=True)
    with st.expander("原始埋点(最近 50 条)"):
        st.dataframe(pd.DataFrame(records[-50:]), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="A股日频走势预测", page_icon="📈", layout="wide")
    st.title("A股日频走势预测系统 (MVP)")
    st.caption(DISCLAIMER)
    tab_predict, tab_history = st.tabs(["当日预测", "历史浏览"])
    with tab_predict:
        _predict_tab()
    with tab_history:
        _history_tab()


if __name__ == "__main__":
    main()

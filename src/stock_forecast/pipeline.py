"""端到端预测编排 (Phase 6 / 6.3): CLI 与 Web 共用的唯一预测入口.

设计: 原 `cli.run_predict` 的编排逻辑抽到 `predict_once`(纯编排, 不做终端渲染),
CLI 与 Streamlit Web 消费**同一个** result["report"] dict —— "Web 与 CLI --json
输出一致"(6.3 验收)由构造保证, `tests/test_web_consistency.py` 做结构断言。

result 契约:
- {"ok": False, "error": 中文错误, "notices": [...]}      模型/K线/特征不可用;
- {"ok": True, "blocked": True, "violations", "blocked_payload"}  合规阻断(不写埋点);
- {"ok": True, "blocked": False, "report", "simple", "fused", "sentiment",
  "narrative", "regime_info", "scenarios", "track_summary", "record", ...}  正常
  (埋点已在 pipeline 内写入 data/predictions.csvl)。
notices = 过程中的降级提示(黄色), 展示端自行决定如何呈现。
"""

from __future__ import annotations

import json


def predict_once(
    symbol: str,
    use_llm: bool = True,
    settings=None,
    llm_skip_reason: str = "--no-llm: 已跳过 LLM 舆情线",
) -> dict:
    """fetch -> features -> ML 推理 -> LLM 舆情 -> 风控 -> 融合 -> 合规 -> report+埋点."""
    from datetime import datetime

    from stock_forecast.config import DATA_DIR, PREDICTIONS_LOG, get_settings
    from stock_forecast.data import collector
    from stock_forecast.features import build as fb
    from stock_forecast.features import regime
    from stock_forecast.fusion import combine
    from stock_forecast.llm import report as rpt
    from stock_forecast.llm.sentiment import analyze_sentiment
    from stock_forecast.model import registry
    from stock_forecast.monitor import track as track_mod
    from stock_forecast.output import render
    from stock_forecast.output import scenarios as scen
    from stock_forecast.output import simple as simple_mod
    from stock_forecast.risk import compliance
    from stock_forecast.risk import status as risk_status

    settings = settings or get_settings()
    notices: list[str] = []

    loaded = registry.load_predictor()
    if loaded is None:
        return {"ok": False, "notices": notices,
                "error": "未找到已训练模型. 请先运行: docker compose run --rm sf backtest --pool"}
    predictor, model_meta = loaded
    if model_meta.get("skipped_members"):
        notices.append(
            f"集成成员不可用, 已退化为 {predictor.name}: {model_meta['skipped_members']}"
        )

    data = collector.fetch_all(symbol)
    kline = data.get("kline")
    if kline is None or kline.empty:
        return {"ok": False, "notices": notices, "error": f"{symbol} K线数据不可用"}

    X, _y, _optional = fb.build_feature_matrix(
        kline, data["index"], data["fund_flow"], settings.label_threshold,
        extras=fb.make_extras(data),
    )
    X_latest = X.iloc[[-1]]
    feats = predictor.feature_names()
    missing = [c for c in feats if c not in X_latest.columns]
    if missing:
        return {"ok": False, "notices": notices, "error": f"特征缺失(缓存过旧?): {missing[:5]} ..."}
    probs = predictor.predict_proba(X_latest[feats]).iloc[0].to_dict()
    # 集成分歧度(4.5): 成员概率 TV 距离; 单模型运行时为 None(融合层不降档)
    disagreement = None
    if isinstance(predictor, registry.EnsembleModel):
        disagreement = float(predictor.disagreement(X_latest[feats]).iloc[0])

    sentiment = (
        {"available": False, "reasoning": llm_skip_reason}
        if not use_llm
        else analyze_sentiment(symbol)
    )

    # 市场状态(L4): T-1 收盘口径, 与 X[T] 特征信息时点严格对齐
    regime_info = regime.regime_for_prediction(data.get("index"), settings)
    # 交易状态(L6 / 5.1): 停牌/涨跌停/ST; 简称缺失时 ST 检查自动跳过
    stock_name = collector.fetch_stock_name(symbol)
    trading_status = risk_status.assess_trading_status(
        symbol, kline, data.get("index"), stock_name, settings
    )

    # kline_tail 需覆盖最长极端场景窗口(量能分位 60 日 -> 取 70 行留余量)
    fused = combine.fuse(
        probs, sentiment, kline_tail=kline.tail(70), settings=settings,
        regime=regime_info["regime"] if regime_info["available"] else None,
        disagreement=disagreement,
        risk_status=trading_status,
    )

    # top5 特征归因: 口径 = LightGBM 成员 gain(集成的可解释性锚点; 缺失则留空)
    lgbm = (
        predictor if isinstance(predictor, registry.LGBMModel)
        else next((m for m in predictor.members if isinstance(m, registry.LGBMModel)), None)
    )
    if lgbm is not None:
        gains = lgbm.booster.feature_importance(importance_type="gain")
        top_names = [n for n, _ in sorted(zip(lgbm.feature_names(), gains),
                                          key=lambda t: -t[1])[:5]]
    else:
        top_names = []

    date_str = str(X.index[-1].date())
    narrative = rpt.build_narrative(
        fused["direction"], fused["confidence"],
        rpt.ml_narrative(probs["p_up"], probs["p_flat"], probs["p_down"], top_names),
        rpt.llm_narrative(sentiment),
        fused["warnings"],
    )

    # L7 (6.1): 预测分布区间(情景示意) + track 历史实盘准确率摘要(均可 None 降级)
    scenarios = scen.scenario_quantiles(
        fused["probs"], kline, settings.label_threshold, settings.scenario_lookback
    )
    track_summary = track_mod.summary_view(symbol)

    # 合规校验(L6 / 5.2): render 之前的最后闸门
    llm_texts: list[str] = []
    if fused["llm_available"]:
        llm_texts = list(sentiment.get("key_positive", [])) + list(sentiment.get("key_negative", []))
        if sentiment.get("reasoning"):
            llm_texts.append(str(sentiment["reasoning"]))
    violations = compliance.validate(fused, narrative=narrative, extra_texts=llm_texts)
    if violations:
        compliance.log_violations(symbol, date_str, violations)
        if settings.compliance_block:
            return {"ok": True, "blocked": True, "notices": notices, "violations": violations,
                    "symbol": symbol, "date": date_str,
                    "blocked_payload": compliance.blocked_payload(symbol, date_str, violations)}
        notices.append("合规校验发现问题(compliance_block=False, 未阻断): "
                       + "; ".join(violations))

    report = render.build_report_dict(
        symbol, date_str, fused, sentiment, narrative,
        regime_info=regime_info, scenarios=scenarios, track_summary=track_summary,
    )
    simple_payload = simple_mod.build_simple_payload(fused, symbol, date_str)

    # L8 埋点: 预测落盘, 供后续准确率追踪
    record = {
        "schema_version": track_mod.SCHEMA_VERSION,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol, "date": date_str,
        "direction": fused["direction"], "confidence": fused["confidence"],
        **{k: v for k, v in fused["probs"].items()},
        "s_ml": fused["s_ml"], "s_llm": fused["s_llm"], "s_final": fused["s_final"],
        "llm_available": fused["llm_available"],
        # v2: 预测信息时点(T-1)的市场状态, 供 track 分状态结算
        "regime": regime_info["regime"] if regime_info["available"] else "未识别",
        "regime_as_of": regime_info["as_of"],
        # v3: 集成成员名与分歧度(单模型 disagreement=null)
        "model": predictor.name,
        "disagreement": fused["disagreement"],
        # v4: 交易状态与适用性(Phase 5)
        "trading_status": fused["trading_status"],
        "applicability": fused["applicability"],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"ok": True, "blocked": False, "notices": notices, "violations": [],
            "symbol": symbol, "date": date_str, "report": report,
            "simple": simple_payload, "fused": fused, "sentiment": sentiment,
            "narrative": narrative, "regime_info": regime_info, "scenarios": scenarios,
            "track_summary": track_summary, "record": record,
            "predictor_name": predictor.name}


def report_json_text(result: dict) -> str:
    """CLI --json 的输出文本; Web 直接展示 result["report"] —— 同一 dict, 构造一致."""
    return json.dumps(result["report"], ensure_ascii=False, indent=2)
